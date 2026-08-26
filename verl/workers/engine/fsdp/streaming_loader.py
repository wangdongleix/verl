"""Streaming safetensors loading for Kimi-K3 FSDP-Turbo models.

The legacy Kimi loader first assembles every expert into a CPU full state dict
and lets FSDP-Turbo broadcast/shard that state.  This module builds the final
DTensor layout first and reads only the slice owned by the current rank.

The loader intentionally does not call ``state_dict`` or ``full_tensor``.
Those operations are both correctness hazards for a large EP model and the
source of the multi-hundred-GB initialization peak seen with the old path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

logger = logging.getLogger(__name__)


@dataclass
class _TensorSpec:
    name: str
    tensor: torch.Tensor
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_tensor: torch.Tensor


@dataclass
class _LoadTask:
    target_name: str
    source_key: str
    destination: torch.Tensor
    destination_slices: tuple[slice, ...]
    source_slices: tuple[slice, ...]
    expected_source_shape: tuple[int, ...]
    transpose: bool = False
    allow_prefix_crop: bool = False


@dataclass
class StreamingLoadReport:
    checkpoint_path: str
    rank: int
    target_parameters: int = 0
    loaded_parameters: int = 0
    target_local_numel: int = 0
    loaded_local_numel: int = 0
    loaded_bytes: int = 0
    files_opened: int = 0
    tasks: int = 0
    packed_parameters: int = 0
    missing_parameters: list[str] = field(default_factory=list)
    missing_buffers: list[str] = field(default_factory=list)
    dtype_conversions: list[str] = field(default_factory=list)
    elapsed_materialize_s: float = 0.0
    elapsed_plan_s: float = 0.0
    elapsed_io_s: float = 0.0

    @property
    def parameter_coverage(self) -> float:
        return self.loaded_parameters / self.target_parameters if self.target_parameters else 1.0

    @property
    def local_numel_coverage(self) -> float:
        return self.loaded_local_numel / self.target_local_numel if self.target_local_numel else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "rank": self.rank,
            "target_parameters": self.target_parameters,
            "loaded_parameters": self.loaded_parameters,
            "target_local_numel": self.target_local_numel,
            "loaded_local_numel": self.loaded_local_numel,
            "parameter_coverage": self.parameter_coverage,
            "local_numel_coverage": self.local_numel_coverage,
            "loaded_bytes": self.loaded_bytes,
            "files_opened": self.files_opened,
            "tasks": self.tasks,
            "packed_parameters": self.packed_parameters,
            "missing_parameters": self.missing_parameters,
            "missing_buffers": self.missing_buffers,
            "dtype_conversions": self.dtype_conversions,
            "elapsed_materialize_s": self.elapsed_materialize_s,
            "elapsed_plan_s": self.elapsed_plan_s,
            "elapsed_io_s": self.elapsed_io_s,
        }


def _normalise_name(name: str) -> str:
    """Remove wrappers that are not part of the HF checkpoint ABI."""

    return name.replace("_fsdp_wrapped_module.", "")


def _as_shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _as_offset(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _local_spec(name: str, tensor: torch.Tensor) -> _TensorSpec:
    if isinstance(tensor, DTensor):
        local_tensor = tensor.to_local()
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tensor.shape,
            tensor.device_mesh,
            tensor.placements,
        )
        local_shape = _as_shape(local_shape)
        global_offset = _as_offset(global_offset)
    else:
        local_tensor = tensor
        local_shape = _as_shape(tensor.shape)
        global_offset = (0,) * len(local_shape)

    global_shape = _as_shape(tensor.shape)
    if _as_shape(local_tensor.shape) != local_shape:
        raise RuntimeError(
            f"Local DTensor shape mismatch for {name}: "
            f"computed={local_shape}, actual={tuple(local_tensor.shape)}, "
            f"global={global_shape}"
        )
    if any(offset < 0 or offset + size > global_size for offset, size, global_size in zip(
        global_offset, local_shape, global_shape
    )):
        raise RuntimeError(
            f"Invalid local DTensor range for {name}: "
            f"offset={global_offset}, local={local_shape}, global={global_shape}"
        )
    return _TensorSpec(
        name=name,
        tensor=tensor,
        global_shape=global_shape,
        local_shape=local_shape,
        global_offset=global_offset,
        local_tensor=local_tensor,
    )


def _named_targets(model: torch.nn.Module) -> tuple[dict[str, _TensorSpec], dict[str, _TensorSpec]]:
    parameters: dict[str, _TensorSpec] = {}
    buffers: dict[str, _TensorSpec] = {}
    for name, tensor in model.named_parameters():
        normalised = _normalise_name(name)
        if normalised in parameters:
            raise RuntimeError(f"Duplicate normalized parameter name: {name} -> {normalised}")
        parameters[normalised] = _local_spec(name, tensor)
    for name, tensor in model.named_buffers():
        normalised = _normalise_name(name)
        if normalised in buffers:
            raise RuntimeError(f"Duplicate normalized buffer name: {name} -> {normalised}")
        buffers[normalised] = _local_spec(name, tensor)
    return parameters, buffers


def _snapshot_buffers(model: torch.nn.Module):
    """Save initialized buffers before ``to_empty`` materializes parameters.

    ``init_empty_weights`` leaves non-persistent rotary/cache buffers on CPU.
    ``Module.to_empty`` would otherwise replace them with uninitialized data.
    They are restored after parameter materialization and are overwritten from
    the checkpoint when a persistent checkpoint buffer exists.
    """

    snapshots = {}
    for name, tensor in model.named_buffers():
        if getattr(tensor, "is_meta", False):
            continue
        if isinstance(tensor, DTensor):
            value = tensor.to_local().detach().cpu().clone()
        else:
            value = tensor.detach().cpu().clone()
        snapshots[name] = value
    return snapshots


def _restore_buffers(model: torch.nn.Module, snapshots) -> None:
    if not snapshots:
        return
    current_buffers = dict(model.named_buffers())
    for name, saved in snapshots.items():
        current = current_buffers.get(name)
        if current is None:
            continue
        if isinstance(current, DTensor):
            destination = current.to_local()
        else:
            destination = current
        if getattr(destination, "is_meta", False):
            raise RuntimeError(f"Buffer remained meta after local materialization: {name}")
        if tuple(destination.shape) != tuple(saved.shape):
            raise RuntimeError(
                f"Buffer shape changed during local materialization for {name}: "
                f"current={tuple(destination.shape)}, saved={tuple(saved.shape)}"
            )
        destination.copy_(saved.to(device=destination.device, dtype=destination.dtype))


@torch.no_grad()
def _materialize_local_parameters(model: torch.nn.Module, device) -> None:
    snapshots = _snapshot_buffers(model)
    try:
        model.to_empty(device=device)
    except Exception as exc:
        raise RuntimeError(
            "FSDP-Turbo streaming load could not materialize empty local parameters. "
            "The model must be fully sharded before the streaming loader runs."
        ) from exc
    _restore_buffers(model, snapshots)


def _slice_tuple(offset: tuple[int, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(start, start + size) for start, size in zip(offset, shape))


def _intersection(start_a: int, end_a: int, start_b: int, end_b: int) -> tuple[int, int] | None:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return (start, end) if start < end else None


def _packed_prefix(target_name: str, suffix: str) -> str:
    marker = f".experts.{suffix}"
    if not target_name.endswith(marker):
        raise RuntimeError(f"Unexpected packed Kimi parameter name: {target_name}")
    return target_name[: -len(marker)] + ".experts."


def _add_task(
    tasks_by_file: dict[str, list[_LoadTask]],
    weight_map: dict[str, str],
    task: _LoadTask,
) -> None:
    shard_name = weight_map.get(task.source_key)
    if shard_name is None:
        raise KeyError(task.source_key)
    tasks_by_file[shard_name].append(task)


def _add_direct_task(
    spec: _TensorSpec,
    source_key: str,
    weight_map: dict[str, str],
    tasks_by_file,
    *,
    allow_prefix_crop: bool = False,
) -> int:
    source_slices = _slice_tuple(spec.global_offset, spec.local_shape)
    destination_slices = tuple(slice(None) for _ in spec.local_shape)
    _add_task(
        tasks_by_file,
        weight_map,
        _LoadTask(
            target_name=spec.name,
            source_key=source_key,
            destination=spec.local_tensor,
            destination_slices=destination_slices,
            source_slices=source_slices,
            expected_source_shape=spec.global_shape,
            allow_prefix_crop=allow_prefix_crop,
        ),
    )
    return math.prod(spec.local_shape)


def _add_kimi_packed_tasks(
    spec: _TensorSpec,
    weight_map: dict[str, str],
    tasks_by_file,
    strict: bool,
) -> tuple[int, int]:
    """Create local tasks for raw Kimi w1/w2/w3 expert tensors.

    Returns ``(written_local_numel, task_count)``.  The destination is the
    packed Kimi ABI: gate_up is [expert, hidden, 2 * intermediate], down is
    [expert, intermediate, hidden].
    """

    shape = spec.global_shape
    if len(shape) != 3:
        raise RuntimeError(f"Packed Kimi parameter must be 3-D: {spec.name} shape={shape}")
    num_experts = shape[0]
    expert_width = shape[1]
    packed_width = shape[2]
    expert_start, expert_end = spec.global_offset[0], spec.global_offset[0] + spec.local_shape[0]
    written = 0
    task_count = 0

    if spec.name.endswith(".experts.gate_up_proj"):
        if packed_width % 2:
            raise RuntimeError(f"Packed gate_up width is not even: {spec.name} shape={shape}")
        intermediate_size = packed_width // 2
        prefix = _packed_prefix(spec.name, "gate_up_proj")
        axis1_start = spec.global_offset[1]
        axis1_end = axis1_start + spec.local_shape[1]
        axis2_start = spec.global_offset[2]
        axis2_end = axis2_start + spec.local_shape[2]
        source_shape = (intermediate_size, expert_width)
        for expert_id in range(num_experts):
            if not (expert_start <= expert_id < expert_end):
                continue
            local_expert = expert_id - expert_start
            for projection, packed_start, packed_end in (
                ("w1", 0, intermediate_size),
                ("w3", intermediate_size, packed_width),
            ):
                packed_intersection = _intersection(axis2_start, axis2_end, packed_start, packed_end)
                if packed_intersection is None:
                    continue
                hidden_intersection = _intersection(axis1_start, axis1_end, 0, expert_width)
                if hidden_intersection is None:
                    continue
                dst_axis2_start, dst_axis2_end = packed_intersection
                dst_axis1_start, dst_axis1_end = hidden_intersection
                source_key = f"{prefix}{expert_id}.{projection}.weight"
                if source_key not in weight_map:
                    raise KeyError(source_key)
                source_inter_start = dst_axis2_start - packed_start
                source_inter_end = dst_axis2_end - packed_start
                destination_slices = (
                    slice(local_expert, local_expert + 1),
                    slice(dst_axis1_start - axis1_start, dst_axis1_end - axis1_start),
                    slice(dst_axis2_start - axis2_start, dst_axis2_end - axis2_start),
                )
                source_slices = (
                    slice(source_inter_start, source_inter_end),
                    slice(dst_axis1_start, dst_axis1_end),
                )
                task = _LoadTask(
                    target_name=spec.name,
                    source_key=source_key,
                    destination=spec.local_tensor,
                    destination_slices=destination_slices,
                    source_slices=source_slices,
                    expected_source_shape=source_shape,
                    transpose=True,
                )
                _add_task(tasks_by_file, weight_map, task)
                written += (dst_axis1_end - dst_axis1_start) * (dst_axis2_end - dst_axis2_start)
                task_count += 1
    elif spec.name.endswith(".experts.down_proj"):
        intermediate_size = shape[1]
        hidden_size = shape[2]
        prefix = _packed_prefix(spec.name, "down_proj")
        axis1_start = spec.global_offset[1]
        axis1_end = axis1_start + spec.local_shape[1]
        axis2_start = spec.global_offset[2]
        axis2_end = axis2_start + spec.local_shape[2]
        source_shape = (hidden_size, intermediate_size)
        for expert_id in range(num_experts):
            if not (expert_start <= expert_id < expert_end):
                continue
            local_expert = expert_id - expert_start
            intermediate_intersection = _intersection(axis1_start, axis1_end, 0, intermediate_size)
            hidden_intersection = _intersection(axis2_start, axis2_end, 0, hidden_size)
            if intermediate_intersection is None or hidden_intersection is None:
                continue
            dst_axis1_start, dst_axis1_end = intermediate_intersection
            dst_axis2_start, dst_axis2_end = hidden_intersection
            source_key = f"{prefix}{expert_id}.w2.weight"
            if source_key not in weight_map:
                raise KeyError(source_key)
            destination_slices = (
                slice(local_expert, local_expert + 1),
                slice(dst_axis1_start - axis1_start, dst_axis1_end - axis1_start),
                slice(dst_axis2_start - axis2_start, dst_axis2_end - axis2_start),
            )
            source_slices = (
                slice(dst_axis2_start, dst_axis2_end),
                slice(dst_axis1_start, dst_axis1_end),
            )
            task = _LoadTask(
                target_name=spec.name,
                source_key=source_key,
                destination=spec.local_tensor,
                destination_slices=destination_slices,
                source_slices=source_slices,
                expected_source_shape=source_shape,
                transpose=True,
            )
            _add_task(tasks_by_file, weight_map, task)
            written += (dst_axis1_end - dst_axis1_start) * (dst_axis2_end - dst_axis2_start)
            task_count += 1
    else:
        raise RuntimeError(f"Unsupported packed Kimi parameter: {spec.name}")

    expected = math.prod(spec.local_shape)
    if written != expected:
        message = f"Packed local coverage mismatch for {spec.name}: written={written}, expected={expected}"
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    return written, task_count


def _read_index(checkpoint_path: str) -> tuple[dict[str, str], str]:
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        single_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.isfile(single_path):
            # Read only metadata; no tensor payload is materialized.
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                return {key: "model.safetensors" for key in handle.keys()}, "model.safetensors"
        raise FileNotFoundError(f"No safetensors index or single checkpoint found in {checkpoint_path}")
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid safetensors weight_map in {index_path}")
    return {str(key): str(value) for key, value in weight_map.items()}, ""


def _check_shape(task: _LoadTask, source_shape: tuple[int, ...]) -> None:
    if task.allow_prefix_crop:
        if len(source_shape) != len(task.expected_source_shape) or any(
            source_size < target_size
            for axis, (source_size, target_size) in enumerate(zip(source_shape, task.expected_source_shape))
            if axis == 0
        ) or any(
            source_size != target_size
            for axis, (source_size, target_size) in enumerate(zip(source_shape, task.expected_source_shape))
            if axis != 0
        ):
            raise RuntimeError(
                f"Checkpoint crop shape mismatch for {task.source_key} -> {task.target_name}: "
                f"checkpoint={source_shape}, target={task.expected_source_shape}"
            )
    elif source_shape != task.expected_source_shape:
        raise RuntimeError(
            f"Checkpoint shape mismatch for {task.source_key} -> {task.target_name}: "
            f"checkpoint={source_shape}, expected={task.expected_source_shape}"
        )
    for axis, item in enumerate(task.source_slices):
        start = 0 if item.start is None else item.start
        stop = source_shape[axis] if item.stop is None else item.stop
        if start < 0 or stop < start or stop > source_shape[axis]:
            raise RuntimeError(
                f"Source slice out of range for {task.source_key}: slices={task.source_slices}, shape={source_shape}"
            )


def _copy_task(task: _LoadTask, accessor, report: StreamingLoadReport) -> None:
    source_shape = _as_shape(accessor.get_shape())
    _check_shape(task, source_shape)
    source = accessor[task.source_slices] if task.source_slices else accessor[()]
    if task.transpose:
        if source.ndim != 2:
            raise RuntimeError(f"Transpose task is not 2-D: {task.source_key} shape={tuple(source.shape)}")
        source = source.transpose(0, 1).contiguous()
    destination = task.destination[task.destination_slices] if task.destination_slices else task.destination
    if task.transpose and source.ndim + 1 == destination.ndim and destination.shape[0] == 1:
        source = source.unsqueeze(0)
    if tuple(source.shape) != tuple(destination.shape):
        raise RuntimeError(
            f"Local slice shape mismatch for {task.source_key} -> {task.target_name}: "
            f"source={tuple(source.shape)}, destination={tuple(destination.shape)}"
        )
    source_bytes = source.numel() * source.element_size()
    if source.dtype != destination.dtype:
        report.dtype_conversions.append(
            f"{task.source_key}: {source.dtype} -> {destination.dtype}"
        )
        source = source.to(dtype=destination.dtype)
    if source.device != destination.device:
        source = source.to(device=destination.device, non_blocking=True)
    destination.copy_(source)
    report.loaded_bytes += source_bytes


@torch.no_grad()
def load_kimi_k3_checkpoint_to_local_shards(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = True,
    materialize_device=None,
) -> StreamingLoadReport:
    """Load raw Kimi checkpoint weights directly into local DTensor shards.

    ``model`` must already have the final FSDP-Turbo/TP/EP wrappers applied.
    The raw checkpoint may contain more layers or experts than the configured
    model; only target model parameters are planned and loaded.
    """

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    report = StreamingLoadReport(checkpoint_path=checkpoint_path, rank=rank)
    total_start = time.perf_counter()
    materialize_start = time.perf_counter()
    if materialize_device is None:
        materialize_device = torch.device("cpu")
        if torch.accelerator.is_available():
            materialize_device = torch.accelerator.current_device()
    _materialize_local_parameters(model, materialize_device)
    report.elapsed_materialize_s = time.perf_counter() - materialize_start

    weight_map, single_shard = _read_index(checkpoint_path)
    plan_start = time.perf_counter()
    parameters, buffers = _named_targets(model)
    report.target_parameters = len(parameters)
    report.target_local_numel = sum(math.prod(spec.local_shape) for spec in parameters.values())
    tasks_by_file: dict[str, list[_LoadTask]] = defaultdict(list)
    loaded_parameter_names: set[str] = set()
    packed_count = 0

    for target_name, spec in parameters.items():
        source_key = target_name
        if source_key in weight_map:
            allow_prefix_crop = target_name.endswith(
                (".block_sparse_moe.gate.weight", ".block_sparse_moe.gate.e_score_correction_bias", ".self_attn.A_log")
            )
            _add_direct_task(
                spec,
                source_key,
                weight_map,
                tasks_by_file,
                allow_prefix_crop=allow_prefix_crop,
            )
            loaded_parameter_names.add(target_name)
            continue

        if target_name.endswith((".experts.gate_up_proj", ".experts.down_proj")) and len(spec.global_shape) == 3:
            written, _ = _add_kimi_packed_tasks(spec, weight_map, tasks_by_file, strict)
            if written == math.prod(spec.local_shape):
                loaded_parameter_names.add(target_name)
            packed_count += 1
            continue

        report.missing_parameters.append(target_name)

    for target_name, spec in buffers.items():
        if target_name in weight_map:
            _add_direct_task(spec, target_name, weight_map, tasks_by_file)
        else:
            report.missing_buffers.append(target_name)

    report.packed_parameters = packed_count
    report.elapsed_plan_s = time.perf_counter() - plan_start
    report.tasks += sum(len(tasks) for tasks in tasks_by_file.values())

    if report.missing_parameters and strict:
        raise RuntimeError(
            "Streaming Kimi checkpoint is missing target parameters: "
            + ", ".join(report.missing_parameters[:32])
        )

    io_start = time.perf_counter()
    opened_files = 0
    for shard_name in sorted(tasks_by_file):
        shard_path = shard_name if os.path.isabs(shard_name) else os.path.join(checkpoint_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            opened_files += 1
            available_keys = set(handle.keys())
            for task in tasks_by_file[shard_name]:
                if task.source_key not in available_keys:
                    raise KeyError(f"Checkpoint key {task.source_key} is not in shard {shard_path}")
                _copy_task(task, handle.get_slice(task.source_key), report)
    report.files_opened = opened_files
    report.elapsed_io_s = time.perf_counter() - io_start

    # A packed target has many expert tasks, but still counts as one parameter.
    report.loaded_parameters = len(loaded_parameter_names)
    report.loaded_local_numel = sum(
        math.prod(spec.local_shape)
        for name, spec in parameters.items()
        if name in loaded_parameter_names
    )

    if report.loaded_parameters != report.target_parameters or report.loaded_local_numel != report.target_local_numel:
        message = (
            "Streaming Kimi checkpoint coverage failed: "
            f"parameters={report.loaded_parameters}/{report.target_parameters}, "
            f"local_numel={report.loaded_local_numel}/{report.target_local_numel}, "
            f"missing={report.missing_parameters[:8]}"
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)

    if report.dtype_conversions:
        logger.warning("Streaming checkpoint applied %d dtype conversions", len(report.dtype_conversions))

    logger.info(
        "Streaming Kimi checkpoint loaded: rank=%d files=%d tasks=%d parameters=%d/%d "
        "local_numel=%d/%d bytes=%d materialize=%.2fs plan=%.2fs io=%.2fs total=%.2fs",
        report.rank,
        report.files_opened,
        report.tasks,
        report.loaded_parameters,
        report.target_parameters,
        report.loaded_local_numel,
        report.target_local_numel,
        report.loaded_bytes,
        report.elapsed_materialize_s,
        report.elapsed_plan_s,
        report.elapsed_io_s,
        time.perf_counter() - total_start,
    )
    return report
