# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in diagnostics for actor-to-rollout weight synchronization.

The diagnostic is intentionally environment-controlled so normal training does not
copy model weights to CPU or add per-tensor logging.  Enable it with::

    VERL_WEIGHT_SYNC_DEBUG=1

Useful optional settings are::

    VERL_WEIGHT_SYNC_DEBUG_NAMES='(embed_tokens|lm_head|experts\\.(0|831)\\.w[123]\\.weight)'
    VERL_WEIGHT_SYNC_DEBUG_TARGET_NAMES='(embed_tokens|lm_head|w13_weight|w2_weight)'
    VERL_WEIGHT_SYNC_DEBUG_STEPS=0,1,5-7
    VERL_WEIGHT_SYNC_DEBUG_MAX_TENSORS=0          # 0 (default) means unlimited
    VERL_WEIGHT_SYNC_DEBUG_HASH=sample       # sample (default) or full
    VERL_WEIGHT_SYNC_DEBUG_STATS=sample      # sample (default) or full
    VERL_WEIGHT_SYNC_DEBUG_SAMPLE_SIZE=4096
    VERL_WEIGHT_SYNC_DEBUG_OUTPUT=./weight_sync_debug.jsonl

``sample`` hashes deterministic positions from each tensor and avoids a full
device-to-host copy.  ``full`` hashes the complete tensor and should only be
used for a small number of tensors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)

Weight = tuple[str, torch.Tensor]
_FILE_ERROR_PATHS: set[str] = set()
_OUTPUT_FILES: dict[str, Any] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Return whether weight-sync diagnostics are enabled for this process."""

    is_enabled = _env_bool("VERL_WEIGHT_SYNC_DEBUG")
    if is_enabled:
        logger.setLevel(logging.INFO)
    return is_enabled


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default))), minimum)
    except ValueError:
        return default


def _compile_pattern(name: str) -> re.Pattern[str] | None:
    pattern = os.getenv(name, "").strip()
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        logger.warning("Ignoring invalid %s=%r: %s", name, pattern, exc)
        return None


def _parse_step_filter(value: str | None) -> tuple[tuple[int, int], ...] | None:
    """Parse ``N,M-K`` into inclusive step intervals.

    ``None`` means that no step filter was configured.  An explicitly configured
    value with no valid entries selects no steps, which is safer than logging all
    steps after a typo in the environment variable.
    """

    if value is None or not value.strip():
        return None

    intervals: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if re.fullmatch(r"\d+", item):
            step = int(item)
            intervals.append((step, step))
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", item)
        if match is None:
            logger.warning("Ignoring invalid step in VERL_WEIGHT_SYNC_DEBUG_STEPS=%r", item)
            continue
        start, end = (int(group) for group in match.groups())
        if start > end:
            logger.warning("Ignoring descending step range in VERL_WEIGHT_SYNC_DEBUG_STEPS=%r", item)
            continue
        intervals.append((start, end))
    return tuple(intervals)


def _step_is_selected(step_filter: tuple[tuple[int, int], ...] | None, global_steps: Any) -> bool:
    if step_filter is None:
        return True
    try:
        step = int(global_steps)
    except (TypeError, ValueError):
        return False
    return any(start <= step <= end for start, end in step_filter)


def _json_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(context or {})
    result.setdefault("pid", os.getpid())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        result.setdefault("rank", torch.distributed.get_rank())
    return result


def _sample_tensor(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Select deterministic positions without copying the full tensor to CPU."""

    flat = tensor.detach().contiguous().reshape(-1)
    if flat.numel() <= sample_size:
        return flat

    # Use integer arithmetic so this also works on accelerators without reliable
    # float64 index generation.  Include the final element explicitly.
    count = sample_size
    step = max((flat.numel() - 1) // (count - 1), 1)
    indices = torch.arange(count, device=flat.device, dtype=torch.long) * step
    indices[-1] = flat.numel() - 1
    return flat.index_select(0, indices)


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu_tensor = tensor.detach().contiguous().cpu()
    # NumPy does not support torch.bfloat16 directly.  Viewing the CPU storage
    # as bytes preserves the exact transferred representation for all dtypes.
    return cpu_tensor.view(torch.uint8).numpy().tobytes()


def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Return a bounded signature for a tensor.

    The default signature hashes deterministic samples.  Set
    ``VERL_WEIGHT_SYNC_DEBUG_HASH=full`` to hash the complete tensor.
    """

    sample_size = _env_int("VERL_WEIGHT_SYNC_DEBUG_SAMPLE_SIZE", 4096)
    hash_mode = os.getenv("VERL_WEIGHT_SYNC_DEBUG_HASH", "sample").lower()
    stats_mode = os.getenv("VERL_WEIGHT_SYNC_DEBUG_STATS", "sample").lower()
    if stats_mode not in {"sample", "full"}:
        logger.warning("Ignoring invalid VERL_WEIGHT_SYNC_DEBUG_STATS=%r; using sample", stats_mode)
        stats_mode = "sample"
    sampled = _sample_tensor(tensor, sample_size)
    hash_tensor = tensor if hash_mode == "full" else sampled
    sampled_float = sampled.float()
    stats_tensor = tensor.detach() if stats_mode == "full" else sampled
    stats_float = stats_tensor.float()
    has_stats = stats_float.numel() > 0

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "sample_numel": sampled.numel(),
        "sample_sum": sampled_float.sum().item(),
        "sample_absmax": sampled_float.abs().max().item() if sampled.numel() else 0.0,
        "sample_l2": sampled_float.square().sum().sqrt().item(),
        "stats_mode": stats_mode,
        "stats_numel": stats_float.numel(),
        "mean": stats_float.mean().item() if has_stats else 0.0,
        "min": stats_float.min().item() if has_stats else 0.0,
        "max": stats_float.max().item() if has_stats else 0.0,
        "sum": stats_float.sum().item(),
        "absmax": stats_float.abs().max().item() if has_stats else 0.0,
        "l2": stats_float.square().sum().sqrt().item(),
        "hash_mode": hash_mode if hash_mode == "full" else "sample",
        "sha256": hashlib.sha256(_raw_bytes(hash_tensor)).hexdigest(),
    }


def _output_path(record: Mapping[str, Any]) -> str | None:
    template = os.getenv("VERL_WEIGHT_SYNC_DEBUG_OUTPUT", "").strip()
    if not template:
        return None
    try:
        return template.format(
            pid=record.get("pid", "unknown"),
            rank=record.get("rank", "unknown"),
            stage=record.get("stage", "unknown"),
            global_steps=record.get("global_steps", "unknown"),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Ignoring invalid VERL_WEIGHT_SYNC_DEBUG_OUTPUT=%r: %s", template, exc)
        return None


def _emit_record(record: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(record), sort_keys=True, default=str)
    logger.info("[weight_sync_debug] %s", payload)

    path = _output_path(record)
    if path is None:
        return
    if path in _FILE_ERROR_PATHS:
        return
    try:
        output_file = _OUTPUT_FILES.get(path)
        if output_file is None:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            output_file = open(path, "a", encoding="utf-8")
            _OUTPUT_FILES[path] = output_file
        output_file.write(f"[weight_sync_debug] {payload}\n")
        output_file.flush()
    except OSError as exc:
        if path not in _FILE_ERROR_PATHS:
            _FILE_ERROR_PATHS.add(path)
            logger.warning("Cannot write weight-sync diagnostics to %r: %s", path, exc)


class _DebugSession:
    def __init__(self, stage: str, context: Mapping[str, Any] | None = None, target: bool = False):
        self.stage = stage
        self.context = _json_context(context)
        self.step_filter = _parse_step_filter(os.getenv("VERL_WEIGHT_SYNC_DEBUG_STEPS"))
        self.active = _step_is_selected(self.step_filter, self.context.get("global_steps"))
        self.pattern = _compile_pattern(
            "VERL_WEIGHT_SYNC_DEBUG_TARGET_NAMES" if target else "VERL_WEIGHT_SYNC_DEBUG_NAMES"
        )
        self.max_tensors = _env_int("VERL_WEIGHT_SYNC_DEBUG_MAX_TENSORS", 0, minimum=0)
        self.logged = 0

    def matches(self, name: str) -> bool:
        if not enabled() or not self.active or (self.max_tensors and self.logged >= self.max_tensors):
            return False
        return self.pattern is None or self.pattern.search(name) is not None

    def log(self, name: str, tensor: torch.Tensor, **extra: Any) -> None:
        if not self.matches(name):
            return
        self.logged += 1
        record = {
            "stage": self.stage,
            "name": name,
            "signature": tensor_signature(tensor),
            **self.context,
            **extra,
        }
        _emit_record(record)


def trace_weight_stream(
    weights: Iterable[Weight], stage: str, context: Mapping[str, Any] | None = None
) -> Iterable[Weight]:
    """Log selected tensors while preserving a possibly lazy weight stream."""

    if not enabled():
        return weights

    def _traced() -> Iterator[Weight]:
        session = _DebugSession(stage, context)
        for name, tensor in weights:
            session.log(name, tensor)
            yield name, tensor

    return _traced()


def log_received_weights(
    weights: Iterable[Weight], stage: str, context: Mapping[str, Any] | None = None
) -> None:
    """Log selected tensors already materialized by the vLLM receiver."""

    if not enabled():
        return
    session = _DebugSession(stage, context)
    if not session.active:
        return
    for name, tensor in weights:
        session.log(name, tensor)


def log_loaded_model_parameters(
    model: torch.nn.Module,
    loaded_names: Iterable[str] | None,
    stage: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Log checksums for target parameters reported by ``model.load_weights``."""

    if not enabled():
        return

    session = _DebugSession(stage, context, target=True)
    if not session.active:
        return

    if loaded_names is None:
        _emit_record({"stage": stage, "loaded_names": None, **_json_context(context)})
        return

    named_parameters = dict(model.named_parameters())
    missing = 0
    loaded_count = 0
    for name in loaded_names:
        loaded_count += 1
        if not isinstance(name, str):
            missing += 1
            continue
        target = named_parameters.get(name)
        if target is None:
            missing += 1
            continue
        session.log(name, target, target_parameter=True)

    _emit_record(
        {
            "stage": stage,
            "loaded_count": loaded_count,
            "target_missing_count": missing,
            **_json_context(context),
        }
    )
