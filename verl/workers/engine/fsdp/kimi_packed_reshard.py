"""Target-layout resharding for Kimi K3 packed experts.

The training model stores Kimi routed experts as packed 3-D tensors and FSDP-
Turbo shards them on the expert dimension.  The rollout model may use a
different expert-parallel layout, so sending the source-local range directly
is only correct when both layouts happen to line up.

This module deliberately does not materialize the global packed tensor.  It
first replicates only non-expert DTensor placements (normally the eFSDP
matrix shard), then exchanges the required expert-row intervals with
``all_to_all_single``.  Every target rank receives exactly its local expert
range, including every rollout replica.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

KIMI_PACKED_LOCAL_MARKER = ".__verl_packed_local__."
_MESH_RANK_MAP_CACHE: dict[int, tuple[tuple[int, ...], dict[int, tuple[int, ...]]]] = {}


def export_kimi_source_local_param(name: str, param: DTensor, device):
    """Export the current source EP range for non-rollout callers.

    Checkpoint/delta consumers that do not provide a rollout layout retain the
    historical local-shard contract.  Online actor->vLLM sync always supplies
    :class:`KimiRolloutLayout` and therefore uses the target-layout path below.
    """

    placements = tuple(param.placements)
    expert_dims = [
        index
        for index, placement in enumerate(placements)
        if isinstance(placement, Shard) and placement.dim == 0
    ]
    if len(expert_dims) != 1:
        raise RuntimeError(
            f"Kimi source-local export requires exactly one Shard(0) placement for {name}, "
            f"got {placements}"
        )
    ep_dim = expert_dims[0]
    local_placements = tuple(
        placement if index == ep_dim else Replicate()
        for index, placement in enumerate(placements)
    )
    local_dtensor = param.to(device, non_blocking=True)
    if local_placements != placements:
        local_dtensor = local_dtensor.redistribute(param.device_mesh, local_placements)
    _, global_offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, local_placements)
    local_param = local_dtensor.to_local().detach().contiguous()
    if local_param.shape[0] == 0:
        raise RuntimeError(f"Kimi source-local export produced an empty expert shard for {name}")
    return f"{name}{KIMI_PACKED_LOCAL_MARKER}{int(global_offset[0])}", local_param


@dataclass(frozen=True)
class KimiRolloutLayout:
    """The layout visible to one synchronous vLLM rollout deployment."""

    expert_parallel_size: int
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        return (
            max(int(self.tensor_parallel_size or 1), 1)
            * max(int(self.data_parallel_size or 1), 1)
            * max(int(self.pipeline_model_parallel_size or 1), 1)
        )


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(value, 1)


def make_kimi_rollout_layout(
    expert_parallel_size: Any,
    tensor_parallel_size: Any = 1,
    data_parallel_size: Any = 1,
    pipeline_model_parallel_size: Any = 1,
) -> KimiRolloutLayout:
    return KimiRolloutLayout(
        expert_parallel_size=_positive_int(expert_parallel_size),
        tensor_parallel_size=_positive_int(tensor_parallel_size),
        data_parallel_size=_positive_int(data_parallel_size),
        pipeline_model_parallel_size=_positive_int(pipeline_model_parallel_size),
    )


def target_expert_interval(num_experts: int, target_ep: int, target_rank: int) -> tuple[int, int]:
    """Return the contiguous expert interval owned by a target EP rank."""

    num_experts = _positive_int(num_experts)
    target_ep = _positive_int(target_ep)
    target_rank = int(target_rank)
    if num_experts % target_ep != 0:
        raise ValueError(
            f"Kimi packed experts require num_experts divisible by rollout EP: "
            f"num_experts={num_experts}, rollout_ep={target_ep}"
        )
    local_experts = num_experts // target_ep
    start = target_rank * local_experts
    return start, start + local_experts


def _mesh_rank_map(mesh) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    """Return mesh shape and global-rank -> mesh-coordinate mapping."""

    cache_key = id(mesh)
    cached = _MESH_RANK_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mesh_tensor = mesh.mesh.detach().cpu()
    shape = tuple(int(size) for size in mesh_tensor.shape)
    rank_map: dict[int, tuple[int, ...]] = {}
    for coordinate in product(*(range(size) for size in shape)):
        rank = int(mesh_tensor[coordinate].item())
        rank_map[rank] = tuple(int(index) for index in coordinate)
    result = (shape, rank_map)
    _MESH_RANK_MAP_CACHE[cache_key] = result
    return result


def _canonical_source_ranks(mesh, ep_dim: int) -> dict[int, int]:
    """Choose one source rank for each source EP shard.

    After non-EP placements are changed to ``Replicate``, every rank with the
    same source EP coordinate has the same complete matrix rows.  The
    all-zero coordinate on all other mesh dimensions is deterministic and
    avoids a second metadata collective.
    """

    shape, _ = _mesh_rank_map(mesh)
    mesh_tensor = mesh.mesh.detach().cpu()
    source_ranks = {}
    for ep_rank in range(shape[ep_dim]):
        coordinate = [0] * len(shape)
        coordinate[ep_dim] = ep_rank
        source_ranks[ep_rank] = int(mesh_tensor[tuple(coordinate)].item())
    return source_ranks


def _source_ep_rank_by_global_rank(mesh, ep_dim: int) -> dict[int, int]:
    shape, rank_map = _mesh_rank_map(mesh)
    del shape
    return {rank: coordinate[ep_dim] for rank, coordinate in rank_map.items()}


def _validate_process_layout(layout: KimiRolloutLayout, world_size: int) -> tuple[int, int]:
    target_world_size = layout.world_size
    if target_world_size > world_size or world_size % target_world_size != 0:
        raise RuntimeError(
            "Kimi packed-local reshard requires the actor process group to contain an integer "
            "number of vLLM rollout replicas: "
            f"actor_world_size={world_size}, rollout_world_size={target_world_size} "
            f"(tp={layout.tensor_parallel_size}, dp={layout.data_parallel_size}, "
            f"pp={layout.pipeline_model_parallel_size})"
        )
    if target_world_size % layout.expert_parallel_size != 0:
        raise RuntimeError(
            "Kimi rollout world size must be divisible by rollout expert parallel size: "
            f"rollout_world_size={target_world_size}, rollout_ep={layout.expert_parallel_size}"
        )
    return target_world_size, world_size // target_world_size


def _direct_layout_matches(
    *,
    num_experts: int,
    source_ep: int,
    target_layout: KimiRolloutLayout,
    world_size: int,
    source_rank_by_ep: dict[int, int],
    source_ep_by_rank: dict[int, int],
) -> bool:
    """Whether every actor rank already maps to its target expert interval."""

    target_world_size, _ = _validate_process_layout(target_layout, world_size)
    if source_ep != target_layout.expert_parallel_size or num_experts % source_ep != 0:
        return False
    source_local = num_experts // source_ep
    target_local = num_experts // target_layout.expert_parallel_size
    if source_local != target_local:
        return False

    # The map is intentionally checked for all ranks.  A local-only check can
    # make different ranks choose different collective paths and deadlock.
    del source_rank_by_ep
    for rank in range(world_size):
        target_rank = rank % target_world_size
        target_ep_rank = target_rank % target_layout.expert_parallel_size
        if source_ep_by_rank.get(rank) != target_ep_rank:
            return False
    return True


def export_kimi_packed_local_param(
    name: str,
    param: DTensor,
    device,
    rollout_layout: KimiRolloutLayout,
):
    """Reshard one packed Kimi expert parameter to the rollout target layout."""

    if not isinstance(param, DTensor) or param.ndim != 3:
        raise TypeError(f"Kimi packed-local export expects a 3-D DTensor: {name}")

    placements = tuple(param.placements)
    expert_dims = [
        index
        for index, placement in enumerate(placements)
        if isinstance(placement, Shard) and placement.dim == 0
    ]
    if len(expert_dims) != 1:
        raise RuntimeError(
            f"Kimi packed-local export requires exactly one Shard(0) placement for {name}, "
            f"got {placements}"
        )
    ep_dim = expert_dims[0]
    source_ep = int(param.device_mesh.mesh.shape[ep_dim])
    num_experts = int(param.shape[0])
    if num_experts % source_ep != 0:
        raise RuntimeError(
            f"Kimi packed source experts are not evenly sharded: {name}, "
            f"num_experts={num_experts}, source_ep={source_ep}"
        )

    target_ep = rollout_layout.expert_parallel_size
    if num_experts % target_ep != 0:
        raise RuntimeError(
            f"Kimi packed target experts are not evenly sharded: {name}, "
            f"num_experts={num_experts}, rollout_ep={target_ep}"
        )

    # Materialize only the current source EP range.  Other placements (most
    # importantly eFSDP's matrix shard) are replicated within their mesh
    # groups, never across the complete EP dimension.
    local_placements = tuple(
        placement if index == ep_dim else Replicate()
        for index, placement in enumerate(placements)
    )
    local_dtensor = param.to(device, non_blocking=True)
    if local_placements != placements:
        local_dtensor = local_dtensor.redistribute(param.device_mesh, local_placements)
    _, global_offset = compute_local_shape_and_global_offset(
        param.shape, param.device_mesh, local_placements
    )
    local_param = local_dtensor.to_local().detach().contiguous()
    source_start = int(global_offset[0])
    source_local = num_experts // source_ep
    if local_param.shape[0] != source_local:
        raise RuntimeError(
            f"Unexpected local Kimi expert rows after eFSDP aggregation for {name}: "
            f"got={local_param.shape[0]}, expected={source_local}, placements={placements}"
        )

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Kimi EP-aware packed-local reshard requires an initialized torch.distributed group")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    target_world_size, replica_count = _validate_process_layout(rollout_layout, world_size)
    mesh_world_size = int(param.device_mesh.mesh.numel())
    if mesh_world_size != world_size:
        raise RuntimeError(
            f"Kimi packed-local reshard requires a full actor DeviceMesh: "
            f"mesh_world_size={mesh_world_size}, process_group_world_size={world_size}"
        )
    source_rank_by_ep = _canonical_source_ranks(param.device_mesh, ep_dim)
    source_ep_by_rank = _source_ep_rank_by_global_rank(param.device_mesh, ep_dim)

    if _direct_layout_matches(
        num_experts=num_experts,
        source_ep=source_ep,
        target_layout=rollout_layout,
        world_size=world_size,
        source_rank_by_ep=source_rank_by_ep,
        source_ep_by_rank=source_ep_by_rank,
    ):
        target_rank = rank % target_world_size
        target_ep_rank = target_rank % target_ep
        target_start, target_end = target_expert_interval(num_experts, target_ep, target_ep_rank)
        if source_start != target_start:
            raise RuntimeError(
                f"Kimi direct packed-local mapping disagrees with target layout for {name}: "
                f"source=[{source_start},{source_start + source_local}), "
                f"target=[{target_start},{target_end})"
            )
        return f"{name}{KIMI_PACKED_LOCAL_MARKER}{target_start}", local_param

    # Build a sparse all-to-all plan.  Only the canonical source rank for an
    # EP interval sends data; all other mesh dimensions already have the full
    # matrix rows after the local Replicate redistribution.  Each target PP
    # group and each rollout replica receives the required interval.
    row_numel = int(local_param[0].numel())
    source_ep_by_rank = source_ep_by_rank
    source_local = num_experts // source_ep
    target_local = num_experts // target_ep
    target_rank = rank % target_world_size
    target_ep_rank = target_rank % target_ep
    target_start, target_end = target_expert_interval(num_experts, target_ep, target_ep_rank)

    send_splits = [0] * world_size
    send_chunks: list[tuple[int, torch.Tensor]] = []
    current_source_ep = source_ep_by_rank.get(rank)
    if current_source_ep is not None and source_rank_by_ep.get(current_source_ep) == rank:
        current_source_start = current_source_ep * source_local
        current_source_end = current_source_start + source_local
        if source_start != current_source_start:
            raise RuntimeError(
                f"Kimi source mesh offset mismatch for {name}: "
                f"rank={rank}, placement_offset={source_start}, computed_offset={current_source_start}"
            )
        # Every DP/PP group inside one vLLM rollout world owns the same EP
        # expert interval.  Replicate to all of those groups, then repeat the
        # complete rollout world for every separate rollout replica.
        target_group_count = target_world_size // target_ep
        for replica in range(replica_count):
            for target_group_rank in range(target_group_count):
                destination_base = replica * target_world_size + target_group_rank * target_ep
                for destination_ep_rank in range(target_ep):
                    destination_start, destination_end = target_expert_interval(
                        num_experts, target_ep, destination_ep_rank
                    )
                    overlap_start = max(current_source_start, destination_start)
                    overlap_end = min(current_source_end, destination_end)
                    if overlap_start >= overlap_end:
                        continue
                    destination = destination_base + destination_ep_rank
                    overlap = local_param[
                        overlap_start - current_source_start : overlap_end - current_source_start
                    ]
                    send_chunks.append((destination, overlap.reshape(-1)))
                    send_splits[destination] += int(overlap.numel())

    # Receivers expect one interval from every canonical source shard that
    # intersects their target interval.  The split vectors are deterministic
    # from the mesh/layout metadata, so all ranks enter the same collective.
    recv_splits = [0] * world_size
    receive_sources: list[tuple[int, int, int]] = []
    for source_ep_rank in range(source_ep):
        source_start_i = source_ep_rank * source_local
        source_end_i = source_start_i + source_local
        overlap_start = max(source_start_i, target_start)
        overlap_end = min(source_end_i, target_end)
        if overlap_start >= overlap_end:
            continue
        source_rank = source_rank_by_ep[source_ep_rank]
        rows = overlap_end - overlap_start
        recv_splits[source_rank] = rows * row_numel
        receive_sources.append((source_rank, overlap_start, overlap_end))

    send_chunks.sort(key=lambda item: item[0])
    if send_chunks:
        send_buffer = torch.cat([chunk for _, chunk in send_chunks], dim=0).contiguous()
    else:
        send_buffer = torch.empty(0, dtype=local_param.dtype, device=local_param.device)
    recv_buffer = torch.empty(
        sum(recv_splits), dtype=local_param.dtype, device=local_param.device
    )

    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    target_param = torch.empty(
        (target_local, *tuple(local_param.shape[1:])),
        dtype=local_param.dtype,
        device=local_param.device,
    )
    filled = [False] * target_local
    offset = 0
    for source_rank, overlap_start, overlap_end in sorted(receive_sources):
        rows = overlap_end - overlap_start
        chunk_numel = rows * row_numel
        chunk = recv_buffer[offset : offset + chunk_numel].view(rows, *tuple(local_param.shape[1:]))
        target_offset = overlap_start - target_start
        target_param[target_offset : target_offset + rows].copy_(chunk)
        for index in range(target_offset, target_offset + rows):
            filled[index] = True
        offset += chunk_numel

    if offset != recv_buffer.numel() or not all(filled):
        missing = [index for index, value in enumerate(filled) if not value]
        raise RuntimeError(
            f"Kimi packed-local reshard did not fill target range for {name}: "
            f"target=[{target_start},{target_end}), missing_local_rows={missing[:8]}"
        )

    return f"{name}{KIMI_PACKED_LOCAL_MARKER}{target_start}", target_param
