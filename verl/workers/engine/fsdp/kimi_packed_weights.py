"""Export packed Kimi experts for the colocated FSDP-Turbo rollout layout."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

KIMI_PACKED_LOCAL_MARKER = ".__verl_packed_local__."


def export_kimi_packed_local_param(name: str, param: DTensor, device, rollout_layout: dict):
    """Gather eFSDP matrix shards while retaining each rank's expert interval.

    The Kimi example uses matching actor EP and rollout TP/EP, with DP=PP=1
    inside each rollout replica. Validate that contract on the complete mesh
    before any redistribution so every rank takes the same collective path.
    """
    if not isinstance(param, DTensor) or param.ndim != 3:
        raise TypeError(f"Kimi packed export requires a 3-D DTensor: {name}")
    if rollout_layout is None:
        raise ValueError("Kimi packed export requires the rollout layout")
    ep = int(rollout_layout["expert_parallel_size"])
    tp = int(rollout_layout["tensor_parallel_size"])
    dp = int(rollout_layout["data_parallel_size"])
    pp = int(rollout_layout["pipeline_model_parallel_size"])
    if ep <= 0 or tp != ep or dp != 1 or pp != 1:
        raise ValueError("Kimi packed export requires rollout TP=EP>0 and DP=PP=1")

    placements = param.placements
    expert_dims = [
        index for index, placement in enumerate(placements) if isinstance(placement, Shard) and placement.dim == 0
    ]
    if len(expert_dims) != 1:
        raise ValueError(f"Kimi packed export requires one expert Shard(0): {placements}")
    expert_dim = expert_dims[0]
    mesh = param.device_mesh.mesh.cpu()
    world_size = dist.get_world_size()
    if mesh.numel() != world_size or mesh.shape[expert_dim] != ep or world_size % ep:
        raise ValueError("Kimi packed export requires matching actor/rollout EP on the full actor mesh")
    coordinate_shape = [1] * mesh.ndim
    coordinate_shape[expert_dim] = ep
    expected_ranks = torch.arange(ep).reshape(coordinate_shape).expand(mesh.shape)
    if not torch.equal(mesh.remainder(ep), expected_ranks):
        raise ValueError("Kimi actor expert ownership does not match colocated rollout ranks")
    if param.shape[0] % ep:
        raise ValueError(f"Kimi experts must be divisible by EP: experts={param.shape[0]}, EP={ep}")

    export_placements = tuple(
        placement if index == expert_dim else Replicate() for index, placement in enumerate(placements)
    )
    exported = param.to(device, non_blocking=True)
    if export_placements != placements:
        exported = exported.redistribute(param.device_mesh, export_placements)
    _, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, export_placements)
    local = exported.to_local().detach().contiguous()
    expected_start = (dist.get_rank() % ep) * (param.shape[0] // ep)
    if offset[0] != expected_start or local.shape[0] != param.shape[0] // ep:
        raise RuntimeError(f"Unexpected Kimi local expert interval: {name}, offset={offset}, shape={local.shape}")
    return f"{name}{KIMI_PACKED_LOCAL_MARKER}{expected_start}", local
