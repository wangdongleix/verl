# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.workers.engine.fsdp.kimi_packed_weights import export_kimi_packed_local_param
from verl.workers.engine.fsdp.utils import unfuse_moe_params


def test_kimi_keeps_offset_aware_packed_expert_transport_abi():
    tensor = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.4"

    updates = list(unfuse_moe_params([(name, tensor)], model_type="kimi_k3"))

    assert len(updates) == 1
    assert updates[0][0] == name
    assert updates[0][1] is tensor


def test_kimi_packed_transport_rejects_other_model_types():
    tensor = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.4"

    with pytest.raises(ValueError, match="valid only for the Kimi K3"):
        list(unfuse_moe_params([(name, tensor)], model_type="legacy"))


def test_kimi_packed_transport_rejects_invalid_offset_before_passthrough():
    tensor = torch.zeros(1, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.bad"

    with pytest.raises(ValueError, match="Invalid Kimi packed-local expert offset"):
        list(unfuse_moe_params([(name, tensor)], model_type="kimi_k3"))


def _check_colocated_export(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))
        original = torch.arange(4 * 6 * 8).reshape(4, 6, 8).float()
        layout = dict(
            expert_parallel_size=2, tensor_parallel_size=2, data_parallel_size=1, pipeline_model_parallel_size=1
        )
        # Two eFSDP matrix shards x two expert shards. Check initial weights
        # and a changed version, including delivery to both rollout replicas.
        for version in (0, 1):
            weights = original + version
            param = distribute_tensor(weights, mesh, [Shard(1), Shard(0)])
            name, local = export_kimi_packed_local_param("experts.gate_up_proj", param, "cpu", layout)
            start = (rank % 2) * 2
            assert name == f"experts.gate_up_proj.__verl_packed_local__.{start}"
            torch.testing.assert_close(local, weights[start : start + 2], rtol=0, atol=0)
            with pytest.raises(ValueError, match="matching actor/rollout EP"):
                export_kimi_packed_local_param(
                    "experts.gate_up_proj",
                    param,
                    "cpu",
                    {**layout, "expert_parallel_size": 4, "tensor_parallel_size": 4},
                )
        reordered = DeviceMesh("cpu", torch.tensor([[0, 2], [1, 3]]))
        param = distribute_tensor(original, reordered, [Shard(1), Shard(0)])
        with pytest.raises(ValueError, match="ownership does not match"):
            export_kimi_packed_local_param("experts.gate_up_proj", param, "cpu", layout)
    finally:
        dist.destroy_process_group()


def test_colocated_export_gathers_matrix_shards_and_keeps_expert_ownership(tmp_path):
    mp.spawn(_check_colocated_export, args=(str(tmp_path / "gloo_rendezvous"),), nprocs=4, join=True)
