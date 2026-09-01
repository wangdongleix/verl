# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
