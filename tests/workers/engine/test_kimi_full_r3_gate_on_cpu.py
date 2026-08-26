# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fsdp_turbo.models.kimi_k3.modeling.modeling_kimi_k3 import (
    _align_kimi_full_model_routes,
)
from fsdp_turbo.models.kimi_k3.modeling.modeling_kimi_linear import KimiMoEGate
from fsdp_turbo.models.kimi_k3.modeling.modeling_kimi_linear import (
    _assert_kimi_finite,
)
from verl.trainer.ppo.padding_utils import build_padding_full_r3_pair


def _gate_config():
    return SimpleNamespace(
        num_experts_per_token=2,
        num_experts=4,
        routed_scaling_factor=2.5,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        hidden_size=3,
    )


def test_gate_forward_uses_replayed_values_and_keeps_router_gradient():
    torch.manual_seed(0)
    gate = KimiMoEGate(_gate_config())
    gate.layer_idx = 7
    with torch.no_grad():
        gate.e_score_correction_bias.zero_()

    hidden = torch.randn(1, 2, 3, requires_grad=True)
    replay_ids = torch.tensor([[[0, 1], [2, 3]]])
    replay_weights = torch.tensor(
        [[[0.5, 2.0], [1.0, 1.5]]], dtype=torch.bfloat16
    )
    replay_mask = torch.tensor([[True, False]])

    selected_ids, selected_weights = gate(
        hidden,
        replay_topk_idx=replay_ids,
        replay_topk_weight=replay_weights,
        replay_mask=replay_mask,
    )

    torch.testing.assert_close(selected_ids[0], replay_ids.reshape(-1, 2)[0])
    torch.testing.assert_close(
        selected_weights[0], replay_weights.float().reshape(-1, 2)[0]
    )
    # A non-symmetric downstream sensitivity proves the replayed forward
    # values retain the local router derivative through the straight-through
    # expression.
    loss = (selected_weights[0] * torch.tensor([1.0, 3.0])).sum()
    loss.backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())
    assert float(hidden.grad[0, 0].abs().sum()) > 0.0
    assert gate.weight.grad is not None
    assert bool(torch.isfinite(gate.weight.grad).all())
    assert gate.e_score_correction_bias.grad is None


def test_full_model_alignment_moves_ids_weights_and_mask_together():
    routes = torch.arange(3 * 2 * 2).reshape(1, 3, 2, 2)
    weights = (routes.float() + 1).to(torch.bfloat16) / 16
    source_mask = torch.tensor([[True, True, True]])
    target_attention_mask = torch.tensor([[False, True, True, True, True]])

    aligned_ids, aligned_weights, aligned_mask = _align_kimi_full_model_routes(
        routes,
        weights,
        source_mask,
        target_attention_mask,
    )

    torch.testing.assert_close(aligned_ids[:, 1:4], routes)
    torch.testing.assert_close(aligned_weights[:, 1:4], weights)
    assert aligned_mask.tolist() == [[False, True, True, True, False]]
    assert not bool(aligned_ids[:, (0, 4)].any())
    assert not bool(aligned_weights[:, (0, 4)].any())


def test_full_r3_padding_reuses_one_valid_causal_route_row():
    source_ids = torch.tensor(
        [
            [[0, 0], [0, 0]],
            [[0, 1], [2, 3]],
            [[1, 2], [0, 3]],
        ],
        dtype=torch.uint8,
    )
    source_weights = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.25, 0.75], [0.5, 0.5]],
            [[0.5, 0.5], [0.25, 0.75]],
        ],
        dtype=torch.bfloat16,
    )

    padding_ids, padding_weights = build_padding_full_r3_pair(
        source_ids,
        source_weights,
        causal_rows=1,
    )

    assert padding_ids.shape == (1, 2, 2)
    torch.testing.assert_close(padding_ids[0], source_ids[1])
    torch.testing.assert_close(padding_weights[0], source_weights[1])
    aligned_ids, aligned_weights, aligned_mask = _align_kimi_full_model_routes(
        padding_ids.unsqueeze(0),
        padding_weights.unsqueeze(0),
        torch.tensor([[True]]),
        torch.tensor([[True, True]]),
    )
    assert aligned_mask.tolist() == [[True, False]]
    torch.testing.assert_close(aligned_ids[:, :1], padding_ids.unsqueeze(0))
    torch.testing.assert_close(
        aligned_weights[:, :1],
        padding_weights.unsqueeze(0),
    )


def test_r3_diagnostic_reports_nonfinite_backward_boundary(monkeypatch):
    monkeypatch.setenv("VERL_KIMI_R3_DIAG_ACTIVATIONS", "1")
    value = torch.ones(2, requires_grad=True)
    _assert_kimi_finite(value, layer_idx=11, stage="unit_test")

    with torch.autograd.set_detect_anomaly(False):
        try:
            (value * torch.tensor(float("nan"))).sum().backward()
        except FloatingPointError as exc:
            message = str(exc)
            assert "NONFINITE BACKWARD" in message
            assert "layer=11" in message
            assert "stage=unit_test" in message
        else:
            raise AssertionError("non-finite backward boundary was not detected")
