# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch

from fsdp_turbo.models.kimi.kimi_r3_utils import (
    _align_kimi_full_model_routes,
)
from verl.trainer.ppo.padding_utils import build_padding_full_r3_pair


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
