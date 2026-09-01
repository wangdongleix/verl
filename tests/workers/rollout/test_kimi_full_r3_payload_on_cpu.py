# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.workers.rollout.r3_utils import (
    decode_kimi_full_r3_payload,
    get_kimi_full_r3_topk,
    is_kimi_full_r3_config,
)


def _pack_exact_bf16(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exercise the same BF16 byte split used by the Ascend capturer."""
    ids_tensor = torch.from_numpy(ids.astype(np.int32))
    weights_tensor = torch.from_numpy(weights).to(torch.bfloat16)
    weight_bytes = weights_tensor.contiguous().view(torch.uint8).reshape(
        *weights_tensor.shape, 2
    )
    packed = torch.cat(
        (
            ids_tensor,
            weight_bytes[..., 0].to(torch.int32),
            weight_bytes[..., 1].to(torch.int32),
        ),
        dim=-1,
    )
    return packed.numpy()


def test_full_r3_payload_round_trips_ids_and_executed_bf16_weights():
    ids = np.array(
        [
            [[[1, 7], [2, 6]], [[3, 5], [0, 4]]],
            [[[7, 1], [6, 2]], [[5, 3], [4, 0]]],
        ],
        dtype=np.uint16,
    ).reshape(4, 2, 2)
    # These values are exactly representable in BF16, so the expected result
    # is unambiguous and bit-preserving through FP32 decode.
    weights = np.array(
        [0.25, 0.5, 1.0, 2.0] * 4,
        dtype=np.float32,
    ).reshape(ids.shape)

    decoded_ids, decoded_weights = decode_kimi_full_r3_payload(
        _pack_exact_bf16(ids, weights),
        expected_topk=2,
    )

    np.testing.assert_array_equal(decoded_ids, ids)
    np.testing.assert_array_equal(decoded_weights, weights)


def test_id_only_payload_is_rejected_instead_of_silently_downgrading_r3():
    ids = np.zeros((3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="weight capture is not active"):
        decode_kimi_full_r3_payload(ids, expected_topk=2)


def test_corrupt_weight_byte_lane_is_rejected():
    payload = np.zeros((1, 1, 6), dtype=np.int32)
    payload[..., 2] = 256
    with pytest.raises(ValueError, match="byte lanes are out of range"):
        decode_kimi_full_r3_payload(payload, expected_topk=2)


def test_topk_is_resolved_from_nested_kimi_text_config():
    config = SimpleNamespace(
        model_type="kimi_k3",
        text_config=SimpleNamespace(
            model_type="kimi_linear",
            num_experts_per_token=8,
        ),
    )
    assert get_kimi_full_r3_topk(config) == 8
    assert is_kimi_full_r3_config(config)


def test_non_kimi_config_keeps_legacy_id_only_schema():
    config = SimpleNamespace(
        model_type="qwen3_moe",
        num_experts_per_tok=8,
    )
    assert not is_kimi_full_r3_config(config)
    with pytest.raises(ValueError, match="only for Kimi K3"):
        get_kimi_full_r3_topk(config)
