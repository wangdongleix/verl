# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Strict Kimi full-R3 rollout payload helpers.

vLLM's routed-experts API has one integer tensor slot.  The Ascend worker
uses that slot to carry, for every top-k entry, the expert id followed by the
low and high bytes of the *executed BF16 router weight*.  Keeping the wire
payload integer-valued lets the existing vLLM slot manager retain its compact
uint8/uint16 storage without changing the EngineCore protocol.

The payload layout is ``[..., ids[K], weight_low_bytes[K],
weight_high_bytes[K]]``.  It is decoded at the verl boundary and never exposed
to the trainer as an overloaded tensor: downstream code receives explicit
``routed_experts`` and ``routed_expert_weights`` fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np

KIMI_FULL_R3_PACK_FACTOR = 3
_KIMI_FULL_R3_MODEL_TYPES = frozenset({"kimi_k3", "kimi_linear"})


def _config_candidates(hf_config: Any):
    yield hf_config
    for name in ("text_config", "language_config", "llm_config"):
        nested = getattr(hf_config, name, None)
        if nested is not None:
            yield nested


def is_kimi_full_r3_config(hf_config: Any) -> bool:
    """Whether *hf_config* uses Kimi's IDs+executed-weights R3 schema."""
    return any(
        str(getattr(candidate, "model_type", "")).lower()
        in _KIMI_FULL_R3_MODEL_TYPES
        for candidate in _config_candidates(hf_config)
    )


def get_kimi_full_r3_topk(hf_config: Any) -> int:
    """Return Kimi's real (unpacked) experts-per-token width."""
    candidates = list(_config_candidates(hf_config))
    model_types = {
        str(getattr(candidate, "model_type", "")).lower()
        for candidate in candidates
    }
    if not is_kimi_full_r3_config(hf_config):
        raise ValueError(
            "full routing replay with gate weights is currently implemented "
            f"only for Kimi K3, got model_types={sorted(model_types)!r}"
        )

    for candidate in candidates:
        for name in (
            "num_experts_per_token",
            "num_experts_per_tok",
            "top_k_experts",
        ):
            value = getattr(candidate, name, None)
            if value is not None and int(value) > 0:
                return int(value)
    raise ValueError(
        "Kimi full R3 requires a positive num_experts_per_token in the HF config"
    )


def decode_kimi_full_r3_payload(
    packed: Any,
    *,
    expected_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode vLLM's integer Kimi full-R3 payload without losing BF16 bits.

    Returns expert IDs and FP32 values.  Every BF16 value is exactly
    representable in FP32, so converting back to BF16 in the trainer is
    bit-exact.
    """
    payload = np.asarray(packed)
    if payload.ndim != 3:
        raise ValueError(
            "Kimi full R3 packed payload must be [model_rows, layers, 3*topk], "
            f"got {payload.shape!r}"
        )
    if expected_topk <= 0:
        raise ValueError(f"expected_topk must be positive, got {expected_topk}")
    expected_width = expected_topk * KIMI_FULL_R3_PACK_FACTOR
    if payload.shape[-1] != expected_width:
        raise ValueError(
            "Kimi full R3 payload width proves that rollout weight capture is "
            "not active or is incompatible: "
            f"got={payload.shape[-1]}, expected={expected_width} "
            f"(topk={expected_topk})"
        )
    if not np.issubdtype(payload.dtype, np.integer):
        raise TypeError(
            "Kimi full R3 packed payload must use an integer transport dtype, "
            f"got {payload.dtype}"
        )

    expert_ids = np.array(payload[..., :expected_topk], copy=True)
    low = payload[..., expected_topk : 2 * expected_topk]
    high = payload[..., 2 * expected_topk :]
    if low.size:
        low_min, low_max = int(low.min()), int(low.max())
        high_min, high_max = int(high.min()), int(high.max())
        if low_min < 0 or low_max > 255 or high_min < 0 or high_max > 255:
            raise ValueError(
                "Kimi full R3 BF16 byte lanes are out of range: "
                f"low=[{low_min}, {low_max}], high=[{high_min}, {high_max}]"
            )

    bf16_bits = low.astype(np.uint16) | (high.astype(np.uint16) << 8)
    fp32_bits = bf16_bits.astype(np.uint32) << 16
    weights = np.array(fp32_bits.view(np.float32), copy=True)
    if not np.isfinite(weights).all():
        raise ValueError("Kimi full R3 decoded router weights contain NaN or Inf")
    if (weights < 0).any():
        raise ValueError("Kimi full R3 decoded router weights contain negative values")
    return expert_ids, weights
