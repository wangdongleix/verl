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
import torch

KIMI_FULL_R3_PACK_FACTOR = 3
KIMI_FULL_MODEL_ROUTE_SEMANTICS = "kimi_natural_rollout_full_model_rows_v1"
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


def validate_kimi_full_model_routes(
    routes: torch.Tensor,
    route_weights: torch.Tensor,
    hf_config: Any,
    *,
    row_locations: torch.Tensor | None = None,
) -> None:
    """Validate complete Kimi route rows at rollout and actor boundaries."""
    if routes.dim() != 3 or route_weights.shape != routes.shape:
        raise ValueError(
            "Kimi full-model R3 shape mismatch: "
            f"ids={tuple(routes.shape)}, weights={tuple(route_weights.shape)}"
        )
    if not route_weights.is_floating_point():
        raise TypeError(
            "Kimi full-model route weights must be floating point, got "
            f"{route_weights.dtype}"
        )
    if row_locations is not None and (
        row_locations.dim() != 2
        or row_locations.shape[0] != routes.shape[0]
        or row_locations.shape[1] != 2
    ):
        raise ValueError(
            "Kimi full-model row locations must be [rows, 2], got "
            f"{tuple(row_locations.shape)}"
        )

    text_config = getattr(hf_config, "text_config", hf_config)
    expected_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    num_experts = int(getattr(text_config, "num_experts", 0) or 0)
    expected_topk = get_kimi_full_r3_topk(hf_config)
    first_dense = int(getattr(text_config, "first_k_dense_replace", 0) or 0)
    moe_frequency = int(getattr(text_config, "moe_layer_freq", 1) or 1)
    if expected_layers <= 0 or num_experts <= 0:
        raise ValueError(
            "Kimi R3 requires positive num_hidden_layers and num_experts"
        )
    if routes.shape[1:] != (expected_layers, expected_topk):
        raise ValueError(
            "Kimi route capture shape does not match the model config: "
            f"routes={tuple(routes.shape)}, layers={expected_layers}, "
            f"topk={expected_topk}"
        )
    if routes.shape[0] == 0:
        return

    moe_layers = [
        layer_idx
        for layer_idx in range(expected_layers)
        if layer_idx >= first_dense and layer_idx % moe_frequency == 0
    ]
    if not moe_layers:
        return
    moe_routes = routes[:, moe_layers, :].to(dtype=torch.long)
    moe_weights = route_weights[:, moe_layers, :].to(dtype=torch.float32)

    def location(row: int) -> str:
        if row_locations is None:
            return f"row={row}"
        sample, route_row = row_locations[row].tolist()
        return f"sample={sample}, route_row={route_row}"

    bad_range = (moe_routes < 0) | (moe_routes >= num_experts)
    if bool(bad_range.any()):
        bad = torch.nonzero(bad_range, as_tuple=False)[0]
        raise ValueError(
            "Kimi route capture contains an out-of-range expert ID: "
            f"{location(int(bad[0]))}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}, "
            f"value={int(moe_routes[tuple(bad.tolist())])}, "
            f"num_experts={num_experts}"
        )

    ordered = torch.sort(moe_routes, dim=-1).values
    duplicate = ordered[..., 1:] == ordered[..., :-1]
    if bool(duplicate.any()):
        bad = torch.nonzero(duplicate, as_tuple=False)[0]
        row, layer = int(bad[0]), int(bad[1])
        raise ValueError(
            "Kimi route capture contains duplicate top-k expert IDs: "
            f"{location(row)}, layer={moe_layers[layer]}, "
            f"routes={moe_routes[row, layer].tolist()}"
        )

    finite = torch.isfinite(moe_weights)
    if not bool(finite.all()):
        bad = torch.nonzero(~finite, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weight is NaN/Inf: "
            f"{location(int(bad[0]))}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}"
        )
    if bool((moe_weights < 0).any()):
        bad = torch.nonzero(moe_weights < 0, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weight is negative: "
            f"{location(int(bad[0]))}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}, "
            f"value={float(moe_weights[tuple(bad.tolist())])}"
        )

    weight_sums = moe_weights.sum(dim=-1)
    if bool((weight_sums <= 0).any()):
        bad = torch.nonzero(weight_sums <= 0, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weights sum to zero: "
            f"{location(int(bad[0]))}, layer={moe_layers[int(bad[1])]}"
        )
    if not bool(getattr(text_config, "moe_renormalize", False)):
        return

    expected_sum = float(getattr(text_config, "routed_scaling_factor", 1.0))
    tolerance = max(abs(expected_sum) * 0.02, 0.02)
    close = torch.isclose(
        weight_sums,
        torch.full_like(weight_sums, expected_sum),
        rtol=0.0,
        atol=tolerance,
    )
    if not bool(close.all()):
        bad = torch.nonzero(~close, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weights violate the normalized sum contract: "
            f"{location(int(bad[0]))}, layer={moe_layers[int(bad[1])]}, "
            f"actual={float(weight_sums[tuple(bad.tolist())])}, "
            f"expected={expected_sum}, tolerance={tolerance}"
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
