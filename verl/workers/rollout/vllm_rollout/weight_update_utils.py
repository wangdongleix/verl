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

import torch

from verl.utils.kimi_expert_stream import EXPERT_MARKER, LOCAL_MARKER

WeightUpdate = tuple[str, torch.Tensor]


def capture_expert_ownership(model) -> dict[str, frozenset[int] | None]:
    """Snapshot actual expert maps before layerwise reload moves buffers to meta."""
    ownership = {}
    if not hasattr(model, "named_modules"):
        return ownership
    for name, module in model.named_modules():
        if ".experts" not in name or not hasattr(module, "expert_map"):
            continue
        path = name.split(".experts", 1)[0] + ".experts"
        mapping = module.expert_map
        if mapping is not None:
            if mapping.is_meta:
                raise RuntimeError(f"Expert ownership must be captured before layerwise reload: {path}")
            experts = frozenset(i for i, local in enumerate(mapping.detach().cpu().tolist()) if local >= 0)
        else:
            experts = None  # EP=1: every expert is local.
        if path in ownership and ownership[path] != experts:
            raise RuntimeError(f"Conflicting expert maps for {path}")
        ownership[path] = experts
    return ownership


def localize_streamed_experts(weights, ownership):
    """Discard nonlocal experts before cloning or buffering deferred loader inputs."""
    for name, tensor in weights:
        if EXPERT_MARKER not in name:
            yield name, tensor
            continue
        base, encoded = name.rsplit(EXPERT_MARKER, 1)
        if not encoded.isdecimal() or tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Invalid streamed Kimi expert: {name}, shape={tuple(tensor.shape)}")
        expert = int(encoded)
        path = base.rsplit(".", 1)[0]
        matches = [key for key in ownership if key == path or key.endswith("." + path) or path.endswith("." + key)]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot resolve receiver expert ownership for {name}: {matches}")
        local = ownership[matches[0]]
        if local is None or expert in local:
            yield f"{base}{LOCAL_MARKER}{expert}", tensor


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
