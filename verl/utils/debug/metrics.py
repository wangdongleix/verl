# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2
    valid_diff_mask = diff_mask & (mask == 1)
    return valid_diff_mask.sum(dim=1)


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto) -> dict:
    """Calculate rollout-vs-actor probability and log-probability differences."""

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
            "training/rollout_log_probs_diff_mean": float("nan"),
            "training/rollout_log_probs_abs_diff_mean": float("nan"),
            "training/rollout_log_probs_signed_diff_mean": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    log_prob_signed_diff = rollout_old_log_probs - actor_old_log_probs
    log_prob_abs_diff = torch.abs(log_prob_signed_diff)
    valid_log_prob_abs_diff = torch.masked_select(log_prob_abs_diff, response_mask_bool)
    valid_log_prob_signed_diff = torch.masked_select(log_prob_signed_diff, response_mask_bool)
    log_prob_abs_diff_mean = valid_log_prob_abs_diff.mean().detach().item()
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        # Backward-compatible, directly greppable key.  It intentionally means
        # mean absolute log-probability difference; the explicit abs/signed
        # metrics below remove any ambiguity for new consumers.
        "training/rollout_log_probs_diff_mean": log_prob_abs_diff_mean,
        "training/rollout_log_probs_abs_diff_mean": log_prob_abs_diff_mean,
        "training/rollout_log_probs_abs_diff_p50": torch.quantile(valid_log_prob_abs_diff, 0.50).detach().item(),
        "training/rollout_log_probs_abs_diff_p90": torch.quantile(valid_log_prob_abs_diff, 0.90).detach().item(),
        "training/rollout_log_probs_abs_diff_p99": torch.quantile(valid_log_prob_abs_diff, 0.99).detach().item(),
        "training/rollout_log_probs_signed_diff_mean": valid_log_prob_signed_diff.mean().detach().item(),
    }

    # A recurrent-decode mismatch should grow with response position, whereas
    # a static weight or logits-kernel mismatch is already visible in the first
    # bucket. Fixed 64-token buckets keep runs with different response limits
    # directly comparable.
    positions = torch.arange(response_length, device=response_mask_bool.device).unsqueeze(0)
    for start in range(0, response_length, 64):
        end = min(start + 64, response_length)
        bucket_mask = response_mask_bool & (positions >= start) & (positions < end)
        if not bucket_mask.any():
            continue
        bucket_abs = torch.masked_select(log_prob_abs_diff, bucket_mask)
        bucket_signed = torch.masked_select(log_prob_signed_diff, bucket_mask)
        key = f"{start:04d}_{end - 1:04d}"
        metrics[f"training/rollout_log_probs_abs_diff_pos_{key}"] = bucket_abs.mean().detach().item()
        metrics[f"training/rollout_log_probs_signed_diff_pos_{key}"] = bucket_signed.mean().detach().item()

    return metrics
