"""Monkey-patch V1 advantage input/output for synthetic reward experiments."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from .reward import build_synthetic_rm_scores

logger = logging.getLogger("synthetic_reward_debug")

_FUNCTION_MARKER = "_synthetic_reward_debug_original"
_METHOD_MARKER = "_synthetic_reward_debug_original"
_PENDING: dict[tuple[str, ...], "PendingReward"] = {}


@dataclass
class PendingReward:
    scores: Any
    sequence_rewards: Any
    active_rows: Any


def _is_grpo(adv_estimator: Any) -> bool:
    return getattr(adv_estimator, "value", adv_estimator) == "grpo"


def _patch_advantage_function(settings: Any, module: Any) -> bool:
    original = module.compute_advantage_for_multi_trajectories
    if getattr(original, _FUNCTION_MARKER, False):
        return False

    def compute_advantage_for_multi_trajectories(
        data,
        batch_keys,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        if not _is_grpo(adv_estimator):
            raise RuntimeError("Synthetic reward debug requires algorithm.adv_estimator=grpo")
        if config is not None and config.get("use_kl_in_reward", False):
            raise RuntimeError("Synthetic reward debug requires algorithm.use_kl_in_reward=false")

        scores, sequence_rewards, active_rows = build_synthetic_rm_scores(
            batch_keys, data.batch["response_mask"], settings.scale
        )
        data.batch["rm_scores"] = scores
        data.batch["token_level_scores"] = scores
        data.batch["token_level_rewards"] = scores
        _PENDING[tuple(batch_keys)] = PendingReward(scores, sequence_rewards, active_rows)
        return original(
            data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    setattr(compute_advantage_for_multi_trajectories, _FUNCTION_MARKER, True)
    module.compute_advantage_for_multi_trajectories = compute_advantage_for_multi_trajectories
    return True


def _patch_trainer_method(settings: Any, module: Any) -> bool:
    trainer_cls = module.PPOTrainer
    original = trainer_cls._compute_advantage
    if getattr(original, _METHOD_MARKER, False):
        return False

    def compute_advantage(self, batch, metrics):
        key = tuple(batch.keys)
        try:
            result = original(self, batch, metrics)
        except Exception:
            _PENDING.pop(key, None)
            raise

        pending = _PENDING.pop(key, None)
        if pending is None:
            raise RuntimeError("Synthetic reward patch did not observe the V1 advantage computation")

        active_rewards = pending.sequence_rewards[pending.active_rows]
        metrics.update(
            {
                "debug/synthetic_reward/enabled": 1.0,
                "debug/synthetic_reward/scale": settings.scale,
                "debug/synthetic_reward/mean": active_rewards.mean().detach().item(),
                "debug/synthetic_reward/std": active_rewards.std(unbiased=False).detach().item(),
                "debug/synthetic_reward/max": active_rewards.max().detach().item(),
                "debug/synthetic_reward/min": active_rewards.min().detach().item(),
            }
        )

        response_mask = module.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask"],
        )["response_mask"]
        fields = module.TensorDict(
            {
                "rm_scores": module.response_to_nested(pending.scores, response_mask),
                "token_level_rewards": module.response_to_nested(pending.scores, response_mask),
            },
            batch_size=len(batch),
        )
        module.tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=fields)
        return result

    setattr(compute_advantage, _METHOD_MARKER, True)
    trainer_cls._compute_advantage = compute_advantage
    return True


def install_loaded(settings: Any, module_name: str, module: Any) -> bool:
    if module_name != "verl.trainer.ppo.v1.trainer_base":
        return False
    installed_function = _patch_advantage_function(settings, module)
    installed_method = _patch_trainer_method(settings, module)
    logger.warning(
        "[synthetic-reward-debug] patched process=%s module=%s function=%s method=%s scale=%s",
        os.getpid(),
        module_name,
        installed_function,
        installed_method,
        settings.scale,
    )
    return installed_function or installed_method
