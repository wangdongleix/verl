# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from verl.workers.rollout.vllm_rollout.routing_cache import RoutingCachePolicy


@pytest.mark.parametrize("aux_output", [False, True])
@pytest.mark.parametrize("routing_replay", [False, True])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_cache_policy_preserves_explicit_reuse_choice(aux_output, routing_replay, prefix_caching):
    engine_args = SimpleNamespace(__dataclass_fields__={"aux_output_config": None} if aux_output else {})
    policy = RoutingCachePolicy.from_config(
        prefix_caching=prefix_caching, routing_replay=routing_replay, engine_args_type=engine_args
    )
    first = {"prompt_token_ids": [1, 2, 3]}
    retry = dict(first)
    policy.prepare_prompt(first)
    policy.prepare_prompt(retry)
    if aux_output and routing_replay and not prefix_caching:
        assert policy.prefix_caching
        assert first["cache_salt"] != retry["cache_salt"]
    else:
        assert policy.prefix_caching == prefix_caching
        assert "cache_salt" not in first and "cache_salt" not in retry
    assert first["prompt_token_ids"] == retry["prompt_token_ids"] == [1, 2, 3]
