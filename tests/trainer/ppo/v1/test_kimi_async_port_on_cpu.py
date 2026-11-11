# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Behavior checks for the Kimi separate-async migration."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.kimi_expert_stream import expert_tile_name, merge_expert_tiles
from verl.workers.engine.fsdp.kimi_packed_weights import stream_kimi_packed_param
from verl.workers.rollout.vllm_rollout.weight_update_utils import localize_streamed_experts


@pytest.mark.parametrize("chunk,restored,expected", [(8, 0, 96), (10, 0, 90), (8, 32, 64), (8, 128, 0)])
def test_fractional_warmup_respects_chunk_size_and_restored_prompts(chunk, restored, expected):
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": False}},
                "data": {"train_batch_size": 64, "gen_batch_size": chunk},
            }
        ),
        _restored_tq_prompt_count=restored,
        _add_prompts_to_generate=MagicMock(),
    )
    PPOTrainer._add_async_warmup_batches(trainer, 1.5)
    if expected:
        trainer._add_prompts_to_generate.assert_called_once_with(expected)
    else:
        trainer._add_prompts_to_generate.assert_not_called()


@pytest.mark.parametrize("strategy", ["drop", "wait"])
def test_null_staleness_threshold_keeps_old_samples_without_waiting(strategy):
    buffer = object.__new__(ReplayBufferAsync)
    buffer.max_off_policy_threshold = None
    buffer.max_off_policy_strategy = strategy
    buffer.prompt_global_steps = {"train": {"done": 0, "inflight": 0}}
    buffer.finished_keys = {"train": {"done"}}
    buffer.pending_keys = {"train": {"inflight"}}
    buffer.running_keys = {"train": set()}
    assert buffer._stale_terminal_keys(100, "train") == set()
    assert buffer._has_enough_samples(100, "train", 1, {"done"})
    buffer.max_off_policy_threshold = 2
    if strategy == "drop":
        assert buffer._stale_terminal_keys(100, "train") == {"done"}
    else:
        assert not buffer._has_enough_samples(100, "train", 1, {"done"})


def test_no_hybrid_mode_only_synchronizes_standalone_pool():
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer._enable_hybrid_replicas = False
    trainer.global_steps = 4
    trainer.standalone_checkpoint_manager = MagicMock()
    # No hybrid manager exists: any access to it would fail this test.
    trainer.on_init_end()
    trainer.switch_to_rollout()
    trainer.switch_to_trainer()
    trainer.add_replicas_to_balancer()
    trainer.remove_replicas_from_balancer()
    trainer.standalone_checkpoint_manager.update_weights.assert_called_once_with(4)


async def _merge(items):
    async def source():
        for item in items:
            yield item
    return [item async for item in merge_expert_tiles(source())]


@pytest.mark.parametrize("invalid", ["missing", "overlap", "missing_expert"])
def test_stream_rejects_incomplete_or_duplicate_weights(invalid):
    base = "model.layers.0.block_sparse_moe.experts.gate_up_proj"
    first = (expert_tile_name(base, 0, 2, 2, 0, 0, num_experts=2), torch.ones(1, 2))
    if invalid == "missing":
        items = [first]
    elif invalid == "overlap":
        items = [first, first]
    else:
        items = [(expert_tile_name(base, 0, 2, 2, 0, 0, num_experts=2), torch.ones(2, 2))]
    with pytest.raises(ValueError):
        asyncio.run(_merge(items))


def _stream_roundtrip(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))
        name = "model.layers.0.block_sparse_moe.experts.gate_up_proj"
        ownership = {"model.layers.0.block_sparse_moe.experts": frozenset(range(rank % 2, 8, 2))}
        for matrix_dimension in [1, 2]:
            for version in [0, 1]:
                full = torch.arange(8 * 5 * 7).reshape(8, 5, 7).float() + 17 * version
                param = distribute_tensor(full, mesh, [Shard(matrix_dimension), Shard(0)])
                with (
                    patch.object(DTensor, "full_tensor", side_effect=AssertionError("no full gather")),
                    patch.object(DTensor, "redistribute", side_effect=AssertionError("no redistribution")),
                ):
                    tiles = list(stream_kimi_packed_param(name, param, "cpu", max_tile_bytes=24))
                assert all(t.numel() * t.element_size() <= 24 for _, t in tiles)
                complete = asyncio.run(_merge(tiles))
                assert len(complete) == 8
                for expert, (_, value) in enumerate(complete):
                    torch.testing.assert_close(value, full[expert:expert + 1], rtol=0, atol=0)
                local = list(localize_streamed_experts(complete, ownership))
                assert len(local) == 4
                for (local_name, value), expert in zip(local, range(rank % 2, 8, 2), strict=True):
                    assert local_name.endswith(f".__verl_packed_local__.{expert}")
                    torch.testing.assert_close(value, full[expert:expert + 1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_stream_roundtrip_across_uneven_shards_versions_and_receiver_ownership(tmp_path):
    mp.spawn(_stream_roundtrip, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)


@pytest.mark.parametrize("bucket_bytes", [16, 32, 64])
def test_checkpoint_transport_roundtrip_with_reused_buckets(bucket_bytes):
    from verl.checkpoint_engine.base import merge_weight_chunks, split_weight_chunks

    base = "model.layers.0.block_sparse_moe.experts.down_proj"
    plain = torch.arange(37, dtype=torch.float32)
    experts = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    weights = [("ordinary.weight", plain)]
    for expert in range(2):
        for row in range(2):
            weights.append(
                (expert_tile_name(base, expert, 2, 3, row, 0, num_experts=2), experts[expert, row:row + 1])
            )

    async def roundtrip():
        async def recycled_transport():
            storage = torch.empty(bucket_bytes, dtype=torch.uint8)
            async for meta, chunk in split_weight_chunks(iter(weights), bucket_bytes):
                storage[:chunk.numel()].copy_(chunk)
                yield meta, storage[:chunk.numel()]
                storage.fill_(255)

        return [
            (name, value.clone())
            async for name, value in merge_weight_chunks(recycled_transport(), bucket_bytes)
        ]

    result = asyncio.run(roundtrip())
    assert [name for name, _ in result] == [
        "ordinary.weight",
        f"{base}.__verl_streamed_expert__.0",
        f"{base}.__verl_streamed_expert__.1",
    ]
    for (_, actual), expected in zip(result, [plain, experts[0:1], experts[1:2]], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
