# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import functools

import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from packaging import version


def _preload_tensors_for_sync_save(write_buckets, non_blocking=True):
    """Stage device tensors while reusing CPU tensors for a synchronous save.

    MegatronAdaptor's async-safe preloader clones every CPU tensor so training
    cannot mutate optimizer state while a background writer is running.  A
    synchronous checkpoint blocks the training worker until all writer threads
    have joined, so that clone is unnecessary and can nearly double host memory
    when the optimizer itself is CPU-offloaded.
    """
    result = []
    for file_name, storage_key, (bytes_data, tensor_data) in write_buckets:
        staged_tensors = []
        for item, tensor in tensor_data:
            staged = tensor if tensor.is_cpu else tensor.to("cpu", non_blocking=non_blocking)
            staged_tensors.append((item, staged))
        result.append((file_name, storage_key, (bytes_data, staged_tensors)))
    if non_blocking:
        torch.cuda.synchronize()
    return result


class _MemoryEfficientSyncSaveStrategy(TorchDistSaveShardedStrategy):
    """Reuse CPU tensors only while executing a synchronous save request."""

    def save(self, sharded_state_dict, checkpoint_dir):
        request = self.async_save(sharded_state_dict, checkpoint_dir, async_strategy="mcore")
        preload = request.preload_fn
        if preload is not None:
            if not isinstance(preload, functools.partial) or not preload.args:
                raise RuntimeError("Unexpected Megatron checkpoint preload callback")
            non_blocking = preload.args[1] if len(preload.args) > 1 else True
            request = request._replace(
                preload_fn=functools.partial(
                    _preload_tensors_for_sync_save,
                    preload.args[0],
                    non_blocking,
                )
            )
        request.execute_sync()


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = _MemoryEfficientSyncSaveStrategy()
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata
    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = TorchDistLoadShardedStrategy()
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
