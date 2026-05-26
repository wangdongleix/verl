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

import logging
import os

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from verl.trainer.config import CheckpointConfig
from verl.utils.model import print_model_size
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.config import (
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
    MindSpeedOptimizerConfig,
    MindSpeedEngineConfig,
)

from ..base import EngineRegistry, BaseEngine

try:
    from ..megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead
except ImportError:
    MegatronEngineWithLMHead = BaseEngine
    MegatronEngineWithValueHead = BaseEngine

try:
    from ..fsdp import FSDPEngineWithLMHead
except ImportError:
    FSDPEngineWithLMHead = BaseEngine

from .utils import (
    apply_patch,
    gpt_model_provider,
    reset_fp8_reuse_quantized_weight,
    fsdp_turbo_module,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _mindspeed_repatch(engine_config):
    if repatch is not None:
        repatch_config = dict(engine_config.get("override_transformer_config", {}))
        repatch_config.setdefault("use_flash_attn", True)
        if engine_config.context_parallel_size > 1:
            repatch_config["context_parallel_size"] = engine_config.context_parallel_size
        repatch(repatch_config)


@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        reset_fp8_reuse_quantized_weight(self, device, model, optimizer, grad)
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)


@EngineRegistry.register(model_type="value_model", backend="megatron", device="npu")
class MindspeedEngineWithValueHead(MegatronEngineWithValueHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()


@EngineRegistry.register(model_type="language_model", backend="mindspeed_megatron", device="npu")
class MindSpeedMegatronEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MindSpeedEngineConfig,
        optimizer_config: MindSpeedOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        apply_patch(self.model_config, self.engine_config, self.optimizer_config)
        super()._init_device_mesh()

    def _build_megatron_module(self):
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )

        self.is_value_model = is_value_model

        import torch.distributed
        from megatron.core.enums import ModelType
        from megatron.training.training import get_model

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
        else:
            module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)

        if self.vanilla_bridge:
            self.bridge.load_weights(module, self.model_config.local_path)
        else:
            raise ValueError(f"vanilla_bridge should be true now, but got {self.vanilla_bridge}")

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            from verl.utils.megatron.router_replay_patch import RouterReplay
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        reset_fp8_reuse_quantized_weight(self, device, model, optimizer, grad)
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)


@EngineRegistry.register(model_type="language_model", backend="mindspeed_fsdp", device="npu")
class MindSpeedFSDPEngineWithLMHead(FSDPEngineWithLMHead):
    def __init__(
            self,
            model_config: HFModelConfig,
            engine_config: MindSpeedEngineConfig,
            optimizer_config: MindSpeedOptimizerConfig,
            checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _build_model_optimizer(self):
        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Apply QAT before FSDP wrapping (training only)
        if self._qat_enabled and not self.engine_config.forward_only:
            module = self._apply_qat(module)

        # Synchronize all distributed processes before proceeding
        import torch.distributed
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        full_state = module.state_dict()
        module = fsdp_turbo_module(self.engine_config.fsdp_kwargs, module)
        # full_state = {f"model.{k}": v for k, v in full_state.items()}
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict
        fsdp2_load_full_state_dict(module, full_state)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    # def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
    #     per_tensor_param, peft_config_dict = super().get_per_tensor_param(
    #         layered_summon=layered_summon, base_sync_done=base_sync_done, **kwargs
    #     )
    #     # 去掉fsdp turbo引入的"model."前缀，适配vllm期望格式
    #     per_tensor_param = (
    #         (name[len("model."):], param) if name.startswith("model.") else (name, param)
    #         for name, param in per_tensor_param
    #     )
    #     return per_tensor_param, peft_config_dict
