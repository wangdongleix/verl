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
    convert_model_dtype,
    apply_clip_grad_norm_patch,
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
        apply_clip_grad_norm_patch()

    def _init_device_mesh(self):
        self._parallel_state = None
        super()._init_device_mesh()
        self._init_parallel_state()

    def _init_parallel_state(self):
        from omegaconf import OmegaConf
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass
        from fsdp_turbo.distributed.parallel_state import init_parallel_state, get_parallel_state

        fsdp_kwargs = self.engine_config.fsdp_kwargs
        if OmegaConf.is_config(fsdp_kwargs):
            fsdp_kwargs = OmegaConf.to_container(fsdp_kwargs, resolve=True)
        self.fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, fsdp_kwargs)
        init_parallel_state(self.fsdp_turbo_config)
        self._parallel_state = get_parallel_state()

        self._turbo_cp_enabled = self._parallel_state.get_ulysses_group_size() > 1
        if self._turbo_cp_enabled and not self.use_remove_padding:
            raise ValueError(
                "FSDP-Turbo CP for Qwen3.5 currently requires "
                "actor_rollout_ref.model.use_remove_padding=True so that verl's "
                "existing CP output path can gather local log-probs."
            )
        if self._turbo_cp_enabled and self.ulysses_sequence_parallel_size > 1:
            raise ValueError(
                "Do not enable both FSDP-Turbo CP and verl Ulysses SP. "
                "Use fsdp_kwargs.distributed.ulysses_parallel_size for Turbo CP "
                "and set ulysses_sequence_parallel_size=1."
            )
        if self._turbo_cp_enabled:
            # Reuse the parent FSDP engine's CP input/output lifecycle, but make
            # FSDP-Turbo the owner of the actual Ulysses process group.
            self.ulysses_sequence_parallel_size = self._parallel_state.get_ulysses_group_size()
            self.ulysses_parallel_group = self._parallel_state.get_ulysses_group()
            self.use_ulysses_sp = True

    def _build_module(self):
        # Do not let verl's Qwen VLM monkey patch slice the text model before
        # FSDP-Turbo's post-fusion model patch does the CP split.
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return super()._build_module()
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if self._turbo_cp_enabled:
            model_inputs["_fsdp_turbo_post_fusion_ulysses"] = True
        return model_inputs, output_args

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

        full_state = module.state_dict()
        convert_model_dtype(module, self.fsdp_turbo_config.model.torch_dtype)
        module = FSDPTurbo(self.fsdp_turbo_config, module).model
        fsdp2_load_full_state_dict(module, full_state)
        return module

    def _is_ep_enabled(self):
        return (
            self._parallel_state is not None
            and self._parallel_state.is_group_enable("ep")
        )

    def get_data_parallel_rank(self):
        if self._is_ep_enabled():
            return self._parallel_state.get_rank("edp")
        if self._parallel_state is not None:
            if hasattr(self._parallel_state, "get_data_parallel_rank"):
                return self._parallel_state.get_data_parallel_rank()
            fsdp_size = self._parallel_state.get_group_size("fsdp")
            return self._parallel_state.get_rank("dp") * fsdp_size + self._parallel_state.get_rank("fsdp")
        return super().get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self._is_ep_enabled():
            return self._parallel_state.get_group_size("edp")
        if self._parallel_state is not None:
            if hasattr(self._parallel_state, "get_data_parallel_size"):
                return self._parallel_state.get_data_parallel_size()
            return self._parallel_state.get_group_size("dp") * self._parallel_state.get_group_size("fsdp")
        return super().get_data_parallel_size()

    def get_data_parallel_group(self):
        if self._is_ep_enabled():
            return self._parallel_state.get_group("edp")
        if self._parallel_state is not None:
            if hasattr(self._parallel_state, "get_data_parallel_group"):
                return self._parallel_state.get_data_parallel_group()
            return self._parallel_state.get_group("dp_fsdp")
        return super().get_data_parallel_group()

    def get_context_parallel_group(self):
        if self._parallel_state is not None:
            return self._parallel_state.get_cp_group()
        return super().get_context_parallel_group()

    def is_mp_src_rank_with_outputs(self):
        if self._parallel_state is not None:
            is_cp_src = self._parallel_state.get_rank("ulysses") == 0
            is_ep_src = not self._is_ep_enabled() or self._parallel_state.get_rank("ep") == 0
            is_tp_src = not self._parallel_state.is_group_enable("tp") or self._parallel_state.get_rank("tp") == 0
            return is_cp_src and is_ep_src and is_tp_src
        return super().is_mp_src_rank_with_outputs()
