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

import contextlib
import functools
import logging
import os

import torch


def _install_kimi_fp32_router(module):
    """Run Kimi's router outside NPU autocast to preserve FP32 semantics."""
    for gate in module.modules():
        if gate.__class__.__name__ != "KimiMoEGate":
            continue
        if getattr(gate, "_verl_kimi_fp32_router", False):
            continue

        original_forward = gate.forward

        @functools.wraps(original_forward)
        def _forward(hidden_states, _original=original_forward):
            # KimiMoEGate casts its operands to FP32, but NPU autocast can
            # still cast F.linear's result back to BF16.  Disable autocast
            # only for the gate; the rest of the model remains BF16.
            if hidden_states.device.type == "npu":
                autocast = torch.autocast(device_type="npu", enabled=False)
            else:
                autocast = contextlib.nullcontext()
            with autocast:
                return _original(hidden_states)

        gate.forward = _forward
        gate._verl_kimi_fp32_router = True


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
    MindSpeedEngineConfig,
    MindSpeedOptimizerConfig,
)

from ..base import BaseEngine, EngineRegistry

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
    apply_clip_grad_norm_patch,
    apply_patch,
    gpt_model_provider,
    reset_fp8_reuse_quantized_weight,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_kimi_moe_external_state_dict_export(module, enabled):
    """Temporarily select the Kimi-MoE state-dict ABI.

    Kimi-K3 stores each MoE layer's expert weights in two packed parameters
    (``gate_up_proj`` and ``down_proj``), while the FSDP-Turbo model patch
    exposes the HuggingFace-style virtual keys
    ``experts.<id>.w{1,2,3}.weight`` after expert-parallel parameters are
    registered.  The initial FSDP2 load must use one ABI consistently on both
    sides of ``set_model_state_dict``.  Return the previous values so callers
    can restore the mode after the load.
    """

    previous = []
    for submodule in module.modules():
        # Do not use hasattr/getattr here: PatchKimiMoeExperts implements a
        # dynamic __getattr__ for virtual expert modules and that can create
        # misleading attribute lookups while walking the module tree.
        if submodule.__class__.__name__ != "PatchKimiMoeExperts":
            continue
        if "_export_external_state_dict" not in submodule.__dict__:
            continue
        old_value = submodule.__dict__["_export_external_state_dict"]
        previous.append((submodule, old_value))
        submodule.__dict__["_export_external_state_dict"] = enabled
    return previous


def _restore_kimi_moe_external_state_dict_export(previous):
    for submodule, old_value in previous:
        submodule.__dict__["_export_external_state_dict"] = old_value


def _mindspeed_repatch(engine_config):
    if repatch is not None:
        from verl.utils.megatron_utils import mapping_string_to_attn_backend

        repatch_config = mapping_string_to_attn_backend(dict(engine_config.get("override_transformer_config", {})))
        # flash-attn-npu batch-invariant replaces DotProductAttention.forward; fusion attention
        # registers the same patch when use_flash_attn=True and causes "the patch of forward exist".
        if repatch_config.get("use_flash_attn_npu_batch_invariant"):
            repatch_config["use_flash_attn"] = False
        else:
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
        from fsdp_turbo.distributed.parallel_state import get_parallel_state, init_parallel_state
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass

        self.fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, self.engine_config.fsdp_kwargs)
        attn_implementation = self.fsdp_turbo_config.model.attn_implementation
        # A VLM can intentionally use a different attention implementation in
        # its encoder and decoder.  In particular, Kimi-K3 rollout executes the
        # vision tower with vLLM's fused MMEncoderAttention, while the language
        # model remains on eager attention.  Propagating the text setting to the
        # vision sub-config silently changes the training-side vision graph.
        vision_attn_implementation = os.getenv(
            "VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION",
            attn_implementation,
        )
        for config in (
            self.model_config.hf_config,
            getattr(self.model_config.hf_config, "text_config", None),
        ):
            if config is not None:
                config._attn_implementation = attn_implementation
        vision_config = getattr(self.model_config.hf_config, "vision_config", None)
        if vision_config is not None:
            vision_config._attn_implementation = vision_attn_implementation
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

        from verl.utils.fsdp_utils import CPUOffloadPolicy, fsdp2_load_full_state_dict

        # ``from_pretrained`` invokes PatchKimiMoeExperts._load_from_state_dict
        # and marks the just-loaded module as external-export capable.  Reset
        # that flag *before* taking the snapshot; otherwise ``full_state`` is
        # already in the per-expert ABI while the FSDP2 local model is still
        # represented by the packed parameters.
        _set_kimi_moe_external_state_dict_export(module, enabled=False)
        full_state = module.state_dict()
        # convert_model_dtype(module, self.fsdp_turbo_config.model.torch_dtype)
        offload_policy = CPUOffloadPolicy(pin_memory=True) if self.engine_config.offload_policy else None
        self._uses_fsdp2_cpu_offload_policy = offload_policy is not None
        module = FSDPTurbo(
            self.fsdp_turbo_config,
            module,
            offload_policy=offload_policy,
        ).model
        # FSDP-Turbo's EP registration changes Kimi-MoE modules from their
        # packed checkpoint ABI to virtual per-expert keys.  ``full_state``
        # was captured immediately before that conversion, so loading it
        # while the converted model advertises external keys makes DCP look
        # for e.g. ``experts.0.w1.weight`` in a state dict that only contains
        # ``experts.gate_up_proj``/``down_proj`` (KeyError during broadcast).
        # Keep the converted model in packed export mode for this one load,
        # then restore the mode selected by FSDP-Turbo for subsequent syncs.
        previous_kimi_moe_export = _set_kimi_moe_external_state_dict_export(module, enabled=False)
        try:
            fsdp2_load_full_state_dict(module, full_state, cpu_offload=offload_policy)
        finally:
            _restore_kimi_moe_external_state_dict_export(previous_kimi_moe_export)

        # Install after FSDP-Turbo has replaced expert parameters so the
        # wrapper is attached to the actual gate used by the actor.
        _install_kimi_fp32_router(module)

        return module

    def get_data_parallel_rank(self):
        if self._parallel_state is not None:
            return self._parallel_state.get_data_parallel_rank()
        return super().get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self._parallel_state is not None:
            return self._parallel_state.get_data_parallel_size()
        return super().get_data_parallel_size()

    def get_data_parallel_group(self):
        if self._parallel_state is not None:
            return self._parallel_state.get_data_parallel_group()
        return super().get_data_parallel_group()

    def get_context_parallel_group(self):
        if self._parallel_state is not None:
            return self._parallel_state.get_cp_group()
        return super().get_context_parallel_group()

    def is_mp_src_rank_with_outputs(self):
        if self._parallel_state is not None:
            is_cp_src = self._parallel_state.get_rank("cp") == 0
            is_tp_src = not self._parallel_state.is_group_enable("tp") or self._parallel_state.get_rank("tp") == 0
            return is_cp_src and is_tp_src
        return super().is_mp_src_rank_with_outputs()
