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
import logging
import os
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import (
    KIMI_FULL_MODEL_ROUTE_SEMANTICS,
    AgentLoopBase,
    AgentLoopOutput,
    _validate_kimi_full_model_routes,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.r3_utils import is_kimi_full_r3_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self._kimi_full_r3 = bool(
            self.model_config is not None
            and is_kimi_full_r3_config(self.model_config.hf_config)
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        if use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                [],
                [] if output.log_probs else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        if output.routed_experts is None and output.routed_expert_weights is not None:
            raise RuntimeError(
                "router weights cannot be transported without expert IDs"
            )
        if (
            self._kimi_full_r3
            and output.routed_experts is not None
            and output.routed_expert_weights is None
        ):
            raise RuntimeError(
                "Kimi full R3 rollout returned expert IDs without executed "
                "router weights"
            )
        if self._kimi_full_r3 and output.routed_experts is not None and len(response_ids) > self.response_length:
            raise ValueError(
                "Kimi full-model routing replay cannot truncate generated "
                "tokens after route capture: "
                f"generated={len(response_ids)}, configured={self.response_length}"
            )
        if self._kimi_full_r3 and output.routed_experts is not None:
            if self.model_config is None:
                raise RuntimeError(
                    "Kimi full-model routing replay requires the worker's "
                    "materialized HF model config"
                )
            _validate_kimi_full_model_routes(
                torch.as_tensor(output.routed_experts),
                torch.as_tensor(output.routed_expert_weights),
                self.model_config.hf_config,
            )
            if not getattr(self, "_kimi_full_r3_transport_logged", False):
                route_shape = tuple(output.routed_experts.shape)
                marker = (
                    "Kimi full R3 agent-loop TRANSPORT ACTIVE: "
                    f"rows={route_shape[0]} layers={route_shape[1]} "
                    f"topk={route_shape[2]} ids_and_weights=True"
                )
                logger.warning(marker)
                print(marker, flush=True)
                self._kimi_full_r3_transport_logged = True

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            # Kimi's vLLM route rows are the complete natural-rollout model
            # axis: every expanded prompt row followed by generated-token
            # predecessor rows. Preserve all of them for full R3 replay.
            routed_experts=output.routed_experts,
            routed_expert_weights=output.routed_expert_weights,
            routed_experts_source_semantics=(
                KIMI_FULL_MODEL_ROUTE_SEMANTICS
                if self._kimi_full_r3 and output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
