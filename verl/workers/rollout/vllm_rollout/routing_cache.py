# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Keep rollout cache semantics when AuxOutput requires block hashing."""

from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class RoutingCachePolicy:
    prefix_caching: bool
    isolate_requests: bool

    @classmethod
    def from_config(cls, *, prefix_caching: bool, routing_replay: bool, engine_args_type):
        uses_aux_output = "aux_output_config" in getattr(engine_args_type, "__dataclass_fields__", {})
        required = routing_replay and uses_aux_output
        return cls(prefix_caching=prefix_caching or required, isolate_requests=required and not prefix_caching)

    def prepare_prompt(self, prompt: dict) -> None:
        if self.isolate_requests:
            # A fresh salt also isolates retries of the same request ID. vLLM
            # propagates it through multimodal processing into block hashes.
            prompt["cache_salt"] = uuid4().hex
