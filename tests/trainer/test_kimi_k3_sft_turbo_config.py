# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).parents[2] / "verl" / "trainer" / "config"


def test_kimi_k3_sft_turbo_config_is_text_only():
    """Keep the Kimi wrapper recipe on its language-only module path."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="sft_trainer_mindspeed_fsdp_turbo")

    assert config.engine.strategy == "mindspeed_fsdp"
    assert config.trainer.device == "npu"
    assert config.model.trust_remote_code is True
    assert config.model.use_remove_padding is False
    assert config.model.enable_gradient_checkpointing is False
    assert config.engine.ulysses_sequence_parallel_size == 1

    fsdp_plan = config.engine.fsdp_kwargs.distributed.fsdp_plan
    ep_plan = config.engine.fsdp_kwargs.distributed.ep_plan
    recompute_plan = config.engine.fsdp_kwargs.memory.recompute_plan
    configured_paths = [
        *fsdp_plan.apply_modules.keys(),
        *fsdp_plan.hook_modules,
        *ep_plan.apply_modules,
        *recompute_plan,
    ]

    assert "language_model.model.layers.{*}.block_sparse_moe.experts" in ep_plan.apply_modules
    assert all("vision" not in path for path in configured_paths)
    assert all("mm_projector" not in path for path in configured_paths)
    assert all(path.startswith("language_model.") for path in configured_paths)
