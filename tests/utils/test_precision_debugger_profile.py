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

import logging
import types

from verl.utils.profiler.config import PrecisionDebuggerToolConfig, ProfilerConfig
from verl.utils.profiler.precision_debugger_profile import PrecisionDebuggerProfiler
from verl.utils.profiler.profile import DistProfiler


class _FakeModel:
    def forward(self):
        pass


def test_resolve_megatron_model_chunks_uses_first_valid_chunk(caplog):
    """Megatron's engine module may contain pipeline model chunks."""
    first_model = _FakeModel()
    second_model = _FakeModel()
    worker = types.SimpleNamespace(
        actor=types.SimpleNamespace(
            engine=types.SimpleNamespace(module=[object(), first_model, second_model]),
        )
    )
    profiler = PrecisionDebuggerProfiler(PrecisionDebuggerToolConfig())

    with caplog.at_level(logging.WARNING):
        model = profiler._resolve_model(worker, "actor_compute_log_prob")

    assert model is first_model
    assert "only binds the first of 2 model chunks" in caplog.text


def _precision_dist_profiler(rank, *, ranks=None, all_ranks=False):
    tool_config = PrecisionDebuggerToolConfig()
    config = ProfilerConfig(
        tool="precision_debugger",
        enable=True,
        all_ranks=all_ranks,
        ranks=[] if ranks is None else ranks,
        save_path="/tmp/test_precision_debugger_profile",
        tool_config=tool_config,
    )
    return DistProfiler(rank=rank, config=config, tool_config=tool_config)


def test_dist_profiler_honors_explicit_precision_debugger_ranks():
    """Explicit global ranks prevent shared-path collisions across nodes."""
    assert _precision_dist_profiler(0, ranks=[0]).check_this_rank()
    assert not _precision_dist_profiler(16, ranks=[0]).check_this_rank()


def test_dist_profiler_preserves_msprobe_rank_filter_fallback():
    """Without a verl rank selection, msprobe's own rank filter remains authoritative."""
    assert _precision_dist_profiler(16).check_this_rank()
    assert _precision_dist_profiler(16, all_ranks=True).check_this_rank()
