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

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from verl.utils.debug.weight_sync import _parse_step_filter, _step_is_selected, tensor_signature, trace_weight_stream


class TestWeightSyncDebug(unittest.TestCase):
    def test_step_filter_supports_points_and_ranges(self):
        step_filter = _parse_step_filter("0, 5-7, 10")
        self.assertTrue(_step_is_selected(step_filter, 0))
        self.assertTrue(_step_is_selected(step_filter, "6"))
        self.assertTrue(_step_is_selected(step_filter, 10))
        self.assertFalse(_step_is_selected(step_filter, 8))
        self.assertFalse(_step_is_selected(step_filter, None))
        self.assertTrue(_step_is_selected(None, None))

    def test_tensor_signature_is_stable_and_detects_changes(self):
        tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        with patch.dict(
            os.environ,
            {"VERL_WEIGHT_SYNC_DEBUG_SAMPLE_SIZE": "8", "VERL_WEIGHT_SYNC_DEBUG_STATS": "full"},
            clear=False,
        ):
            first = tensor_signature(tensor)
            second = tensor_signature(tensor.clone())
            changed = tensor.clone()
            changed[-1, -1] += 1
            third = tensor_signature(changed)

        self.assertEqual(first, second)
        self.assertNotEqual(first["sha256"], third["sha256"])
        self.assertEqual(first["shape"], [4, 8])
        self.assertEqual(first["sample_numel"], 8)
        self.assertEqual(first["stats_mode"], "full")
        self.assertEqual(first["stats_numel"], 32)
        self.assertAlmostEqual(first["mean"], 15.5)
        self.assertEqual(first["min"], 0.0)
        self.assertEqual(first["max"], 31.0)

    def test_trace_weight_stream_preserves_items_when_disabled(self):
        weights = [("a", torch.ones(2)), ("b", torch.zeros(2))]
        with patch.dict(os.environ, {"VERL_WEIGHT_SYNC_DEBUG": "0"}, clear=False):
            traced = trace_weight_stream(weights, stage="test")
        self.assertIs(traced, weights)

    def test_trace_weight_stream_preserves_items_when_enabled(self):
        weights = [("a", torch.ones(2)), ("b", torch.zeros(2))]
        env = {
            "VERL_WEIGHT_SYNC_DEBUG": "1",
            "VERL_WEIGHT_SYNC_DEBUG_NAMES": "^b$",
            "VERL_WEIGHT_SYNC_DEBUG_MAX_TENSORS": "1",
            "VERL_WEIGHT_SYNC_DEBUG_SAMPLE_SIZE": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            traced = list(trace_weight_stream(weights, stage="test"))
        self.assertEqual([name for name, _ in traced], ["a", "b"])
        torch.testing.assert_close(traced[0][1], weights[0][1])
        torch.testing.assert_close(traced[1][1], weights[1][1])

    def test_debug_output_file_and_unlimited_default(self):
        weights = [("a", torch.ones(2)), ("b", torch.zeros(2))]
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "weights.jsonl")
            env = {
                "VERL_WEIGHT_SYNC_DEBUG": "1",
                "VERL_WEIGHT_SYNC_DEBUG_MAX_TENSORS": "0",
                "VERL_WEIGHT_SYNC_DEBUG_OUTPUT": output_path,
            }
            with patch.dict(os.environ, env, clear=False):
                list(trace_weight_stream(weights, stage="test", context={"global_steps": 3}))

            with open(output_path, encoding="utf-8") as output_file:
                records = [line for line in output_file if "[weight_sync_debug]" in line]
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
