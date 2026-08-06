"""Tests for native msprobe dump comparison."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strict_parity_debug.compare_msprobe import compare, load_records


class CompareMsprobeTest(unittest.TestCase):
    def test_loads_native_dump_json_names_and_statistics(self):
        payload = {
            "task": "statistics",
            "level": "L0",
            "framework": "pytorch",
            "dump_data_dir": None,
            "data": {
                "Module.model.layer.Linear.forward.0": {
                    "input_args": [
                        {
                            "type": "torch.Tensor",
                            "dtype": "torch.bfloat16",
                            "shape": [2, 4],
                            "Max": 3.0,
                            "Min": -2.0,
                            "Mean": 0.25,
                            "Norm": 4.5,
                        }
                    ],
                    "input_kwargs": {},
                    "output": [
                        {
                            "type": "torch.Tensor",
                            "dtype": "torch.bfloat16",
                            "shape": [2, 8],
                            "Max": 5.0,
                            "Min": -4.0,
                            "Mean": 0.5,
                            "Norm": 8.0,
                        }
                    ],
                    "parameters": {
                        "weight": {
                            "type": "torch.Tensor",
                            "dtype": "torch.bfloat16",
                            "shape": [8, 4],
                            "Max": 0.5,
                            "Min": -0.5,
                            "Mean": None,
                            "Norm": None,
                        }
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory, "step0", "rank0", "dump.json")
            dump.parent.mkdir(parents=True)
            dump.write_text(json.dumps(payload), encoding="utf-8")
            records = load_records(directory)

        self.assertEqual(
            set(records),
            {
                "Module.model.layer.Linear.forward.0.input.0",
                "Module.model.layer.Linear.forward.0.output.0",
                "Module.model.layer.Linear.forward.0.parameters.weight",
            },
        )
        self.assertEqual(
            records["Module.model.layer.Linear.forward.0.input.0"][0].stats,
            {
                "max": 3.0,
                "min": -2.0,
                "mean": 0.25,
                "norm": 4.5,
                "shape": [2, 4],
                "dtype": "torch.bfloat16",
            },
        )
        self.assertEqual(
            records["Module.model.layer.Linear.forward.0.parameters.weight"][0].stats,
            {"max": 0.5, "min": -0.5, "shape": [8, 4], "dtype": "torch.bfloat16"},
        )

    def test_native_dumps_compare_equal(self):
        payload = {
            "task": "statistics",
            "level": "L0",
            "framework": "pytorch",
            "data": {
                "Module.model.forward.0": {
                    "output": [
                        {"dtype": "torch.float32", "shape": [1], "Max": 1, "Min": 1, "Mean": 1, "Norm": 1}
                    ]
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ("train", "rollout"):
                dump = root / side / "step0" / "rank0" / "dump.json"
                dump.parent.mkdir(parents=True)
                dump.write_text(json.dumps(payload), encoding="utf-8")
            result = compare(
                load_records(root / "train"),
                load_records(root / "rollout"),
                atol=1e-5,
                rtol=1e-3,
                max_mismatches=10,
            )

        self.assertEqual(result["train_records"], 1)
        self.assertEqual(result["rollout_records"], 1)
        self.assertTrue(result["equal"])


if __name__ == "__main__":
    unittest.main()
