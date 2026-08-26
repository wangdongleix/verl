# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics


class TestMetrics(unittest.TestCase):
    def test_log_prob_diff_compatibility_alias(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0, -2.0, -3.0]]),
                "old_log_probs": torch.tensor([[-1.5, -1.0, -3.5]]),
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )
        metrics = calculate_debug_metrics(data)
        self.assertEqual(metrics["training/rollout_probs_diff_valid"], 1)
        self.assertEqual(
            metrics["training/rollout_log_probs_diff_mean"],
            metrics["training/rollout_log_probs_abs_diff_mean"],
        )
        self.assertAlmostEqual(metrics["training/rollout_log_probs_diff_mean"], 0.75)

    def test_empty_mask_still_emits_compatibility_alias(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.zeros((1, 2)),
                "old_log_probs": torch.zeros((1, 2)),
                "response_mask": torch.zeros((1, 2)),
                "responses": torch.zeros((1, 2)),
            }
        )
        metrics = calculate_debug_metrics(data)
        self.assertIn("training/rollout_log_probs_diff_mean", metrics)


if __name__ == "__main__":
    unittest.main()
