import math
import unittest

from synthetic_reward_debug.reward import build_group_rank_sequence_rewards


class TestGroupRankSequenceRewards(unittest.TestCase):
    def test_assigns_by_session_id_not_row_order(self):
        keys = [
            "prompt_with_underscores_2_0",
            "prompt_with_underscores_0_0",
            "prompt_with_underscores_1_0",
        ]
        rewards = build_group_rank_sequence_rewards(keys, [True, True, True], scale=2.0)
        self.assertEqual(rewards, [2.0, -2.0, 0.0])
        self.assertTrue(math.isclose(sum(rewards), 0.0, abs_tol=1e-12))

    def test_all_outputs_in_one_session_share_reward(self):
        keys = ["uid_0_0", "uid_0_1", "uid_1_0", "uid_1_2"]
        self.assertEqual(
            build_group_rank_sequence_rewards(keys, [True] * 4, scale=0.5),
            [-0.5, -0.5, 0.5, 0.5],
        )

    def test_padding_is_zero(self):
        keys = ["uid_0_0", "uid_1_0", "padding_0_0"]
        self.assertEqual(
            build_group_rank_sequence_rewards(keys, [True, True, False], scale=1.0),
            [-1.0, 1.0, 0.0],
        )

    def test_single_session_fails(self):
        with self.assertRaisesRegex(ValueError, "rollout.n>=2"):
            build_group_rank_sequence_rewards(["uid_0_0"], [True], scale=1.0)


if __name__ == "__main__":
    unittest.main()
