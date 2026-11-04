"""Numerical contracts of the retained PPO implementation; no game strength tests."""
import unittest

import numpy as np
import torch

from ppo.model import (
    ActorCritic, FEATURE_DIM, GROUPS, HEAD_OFFSETS, HEAD_WIDTHS,
    grouped_ppo_loss,
    test_missing_investment_group as check_original_group_weights,
)
from ppo.smoke import run_smoke, synthetic_trajectories
from ppo.update import stack_rollouts


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = ActorCritic()

    def test_original_group_loss_and_inactive_gradients(self):
        result = check_original_group_weights()
        self.assertTrue(result["missing_investment_fixed_weight"])
        self.assertTrue(result["inactive_gradient_zero"])

    def test_clipping_saturates_for_positive_and_negative_advantages(self):
        shape = (1, len(HEAD_WIDTHS))
        for ratio, advantage, clipped in ((1.5, 2.0, 1.2), (.5, -2.0, .8)):
            log_probs = torch.full(shape, float(np.log(ratio)), requires_grad=True)

            class KnownOutputs:
                def evaluate(self, features, masks, actions):
                    return log_probs, torch.zeros(shape), torch.zeros(1)

            active = torch.zeros(shape, dtype=torch.bool)
            active[0, 0] = True
            batch = {"features": None, "masks": None, "actions": None,
                     "active": active, "old_log_probs": torch.zeros(shape),
                     "advantages": torch.full(shape, advantage),
                     "returns": torch.zeros(1)}
            loss, metrics = grouped_ppo_loss(KnownOutputs(), batch, 0.0)
            self.assertAlmostEqual(float(loss.detach()), -clipped * advantage / 5, places=6)
            self.assertAlmostEqual(float(metrics["clip_fraction"]), 1 / 5, places=6)
            loss.backward()
            self.assertTrue(bool((log_probs.grad == 0).all()))

    def test_masks_exclude_invalid_actions(self):
        trajectories = synthetic_trajectories(self.model)
        for trajectory in trajectories:
            selected = trajectory["actions"] + np.array(HEAD_OFFSETS[:-1])
            self.assertTrue(trajectory["masks"][
                np.arange(len(selected))[:, None], selected].all())
        features = torch.zeros((1, FEATURE_DIM))
        masks = torch.ones((1, sum(HEAD_WIDTHS)), dtype=torch.bool)
        masks[:, :HEAD_WIDTHS[0]] = False
        with self.assertRaisesRegex(ValueError, "no legal action"):
            self.model.evaluate(features, masks,
                                torch.zeros((1, len(HEAD_WIDTHS)), dtype=torch.long))

    def test_reject_masked_rollout_action(self):
        trajectories = synthetic_trajectories(self.model)
        trajectories[0]["masks"][0, HEAD_OFFSETS[0] + trajectories[0]["actions"][0, 0]] = False
        with self.assertRaisesRegex(ValueError, "masked action"):
            stack_rollouts(trajectories, feature_dim=FEATURE_DIM,
                           head_widths=HEAD_WIDTHS, groups=GROUPS)

    def test_portable_weight_roundtrip(self):
        clone = ActorCritic()
        clone.load_portable(self.model.portable())
        features = torch.randn((4, FEATURE_DIM))
        with torch.no_grad():
            expected = self.model(features)
            actual = clone(features)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_update_has_finite_gradients_and_changes_weights(self):
        result = run_smoke()
        self.assertTrue(result["synthetic_data"])
        self.assertFalse(result["game_performance_evaluated"])
        self.assertGreater(result["metrics"]["gradient_steps"], 0)
        self.assertGreater(result["parameter_max_change"], 0)
        self.assertLess(result["metrics"]["preupdate_logprob_max_error"], 2e-4)


if __name__ == "__main__":
    unittest.main()
