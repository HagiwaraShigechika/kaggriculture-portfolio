"""Exercise the original PPO update with synthetic, complete-episode batches.

This checks numerical execution only. It does not simulate Kaggriculture or
measure the performance of a trained game-playing agent.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .model import ActorCritic, FEATURE_DIM, GROUPS, HEAD_OFFSETS, HEAD_WIDTHS
from .update import stack_rollouts, update


def synthetic_trajectories(network, *, seed=7, steps=32):
    """Generate two explicitly artificial episodes with opposite terminal rewards."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    rng = np.random.default_rng(seed)
    trajectories = []
    for terminal_return in (-1.0, 1.0):
        features = rng.normal(0, .2, (steps, FEATURE_DIM)).astype(np.float32)
        masks = rng.random((steps, sum(HEAD_WIDTHS))) > .2
        masks[:, list(HEAD_OFFSETS[:-1])] = True  # Every head has a legal option.
        with torch.no_grad():
            logits, _ = network(torch.from_numpy(features))
            action_columns = []
            for left, right in zip(HEAD_OFFSETS[:-1], HEAD_OFFSETS[1:]):
                masked_logits = logits[:, left:right].masked_fill(
                    ~torch.from_numpy(masks[:, left:right]), -1e9)
                probabilities = torch.softmax(masked_logits, dim=1).numpy()
                action_columns.append(np.array([
                    rng.choice(right - left, p=row / row.sum())
                    for row in probabilities], dtype=np.int64))
            actions = np.stack(action_columns, axis=1)
            log_probs, _, values = network.evaluate(
                torch.from_numpy(features), torch.from_numpy(masks),
                torch.from_numpy(actions))
        active = np.ones((steps, len(HEAD_WIDTHS)), dtype=bool)
        active[:, :2] = False
        active[0, 0] = True  # Route is an infrequent decision.
        active[steps // 2, 1] = True
        trajectories.append({
            "features": features, "masks": masks, "actions": actions,
            "active": active, "old_log_probs": log_probs.numpy(),
            "values": values.numpy(),
            "returns": np.full(steps, terminal_return, dtype=np.float32),
        })
    return trajectories


def run_smoke(seed=7):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    network = ActorCritic()
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    trajectories = synthetic_trajectories(network, seed=seed)
    batch = stack_rollouts(trajectories, feature_dim=FEATURE_DIM,
                          head_widths=HEAD_WIDTHS, groups=GROUPS)
    before = {name: parameter.detach().clone()
              for name, parameter in network.named_parameters()}
    metrics = update(network, optimizer, batch, "cpu", .02,
                     np.random.default_rng(seed), minibatch_size=64)
    max_change = max(float((parameter.detach() - before[name]).abs().max())
                     for name, parameter in network.named_parameters())
    if metrics.get("gradient_steps", 0) < 1 or max_change <= 0:
        raise RuntimeError("PPO smoke check did not update any model parameters")
    if not all(bool(torch.isfinite(p).all()) for p in network.parameters()):
        raise RuntimeError("PPO smoke check produced non-finite model parameters")
    return {
        "synthetic_data": True,
        "game_performance_evaluated": False,
        "device": "cpu", "seed": seed, "samples": len(batch["returns"]),
        "parameter_max_change": max_change, "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
