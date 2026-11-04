"""PPO batch preparation and updates extracted from the original v25 experiment.

The function bodies are preserved from strategy_v25/train.py. Only standalone
imports were added. Complete episodes use gamma=lambda=1: advantages are the
terminal return minus the predicted value, not a truncated GAE calculation.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import grouped_ppo_loss


def stack_rollouts(trajectories, *, feature_dim, head_widths, groups):
    if not trajectories:
        raise ValueError("no complete trajectories collected")
    batch = {key: np.concatenate([trajectory[key] for trajectory in trajectories])
             for key in trajectories[0]}
    count = len(batch["returns"])
    shapes = {"features": (count, feature_dim), "masks": (count, sum(head_widths)),
              "values": (count,), "returns": (count,)}
    shapes.update({key: (count, len(head_widths))
                   for key in ("actions", "active", "old_log_probs")})
    if not count or any(batch[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("rollout shapes differ from the expanded model contract")
    if any(not np.isfinite(batch[key]).all()
           for key in ("features", "old_log_probs", "values", "returns")):
        raise ValueError("non-finite rollout features, probabilities, values or returns")
    if np.any(batch["actions"] < 0) or np.any(batch["actions"] >= np.asarray(head_widths)):
        raise ValueError("rollout action is outside its categorical head")
    selected = batch["actions"] + np.cumsum((0,) + tuple(head_widths))[:-1]
    if not batch["masks"][np.arange(count)[:, None], selected].all():
        raise ValueError("rollout selected a masked action")
    advantages = batch["returns"] - batch["values"]
    normalized = np.zeros_like(batch["old_log_probs"])
    for heads in groups:
        heads = list(heads)
        active = batch["active"][:, heads]
        values = np.broadcast_to(advantages[:, None], active.shape)[active]
        if values.size:
            scaled = (advantages - values.mean()) / max(float(values.std()), 1e-8)
            normalized[:, heads] = scaled[:, None]
    batch["advantages"] = normalized
    return batch


def update(network, optimizer, arrays, device, entropy_coefficient, rng, minibatch_size=2048):
    batch = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
    batch["actions"] = batch["actions"].long()
    count = len(arrays["returns"])
    # This catches changed priors, masking, weight layout and export precision.
    with torch.no_grad():
        discrepancy = 0.0
        for start in range(0, count, minibatch_size):
            end = start + minibatch_size
            log_probs, _, _ = network.evaluate(batch["features"][start:end],
                                               batch["masks"][start:end], batch["actions"][start:end])
            active = batch["active"][start:end]
            if bool(active.any()):
                discrepancy = max(discrepancy, float((log_probs - batch["old_log_probs"][start:end])[active].abs().max()))
        if discrepancy > 2e-4:
            raise RuntimeError(f"Torch/rollout logprob parity failed: {discrepancy}")
    measurements = []
    stopped = False
    for epoch in range(4):
        indices = rng.permutation(count)
        for start in range(0, count, minibatch_size):
            index = torch.as_tensor(indices[start:start + minibatch_size], device=device)
            mini = {key: value[index] for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = grouped_ppo_loss(network, mini, entropy_coefficient)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite PPO loss")
            # Do not take another update when this policy already moved too far.
            if float(metrics["approx_kl"]) > .03:
                stopped = True
                break
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), .5))
            if not np.isfinite(gradient_norm):
                raise RuntimeError("non-finite PPO gradient")
            optimizer.step()
            measurements.append({key: float(value) for key, value in metrics.items()})
        if stopped:
            break
    return {key: float(np.mean([entry[key] for entry in measurements]))
            for key in measurements[0]} | {
                "gradient_steps": len(measurements), "kl_stopped": stopped,
                "preupdate_logprob_max_error": discrepancy,
                "head_active_counts": arrays["active"].sum(0).tolist(),
                "head_non_native_counts": ((arrays["actions"] != 0) & arrays["active"]).sum(0).tolist(),
                "entropy_coefficient": entropy_coefficient,
            } if measurements else {"gradient_steps": 0, "kl_stopped": stopped,
                                    "preupdate_logprob_max_error": discrepancy}
