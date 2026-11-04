"""PPO network and the exact portable NumPy inference weight contract."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

FEATURE_DIM = 352
HEAD_WIDTHS = (38, 3) + (6,) * 7 + (5,) * 7 + (3, 4, 3, 16, 4, 3, 8)
HEAD_OFFSETS = tuple(np.cumsum((0,) + HEAD_WIDTHS).tolist())
GROUPS = ((0,), (1,), tuple(range(2, 16)), (16, 17, 18, 20, 21), (19, 22))
GROUP_NAMES = ("route", "investment", "market", "farm", "labor")
FORMAT = "kaggriculture_v25_mlp_v2"


class ActorCritic(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy = nn.Linear(128, sum(HEAD_WIDTHS))
        self.value = nn.Linear(128, 1)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        for layer in (self.policy, self.value):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        prior = torch.zeros(sum(HEAD_WIDTHS))
        for head, offset in enumerate(HEAD_OFFSETS[:-1]):
            prior[offset] = 5.0 if head == 0 else (3.0 if head < 16 else 4.0)
        self.register_buffer("prior", prior)
        width = max(HEAD_WIDTHS)
        self.register_buffer("categorical_indices", torch.tensor([
            [offset + min(index, size - 1) for index in range(width)]
            for offset, size in zip(HEAD_OFFSETS, HEAD_WIDTHS)]), persistent=False)
        self.register_buffer("categorical_valid", torch.tensor([
            [index < size for index in range(width)]
            for size in HEAD_WIDTHS]), persistent=False)

    def forward(self, features):
        hidden = torch.tanh(self.fc1(features))
        hidden = torch.tanh(self.fc2(hidden))
        return self.policy(hidden) + self.prior, self.value(hidden).squeeze(-1)

    def evaluate(self, features, masks, actions):
        logits, values = self(features)
        # Evaluate all categorical heads together; padding is always masked.
        # This avoids 23 separate CUDA synchronizations per minibatch.
        allowed = masks[:, self.categorical_indices] & self.categorical_valid
        if not bool(allowed.any(dim=2).all()):
            raise ValueError("an action head has no legal action")
        log_probs = torch.log_softmax(
            logits[:, self.categorical_indices].masked_fill(~allowed, -1e9), dim=2)
        selected = log_probs.gather(2, actions[:, :, None]).squeeze(2)
        entropy = -(log_probs.exp() * log_probs).sum(dim=2)
        denom = allowed.sum(dim=2).clamp(min=2).float().log()
        return selected, entropy / denom, values

    def portable(self):
        return {name: value.detach().cpu().numpy().astype(np.float32).tolist()
                for name, value in (
                    ("w1", self.fc1.weight), ("b1", self.fc1.bias),
                    ("w2", self.fc2.weight), ("b2", self.fc2.bias),
                    ("wp", self.policy.weight), ("bp", self.policy.bias),
                    ("wv", self.value.weight), ("bv", self.value.bias))}

    def load_portable(self, document):
        if "format" in document and document["format"] != FORMAT:
            raise ValueError("portable model format differs from the expanded controller contract")
        if "feature_dim" in document and document["feature_dim"] != self.fc1.in_features:
            raise ValueError("portable model feature dimension mismatch")
        if "head_widths" in document and document["head_widths"] != list(HEAD_WIDTHS):
            raise ValueError("portable model action heads mismatch")
        weights = document.get("model", document)
        with torch.no_grad():
            for prefix, layer in (("1", self.fc1), ("2", self.fc2),
                                  ("p", self.policy), ("v", self.value)):
                layer.weight.copy_(torch.as_tensor(weights["w" + prefix],
                                                  device=layer.weight.device))
                layer.bias.copy_(torch.as_tensor(weights["b" + prefix],
                                                device=layer.bias.device))


def weight_hash(weights):
    """Hash float32 tensor bytes, unaffected by JSON whitespace."""
    digest = hashlib.sha256()
    for key in ("w1", "b1", "w2", "b2", "wp", "bp", "wv", "bv"):
        array = np.asarray(weights[key], dtype="<f4")
        digest.update(key.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def export_document(network, metadata):
    weights = network.portable()
    return {"format": FORMAT, "feature_dim": network.fc1.in_features,
            "head_widths": list(HEAD_WIDTHS), "model": weights,
            "metadata": {**metadata, "policy_sha256": weight_hash(weights)}}


def grouped_ppo_loss(network, batch, entropy_coefficient, clip=0.2):
    """Per-head clipped PPO, with five fixed equally weighted action groups.

Each group averages its active categorical decisions. This is intentionally
factorwise clipping, not clipping one joint probability ratio for all heads.
Missing groups contribute zero, preserving the frozen 1/5 weight for each
group and the actor/critic scale when a hook or investment is unavailable.
"""
    log_probs, entropy, values = network.evaluate(
        batch["features"], batch["masks"], batch["actions"])
    policy_terms, entropy_terms, kl_terms, clipping_terms = [], [], [], []
    for heads in GROUPS:
        heads = list(heads)
        active = batch["active"][:, heads]
        if not bool(active.any()):
            continue
        log_ratio = (log_probs[:, heads] - batch["old_log_probs"][:, heads])[active]
        ratio = log_ratio.exp()
        advantages = batch["advantages"][:, heads][active]
        policy_terms.append(-torch.minimum(
            ratio * advantages,
            ratio.clamp(1 - clip, 1 + clip) * advantages).mean())
        entropy_terms.append(entropy[:, heads][active].mean())
        kl_terms.append(((ratio - 1) - log_ratio).mean())
        clipping_terms.append(((ratio - 1).abs() > clip).float().mean())
    if not policy_terms:
        raise ValueError("rollout contains no active policy decisions")
    actor_loss = torch.stack(policy_terms).sum() / len(GROUPS)
    entropy_mean = torch.stack(entropy_terms).sum() / len(GROUPS)
    value_loss = 0.5 * (values - batch["returns"]).square().mean()
    total = actor_loss + value_loss - entropy_coefficient * entropy_mean
    metrics = {"actor_loss": actor_loss.detach(), "value_loss": value_loss.detach(),
               "entropy": entropy_mean.detach(), "approx_kl": (torch.stack(kl_terms).sum() / len(GROUPS)).detach(),
               "clip_fraction": (torch.stack(clipping_terms).sum() / len(GROUPS)).detach()}
    return total, metrics


def test_missing_investment_group():
    """A hand-calculated regression for fixed weights and inactive gradients."""
    shape = (1, len(HEAD_WIDTHS))
    log_probs = torch.zeros(shape, requires_grad=True)
    with torch.no_grad():
        log_probs[0, 0] = math.log(1.1)
        log_probs[0, 2] = math.log(.9)
        log_probs[0, 16] = math.log(1.05)
        log_probs[0, 17] = math.log(1.05)
    entropies = torch.zeros(shape)
    entropies[0, 0] = .6
    entropies[0, 2] = .9
    entropies[0, 16] = .2
    entropies[0, 17] = .8
    entropies[0, 19] = .4
    entropies[0, 22] = .6

    class KnownOutputs:
        def evaluate(self, features, masks, actions):
            return log_probs, entropies, torch.zeros(1)

    active = torch.zeros(shape, dtype=torch.bool)
    active[0, [0, 2, 16, 17, 19, 22]] = True
    advantages = torch.zeros(shape)
    advantages[0, 0] = 2.
    advantages[0, 2] = 4.
    advantages[0, 16] = 6.
    advantages[0, 17] = 2.
    advantages[0, 19] = 8.
    advantages[0, 22] = 4.
    batch = {"features": None, "masks": None, "actions": None,
             "active": active, "old_log_probs": torch.zeros(shape),
             "advantages": advantages, "returns": torch.ones(1)}
    loss, metrics = grouped_ppo_loss(KnownOutputs(), batch, entropy_coefficient=.02)
    expected_actor = -(1.1 * 2 + .9 * 4 + 1.05 * (6 + 2) / 2 + (8 + 4) / 2) / 5
    expected_entropy = (.6 + .9 + (.2 + .8) / 2 + (.4 + .6) / 2) / 5
    expected_kl = ((1.1 - 1 - math.log(1.1)) + (.9 - 1 - math.log(.9))
                   + (1.05 - 1 - math.log(1.05))) / 5
    assert abs(float(metrics["actor_loss"]) - expected_actor) < 1e-6
    assert abs(float(metrics["entropy"]) - expected_entropy) < 1e-6
    assert abs(float(metrics["approx_kl"]) - expected_kl) < 1e-6
    assert abs(float(loss.detach()) - (expected_actor + .5 - .02 * expected_entropy)) < 1e-6
    assert float(metrics["clip_fraction"]) == 0
    loss.backward()
    assert bool((log_probs.grad[~active] == 0).all()), "inactive heads must have no actor gradient"
    expected_gradients = {0: -1.1 * 2 / 5, 2: -.9 * 4 / 5,
                          16: -1.05 * 6 / 10, 17: -1.05 * 2 / 10,
                          19: -8 / 10, 22: -4 / 10}
    for head, expected in expected_gradients.items():
        assert abs(float(log_probs.grad[0, head]) - expected) < 1e-6
    return {"missing_investment_fixed_weight": True, "fixed_groups": 5,
            "inactive_gradient_zero": True, "actor_loss": float(metrics["actor_loss"]),
            "entropy": float(metrics["entropy"]), "approx_kl": float(metrics["approx_kl"])}


if __name__ == "__main__":
    print(json.dumps(test_missing_investment_group()))
