"""Persistent CPU workers, fresh policy namespaces, and complete-game rollouts.

No Torch import belongs here: worker policies infer using the portable NumPy
controller. Compiled code is cached, but mutable policy state is never reused.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategy_v9"))
from evaluate_fast import Policy

_COMPILED = {}
_WORKER = {}
CONFIGURATION = {
    "episodeSteps": 720, "turnsPerDay": 24, "boardSize": 10,
    "startingMoney": 3000, "maxMarketOrdersPerTurn": 10, "shedCapacity": 100,
    "farmHandCostMult": 1, "actTimeout": 1, "remainingOverageTime": 60,
    "townCenterSellInterval": 24, "townShopUnlockInterval": 3,
    "townShopSellInterval": 4, "weedSpawnChance": .005,
}


class Configuration(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


CONFIGURATION = Configuration(CONFIGURATION)
CONFIGURATION["seed"] = None


class CachedPolicy(Policy):
    """evaluate_fast.Policy's seat isolation with an immutable compile cache."""
    def __init__(self, path):
        path = Path(path).resolve()
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        code = _COMPILED.get(key)
        if code is None:
            code = compile(path.read_text(), str(path), "exec")
            _COMPILED[key] = code
        self.namespace = {"__name__": "_screen_policy", "__file__": str(path)}
        before = dict(sys.modules)
        exec(code, self.namespace)
        self.modules = {key: value for key, value in sys.modules.items()
                        if before.get(key) is not value}
        self.fn = self.namespace["agent"]
        self.two = len(inspect.signature(self.fn).parameters) >= 2
        self.times = []

    def act(self, obs, cfg):
        return super().act(obs, Configuration(cfg))


def initialize_worker(source, opponents, weights_state, native_threads=1):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(native_threads)
    try:
        from threadpoolctl import threadpool_limits
        _WORKER["thread_limit"] = threadpool_limits(native_threads)
    except ImportError:
        pass
    sys.path.insert(0, str(ROOT / "strategy_v9" / "vendor" / "cppsim"))
    import kagsim
    _WORKER.update(source=str(source), opponents=opponents,
                   weights_state=weights_state, cached_version=None,
                   kagsim=kagsim,
                   engine_sha256=hashlib.sha256(Path(kagsim.__file__).read_bytes()).hexdigest(),
                   engine_version=getattr(kagsim, "ENGINE_VERSION", "unknown"))


def _weights(version):
    if _WORKER["cached_version"] != version:
        payload = _WORKER["weights_state"]["payload"]
        if payload["version"] != version:
            raise RuntimeError("rollout weight version changed during collection")
        _WORKER["weights"] = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["model"].items()}
        _WORKER["cached_version"] = version
    return _WORKER["weights"]


def make_policy(path, model=None, training=False, rng_seed=0):
    policy = CachedPolicy(path)
    if model is not None:
        policy.namespace.update(V25_MODEL=model, V25_TRAINING=bool(training),
                                V25_RNG=random.Random(rng_seed), V25_MODE="policy")
    return policy


def pack_trajectory(records, reward):
    if not records:
        raise ValueError("candidate recorded no decision trajectory")
    size = len(records)
    return {
        "features": np.asarray([r["features"] for r in records], dtype=np.float32),
        "actions": np.asarray([r["actions"] for r in records], dtype=np.int64),
        "masks": np.asarray([r["mask"] for r in records], dtype=bool),
        "active": np.asarray([r["active"] for r in records], dtype=bool),
        "old_log_probs": np.asarray([r["old_log_probs"] for r in records], dtype=np.float32),
        "values": np.asarray([r["value"] for r in records], dtype=np.float32),
        "returns": np.full(size, reward, dtype=np.float32),
    }


def play_episode(job):
    started = time.perf_counter()
    row = {key: job[key] for key in (
        "phase", "episode_id", "seed", "seat", "opponent", "policy_version",
        "policy_sha256", "source_sha256", "opponent_sha256", "inference")}
    row.update(valid=False, error=None)
    row.update(engine_sha256=_WORKER["engine_sha256"],
               engine_version=_WORKER["engine_version"])
    # Agent output is not an unbounded training artifact.
    class Discard:
        def write(self, value): return len(value)
        def flush(self): pass
    try:
        with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
            model = _weights(job["policy_version"])
            candidate = make_policy(_WORKER["source"], model, job["inference"] == "sample",
                                    job["action_seed"])
            if job["opponent"] == "self":
                opponent = make_policy(_WORKER["source"], model,
                                       job["inference"] == "sample", job["action_seed"] + 1)
            else:
                opponent = make_policy(_WORKER["opponents"][job["opponent"]])
            policies = [candidate, opponent]
            if job["seat"]:
                policies.reverse()
            game = _WORKER["kagsim"].Game(job["seed"])
            steps = 0
            while not game.done:
                observations = [game.observe(seat) for seat in (0, 1)]
                actions = [policy.act(obs, CONFIGURATION)
                           for policy, obs in zip(policies, observations)]
                game.step(*actions)
                steps += 1
            if steps != 719:
                raise RuntimeError(f"expected719actions, observed{steps}")
            money = [float(game.reward(seat)) for seat in (0, 1)]
            own = job["seat"]
            margin = money[own] - money[1 - own]
            reward = float(np.sign(margin))
            trajectories = []
            if job["phase"] == "train":
                trajectories.append(pack_trajectory(
                    candidate.namespace["V25_TRAJECTORY"], reward))
                if job["opponent"] == "self":
                    trajectories.append(pack_trajectory(
                        opponent.namespace["V25_TRAJECTORY"], -reward))
            row.update(valid=True, own_money=money[own], opponent_money=money[1-own],
                       margin=margin, win=(reward + 1) / 2, reward=reward, steps=steps,
                       max_action_seconds=max(candidate.times, default=0),
                       opponent_max_action_seconds=max(opponent.times, default=0),
                       decision_count=sum(len(t["returns"]) for t in trajectories),
                       diagnostics=candidate.namespace.get("V25_DIAGNOSTICS", {}))
    except Exception:
        row["error"] = traceback.format_exc(limit=12)
        trajectories = []
    row["elapsed_seconds"] = time.perf_counter() - started
    return row, trajectories
