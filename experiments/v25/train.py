"""Time-bounded, complete-episode PPO; one independent learner per invocation.

Example: python strategy_v25/train.py --hours 7 --workers 8 --device cuda
  --seed 20260927 --run-dir results/runs/v25_train0
  --source strategy_v25/policy_candidate.py --plan strategy_v25/experiment.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
import traceback

# Set these before importing NumPy/Torch, including in spawned workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout import initialize_worker, play_episode

# Learner-only imports are loaded in main(). Spawned workers therefore never
# import Torch or initialize a CUDA context.

DEVELOPMENT_OPPONENTS = ("v24", "Moon", "MoreWheat", "Sato6.1")
DEVELOPMENT_SEEDS = tuple(range(3720001, 3720009))
OPPONENT_KEYS = {"V57": "public19_v57", "CA25": "public20_ca25",
                 "Moon": "public20_moon_current", "MoreWheat": "public19_wheat_latest",
                 "Sato6.1": "sato61"}


class BudgetExpired(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, document):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(document, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def append_rows(path, rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def collect(pool, jobs, directory, deadline, batch_timeout):
    started = time.monotonic()
    if time.time() >= deadline:
        raise BudgetExpired("training deadline reached before another batch")
    rows, trajectories = [], []
    futures = [pool.submit(play_episode, job) for job in jobs]
    positions = {future: index for index, future in enumerate(futures)}
    ordered_results = {}
    remaining = deadline - time.time()
    timeout = min(batch_timeout, max(0.01, remaining))
    try:
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            row, items = future.result()
            ordered_results[positions[future]] = (row, items)
            rows.append(row)
            append_rows(directory / "matches.jsonl", [row])
    except BaseException as error:
        # Python3.12 has no public terminate_workers(). A hung policy must not
        # leave __exit__ waiting forever beyond the reserved research budget.
        processes = list((getattr(pool, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        atomic_json(directory / "interrupted_batch.json", {
            "completed": len(rows), "scheduled": len(jobs), "applied_to_learning": False,
            "phase": jobs[0]["phase"] if jobs else None,
            "policy_version": jobs[0]["policy_version"] if jobs else None,
            "reason": type(error).__name__, "elapsed_seconds": time.monotonic() - started})
        if isinstance(error, concurrent.futures.TimeoutError) and time.time() >= deadline - 1:
            raise BudgetExpired("training deadline reached; incomplete batch discarded") from error
        raise
    errors = [row for row in rows if not row["valid"]]
    if errors:
        atomic_json(directory / "rollout_error.json", errors)
        raise RuntimeError(f"{len(errors)}/{len(rows)} invalid episodes; refusing biased PPO batch")
    # Completion order is scheduling noise, not part of the experiment seed.
    rows = [ordered_results[index][0] for index in range(len(jobs))]
    trajectories = [item for index in range(len(jobs)) for item in ordered_results[index][1]]
    return rows, trajectories, time.monotonic() - started


def development_score(rows):
    by_opponent = {name: [row["win"] for row in rows if row["opponent"] == name]
                   for name in DEVELOPMENT_OPPONENTS}
    if any(len(values) != 16 for values in by_opponent.values()):
        raise ValueError("development evaluation must contain8seeds,bothseats,all4opponents")
    scores = {name: float(np.mean(values)) for name, values in by_opponent.items()}
    return .5 * scores["v24"] + .5 * np.mean([scores[n] for n in DEVELOPMENT_OPPONENTS[1:]]), scores


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


def main(argv=None):
    global torch, ActorCritic, export_document, grouped_ppo_loss, weight_hash
    import torch
    from model import (ActorCritic, export_document, grouped_ppo_loss, weight_hash,
                       FEATURE_DIM, HEAD_WIDTHS, GROUPS, GROUP_NAMES, FORMAT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-episodes", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--development-minutes", type=float, default=30)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--deadline-unix", type=float,
                        help="Absolute global deadline; overrides a longer per-run budget")
    parser.add_argument("--batch-timeout-seconds", type=float, default=900,
                        help="Abort stalled batches, preserving the previous checkpoint")
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.workers < 1 or args.batch_episodes < 1:
        parser.error("hours,workers,batch-episodes must be positive")
    args.source = args.source.resolve()
    args.plan = args.plan.resolve()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "last.pt").exists() and not args.resume:
        raise RuntimeError("run already has checkpoints; use--resume or a fresh run-dir")
    plan = json.loads(args.plan.read_text())
    opponents = {"v24": str((ROOT / plan["baseline"]).resolve())}
    opponents.update({name: str((ROOT / plan["opponents"][key]).resolve())
                      for name, key in OPPONENT_KEYS.items()})
    if plan["development_seeds"] != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seeds differ from the implemented frozen protocol")
    if set(plan["training_opponents"]) != {OPPONENT_KEYS[name] for name in ("V57", "CA25", "Moon", "MoreWheat")}:
        raise ValueError("training opponent list differs from the frozen protocol")
    if set(plan["development_opponents"]) != {"v24", *(OPPONENT_KEYS[name] for name in DEVELOPMENT_OPPONENTS[1:])}:
        raise ValueError("development opponent list differs from the frozen protocol")
    if (plan["model"]["feature_dim"] != FEATURE_DIM
            or plan["model"]["head_widths"] != list(HEAD_WIDTHS)):
        raise ValueError("plan model differs from the controller contract")
    expected_parameters = {"epochs": 4, "clip": .2, "learning_rate": 3e-4,
                           "gamma": 1., "gae_lambda": 1., "target_kl": .03,
                           "entropy_start": .02, "entropy_end": .002,
                           "max_gradient_norm": .5}
    if any(plan["ppo"][key] != value for key, value in expected_parameters.items()):
        raise ValueError("PPO parameters differ from the frozen implementation")
    if plan["training_mixture"] != {"selfplay": .6, "v24": .25, "public": .15}:
        raise ValueError("training mixture differs from frozen implementation")
    source_hash = digest(args.source)
    opponent_hashes = {name: digest(path) for name, path in opponents.items()}
    if opponent_hashes["v24"] != plan["baseline_sha256"]:
        raise ValueError("frozen v24 baseline hash mismatch")
    expected_hashes = {entry["name"]: entry["sha256"] for entry in plan["opponent_metadata"]}
    for name, key in OPPONENT_KEYS.items():
        if opponent_hashes[name] != expected_hashes[key]:
            raise ValueError(f"frozen opponent hash mismatch: {name}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    schedule_rng = random.Random(args.seed)
    network = ActorCritic().to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    version, updates, total_games, elapsed_before = 0, 0, 0, 0.0
    best_score = -float("inf")
    used_seeds = set()
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if state["source_sha256"] != source_hash or state["plan_sha256"] != digest(args.plan):
            raise RuntimeError("resume source or plan differs from checkpoint")
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule_rng.setstate(state["schedule_rng"])
        np_rng.bit_generator.state = state["numpy_rng"]
        version, updates, total_games = state["version"], state["updates"], state["total_games"]
        elapsed_before = state["elapsed_seconds"]
        best_score = state["best_score"]
        used_seeds = set(state["used_training_seeds"])
    metadata = {"seed": args.seed, "source_sha256": source_hash,
                "plan_sha256": digest(args.plan), "opponent_hashes": opponent_hashes,
                "model_format": FORMAT, "feature_dim": FEATURE_DIM,
                "head_widths": list(HEAD_WIDTHS),
                "action_groups": {name: list(heads) for name, heads in zip(GROUP_NAMES, GROUPS)},
                "group_weights": {name: 1 / len(GROUPS) for name in GROUP_NAMES},
                "batch_episodes": args.batch_episodes, "gamma": 1, "gae_lambda": 1,
                "epochs": 4, "clip": .2, "learning_rate": 3e-4, "target_kl": .03,
                "entropy": [.02, .002], "gradient_clip": .5,
                "training_mix": {"self": .6, "v24": .25, "public": .15},
                "training_mix_unit": "complete games; both seats retained for self-play",
                "development_seeds": list(DEVELOPMENT_SEEDS), "device": args.device,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "workers": args.workers, "started_at_unix": time.time(),
                "source": str(args.source), "plan": str(args.plan),
                "opponent_aliases": OPPONENT_KEYS,
                "runtime_source_hashes": {name: digest(Path(__file__).with_name(name))
                                          for name in ("train.py", "rollout.py", "model.py")},
                "batch_timeout_seconds": args.batch_timeout_seconds,
                "deadline_unix": args.deadline_unix}
    atomic_json(directory / "training_manifest.json", metadata)
    started = time.monotonic()
    duration = args.hours * 3600
    deadline = min(time.time() + max(0, duration - elapsed_before),
                   args.deadline_unix if args.deadline_unix is not None else float("inf"))
    last_development = -float("inf")
    latest_development_score = None
    recent_collection_seconds = 0.0
    recent_development_seconds = 0.0
    ctx = mp.get_context("spawn")

    def elapsed(): return elapsed_before + time.monotonic() - started

    def checkpoint(name, development=None):
        info = {**metadata, "update": updates, "policy_version": version,
                "total_training_games": total_games, "elapsed_seconds": elapsed(),
                "development_score": development,
                "inference": "deterministic_argmax"}
        document = export_document(network, info)
        atomic_json(directory / f"{name}.json", document)
        state = {"model": network.state_dict(), "optimizer": optimizer.state_dict(),
                 "version": version, "updates": updates, "total_games": total_games,
                 "elapsed_seconds": elapsed(), "best_score": best_score,
                 "schedule_rng": schedule_rng.getstate(),
                 "numpy_rng": np_rng.bit_generator.state,
                 "used_training_seeds": sorted(used_seeds),
                 "source_sha256": source_hash, "plan_sha256": digest(args.plan)}
        temporary = directory / f"{name}.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(directory / f"{name}.pt")
        return document

    with ctx.Manager() as manager:
        shared = manager.dict()
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx,
                initializer=initialize_worker,
                initargs=(str(args.source), opponents, shared)) as pool:

            def publish():
                if digest(args.source) != source_hash:
                    raise RuntimeError("candidate source changed during training")
                if any(digest(path) != opponent_hashes[name] for name, path in opponents.items()):
                    raise RuntimeError("frozen opponent changed during training")
                weights = network.portable()
                shared["payload"] = {"version": version, "model": weights}
                return weight_hash(weights)

            def make_job(phase, episode_id, seed, seat, opponent, weights_hash):
                return {"phase": phase, "episode_id": episode_id, "seed": seed,
                        "seat": seat, "opponent": opponent, "policy_version": version,
                        "policy_sha256": weights_hash, "source_sha256": source_hash,
                        "opponent_sha256": weights_hash if opponent == "self" else opponent_hashes[opponent],
                        "inference": "sample" if phase == "train" else "argmax",
                        "action_seed": schedule_rng.randrange(1, 2**63 - 2)}

            def evaluate_development(weights_hash):
                nonlocal best_score, last_development, latest_development_score, recent_development_seconds
                jobs = [make_job("development", f"dev{version}:{name}:{seed}:{seat}",
                                 seed, seat, name, weights_hash)
                        for name in DEVELOPMENT_OPPONENTS for seed in DEVELOPMENT_SEEDS for seat in (0, 1)]
                rows, _, seconds = collect(pool, jobs, directory, deadline, args.batch_timeout_seconds)
                recent_development_seconds = seconds
                score, breakdown = development_score(rows)
                latest_development_score = float(score)
                report = {"update": updates, "policy_version": version,
                          "policy_sha256": weights_hash, "score": float(score),
                          "by_opponent": breakdown, "seconds": seconds,
                          "elapsed_seconds": elapsed(), "games": len(rows)}
                append_rows(directory / "development.jsonl", [report])
                if score > best_score:
                    best_score = float(score)
                    checkpoint("best", float(score))
                last_development = elapsed()
                print(json.dumps({"event": "development", **report}), flush=True)

            policy_hash = publish()
            checkpoint("last")
            evaluate_development(policy_hash)
            checkpoint("last", latest_development_score)
            while elapsed() < duration and (args.max_updates is None or updates < args.max_updates):
                # Finish complete batches. Avoid starting one that is unlikely to fit.
                if recent_collection_seconds and deadline - time.time() < recent_collection_seconds * 1.2 + recent_development_seconds * 1.3 + 10:
                    break
                kinds = (["self"] * round(args.batch_episodes * .6)
                         + ["v24"] * round(args.batch_episodes * .25))
                publics = ("V57", "CA25", "Moon", "MoreWheat")
                for index in range(args.batch_episodes - len(kinds)):
                    kinds.append(publics[(index + updates) % len(publics)])
                schedule_rng.shuffle(kinds)
                jobs = []
                for index, opponent in enumerate(kinds):
                    while True:
                        seed = schedule_rng.randrange(4000000, 2**30)
                        if seed not in used_seeds:
                            used_seeds.add(seed)
                            break
                    jobs.append(make_job("train", total_games + index, seed,
                                         (total_games + index) % 2, opponent, policy_hash))
                rows, trajectories, seconds = collect(pool, jobs, directory, deadline, args.batch_timeout_seconds)
                recent_collection_seconds = seconds
                arrays = stack_rollouts(trajectories, feature_dim=FEATURE_DIM,
                                        head_widths=HEAD_WIDTHS, groups=GROUPS)
                fraction = min(1.0, elapsed() / duration)
                entropy_coefficient = .02 + (.002 - .02) * fraction
                update_started = time.monotonic()
                stats = update(network, optimizer, arrays, args.device,
                               entropy_coefficient, np_rng, args.minibatch_size)
                updates += 1
                version += 1
                total_games += len(rows)
                stats.update(update=updates, policy_version=version, games=len(rows),
                             total_games=total_games, trajectories=len(trajectories),
                             decisions=len(arrays["returns"]), rollout_seconds=seconds,
                             update_seconds=time.monotonic() - update_started,
                             environment_steps_per_second=len(rows) * 719 / seconds,
                             elapsed_seconds=elapsed(),
                             sampled_score=float(np.mean([r["win"] for r in rows])))
                stats["games_by_opponent"] = {
                    name: sum(row["opponent"] == name for row in rows)
                    for name in sorted({row["opponent"] for row in rows})}
                append_rows(directory / "updates.jsonl", [stats])
                print(json.dumps({"event": "update", **stats}), flush=True)
                policy_hash = publish()
                checkpoint("last")
                if elapsed() - last_development >= args.development_minutes * 60:
                    evaluate_development(policy_hash)
                    checkpoint("last", latest_development_score)
            # Evaluate the last completed policy once, even after an early stop.
            if latest_development_score is None or (directory / "last.json").exists() and json.loads((directory / "last.json").read_text())["metadata"]["development_score"] is None:
                evaluate_development(policy_hash)
            checkpoint("last", latest_development_score)
    atomic_json(directory / "completed.json", {
        "updates": updates, "training_games": total_games,
        "best_development_score": best_score, "last_development_score": latest_development_score,
        "elapsed_seconds": elapsed(), "status": "complete",
        "best_weights": str(directory / "best.json"), "last_weights": str(directory / "last.json")})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # The previous completed update is already durable in last.pt. Never
        # label an exception or a partially collected batch as successful.
        if "--run-dir" in sys.argv:
            destination = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            atomic_json(destination / "stopped.json", {
                "status": "budget_exhausted" if isinstance(error, BudgetExpired) else "failed",
                "error": repr(error), "traceback": traceback.format_exc(),
                "last_completed_checkpoint": str(destination / "last.pt"),
                "at_unix": time.time()})
        raise
