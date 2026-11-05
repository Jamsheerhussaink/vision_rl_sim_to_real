"""Evaluate whether a saved policy benefits from its camera images.

    python scripts/evaluate.py --run runs/first_lift40_live --episodes 100

Uses deterministic actions and matched initial states for normal, frozen and
shuffled images. Writes a new output directory; never trains or edits a run.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime

MODES = ("normal", "frozen", "shuffled")
ROOT = Path(__file__).resolve().parents[1]


def camera_input(mode, current, initial, xp):
    """Only pixels change. Roll by one gives every world a different donor."""
    return xp.where(mode == 1, initial,
                    xp.where(mode == 2, xp.roll(current, 1, axis=0), current))


def advance_streak(current, longest, condition, active, xp):
    """Count consecutive valid steps, ignoring auto-reset episodes."""
    current = xp.where(active, xp.where(condition, current + 1, 0), current)
    return current, xp.maximum(longest, current)


def arguments():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", help="filename in --run; defaults to latest.txt")
    ap.add_argument("--episodes", type=int, default=100, help="episodes PER condition")
    ap.add_argument("--worlds", type=int, default=4, help="parallel worlds; at least 2 for shuffling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-seconds", type=float, default=1.0)
    ap.add_argument("--output", type=Path, help="new directory for results (must not exist)")
    a = ap.parse_args()
    if a.episodes < 1 or a.worlds < 2 or not math.isfinite(a.hold_seconds) or a.hold_seconds <= 0:
        ap.error("episodes must be positive, worlds >= 2, and hold-seconds finite and positive")
    return a


def main():
    a = arguments()
    # Set before importing JAX/Warp. No display or desktop viewer is used.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(ROOT))
    import warp as wp
    wp.config.quiet = True
    wp.config.log_level = wp.LOG_WARNING
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from vision_rl.train import build, load_config

    cfg = load_config(str(a.run))
    checkpoint = a.run / (a.ckpt or (a.run / "latest.txt").read_text().strip())
    raw = checkpoint.read_bytes()
    saved = serialization.msgpack_restore(raw)
    params = jax.tree.map(jnp.asarray, saved["params"])
    cfg = dataclasses.replace(cfg, ppo=dataclasses.replace(cfg.ppo, num_envs=a.worlds))
    required_steps = math.ceil(a.hold_seconds / cfg.env.ctrl_dt)
    if required_steps > cfg.env.episode_steps:
        raise ValueError("Requested hold duration exceeds the episode length")
    out = a.output or a.run / ("vision_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint: {checkpoint} ({saved.get('steps', '?'):,} steps)", flush=True)
    print(f"{a.episodes} matched episodes per condition; {a.worlds} parallel worlds; mean actions.", flush=True)
    print(f"Sustained lift: >= {cfg.env.reward.pick_height * 100:g} cm for "
          f"{required_steps} steps ({required_steps * cfg.env.ctrl_dt:.2f} s).", flush=True)
    print("Strict metric additionally requires opposite-side finger geometry, not measured grip force.", flush=True)
    print("Uses CURRENT scene/code with the saved config; historical scene equivalence is not guaranteed.", flush=True)
    venv, net = build(cfg)
    reset = jax.jit(venv.reset)
    names = ("reward_return", "held_steps", "brief_success", "max_lift_m", "steps",
             "hold_streak", "max_hold_streak", "strict_streak", "max_strict_streak", "unstable", "fell")

    @jax.jit
    def evaluate(vs, mode):
        initial = vs.frames
        stats = {k: jnp.zeros(a.worlds) for k in names}

        def step(carry, _):
            vs, active, stats = carry
            obs = {"pixels": camera_input(mode, vs.frames, initial, jnp),
                   "proprio": vs.env.obs}
            pi, _ = net.apply(params, obs)
            nxt = venv.step(vs, pi.mean)
            m = nxt.env.metrics  # terminal metrics describe PRE-reset physics
            held = m["held"] > 0.5
            high = m["cube_lift"] >= cfg.env.reward.pick_height
            valid = (m["unstable"] < 0.5) & (m["fell"] < 0.5)
            holding = held & high & valid
            strict = holding & (m["opposing_sides"] > 0.5)
            stats = dict(stats)
            stats["hold_streak"], stats["max_hold_streak"] = advance_streak(
                stats["hold_streak"], stats["max_hold_streak"], holding, active, jnp)
            stats["strict_streak"], stats["max_strict_streak"] = advance_streak(
                stats["strict_streak"], stats["max_strict_streak"], strict, active, jnp)
            for key, value in (("reward_return", nxt.env.reward), ("held_steps", held), ("steps", 1)):
                stats[key] += jnp.where(active, value, 0)
            for key, value in (("brief_success", m["success"]), ("max_lift_m", m["cube_lift"]),
                               ("unstable", m["unstable"]), ("fell", m["fell"])):
                stats[key] = jnp.maximum(stats[key], jnp.where(active, value, 0))
            active = active & (nxt.env.done < 0.5)
            return (nxt, active, stats), None

        (_, _, stats), _ = jax.lax.scan(
            step, (vs, jnp.ones(a.worlds, dtype=bool), stats), None, length=cfg.env.episode_steps)
        return stats

    records = []
    batches = math.ceil(a.episodes / a.worlds)
    for batch in range(batches):
        # One reset is reused by every condition: exact same physics and images.
        key = jax.random.fold_in(jax.random.PRNGKey(a.seed), batch)
        initial = reset(key)
        qpos = np.asarray(initial.env.data.qpos)
        fingerprints = [hashlib.sha256(q.tobytes()).hexdigest() for q in qpos]
        for mode_id, mode in enumerate(MODES):
            values = jax.device_get(evaluate(initial, jnp.asarray(mode_id)))
            for world in range(min(a.worlds, a.episodes - batch * a.worlds)):
                record = {"condition": mode, "episode": batch * a.worlds + world,
                          "initial_state_sha256": fingerprints[world]}
                record.update({k: float(v[world]) for k, v in values.items()})
                record["sustained_held_lift"] = int(record["max_hold_streak"] >= required_steps)
                record["strict_sustained_lift"] = int(record["max_strict_streak"] >= required_steps)
                records.append(record)
            print(f"Batch {batch + 1}/{batches}: {mode} complete", flush=True)

    with (out / "episodes.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    print("\nCondition   Brief success   Sustained held   Strict sustained   Any held")
    for mode in MODES:
        rows = [r for r in records if r["condition"] == mode]
        summary[mode] = {
            "episodes": len(rows),
            "brief_success_rate": float(np.mean([r["brief_success"] for r in rows])),
            "sustained_held_lift_rate": float(np.mean([r["sustained_held_lift"] for r in rows])),
            "strict_sustained_lift_rate": float(np.mean([r["strict_sustained_lift"] for r in rows])),
            "any_held_rate": float(np.mean([r["held_steps"] > 0 for r in rows])),
            "mean_return": float(np.mean([r["reward_return"] for r in rows])),
            "mean_max_lift_m": float(np.mean([r["max_lift_m"] for r in rows])),
        }
        s = summary[mode]
        print(f"{mode:10s} {s['brief_success_rate']:12.1%} {s['sustained_held_lift_rate']:16.1%} "
              f"{s['strict_sustained_lift_rate']:18.1%} {s['any_held_rate']:10.1%}")
    comparisons = {}
    baseline = [r for r in records if r["condition"] == "normal"]
    rng = np.random.default_rng(a.seed)
    for mode in MODES[1:]:
        other = [r for r in records if r["condition"] == mode]
        # Paired bootstrap: both conditions use the same sampled episode indices.
        delta = np.array([n["sustained_held_lift"] - o["sustained_held_lift"]
                          for n, o in zip(baseline, other)])
        boot = np.mean(delta[rng.integers(0, len(delta), size=(2000, len(delta)))], axis=1)
        comparisons[mode] = {"normal_minus_condition_percentage_points": float(delta.mean() * 100),
                             "paired_bootstrap_95_percent_interval_pp":
                             (np.quantile(boot, [0.025, 0.975]) * 100).tolist()}
    metadata = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
                "training_steps": int(saved.get("steps", 0)), "seed": a.seed, "worlds": a.worlds,
                "hold_seconds_actual": required_steps * cfg.env.ctrl_dt,
                "evaluation_config": cfg.to_dict(), "results": summary, "paired_comparisons": comparisons,
                "asset_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "vision_rl/envs/assets").glob("*.xml"))}}
    (out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nNormal > frozen suggests updated images help; normal > shuffled suggests matched images help.")
    print("Shuffling also changes visible robot state; this does not isolate cube recognition.")
    print("Low normal success makes the comparison inconclusive. Similar scores do not prove vision is unused.")
    print("Bootstrap intervals are exploratory; episodes in each shuffled batch share donor images.")
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
