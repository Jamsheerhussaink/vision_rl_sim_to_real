#!/usr/bin/env python3
"""Run one-at-a-time, recorded evaluations for the simple-reward RGB policy."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def summarize(task, checkpoint, seeds, episodes):
    per_episode = []
    for row in episodes:
        trace = row.get("trace", [])
        measures = [sample.get("measurements", {}) for sample in trace]
        lift_rises = [float(m["lift_rise"]) for m in measures if "lift_rise" in m and math.isfinite(float(m["lift_rise"]))]
        clearances = [float(m["clearance"]) for m in measures if "clearance" in m and math.isfinite(float(m["clearance"]))]
        slides = [float(m["slide_distance"]) for m in measures if "slide_distance" in m and math.isfinite(float(m["slide_distance"]))]
        verified_grasp = any(
            bool(m.get("bilateral", False)) and bool(m.get("enclosed", False))
            and float(m.get("jaw", float("inf"))) < 1.35
            for m in measures
        )
        per_episode.append({
            "seed": row["seed"],
            "success": bool(row["success"]),
            "verified_two_finger_grasp": verified_grasp,
            "held_lift_success": bool(row["success"]) if task == "lift" else None,
            "end_reason": row.get("end_reason"),
            "peak_lift_cm": max(lift_rises, default=0.0) * 100.0,
            "table_slide_cm": max(slides, default=0.0) * 100.0,
            "minimum_clearance_mm": min(clearances) * 1000.0 if clearances else None,
            "bilateral_contact_samples": sum(bool(m.get("bilateral", False)) for m in measures),
            "json": row.get("json"),
            "video": row.get("video"),
        })
    count = len(per_episode)
    successes = sum(item["success"] for item in per_episode)
    lift_gate = math.ceil(.8 * count) if task == "lift" else None
    return {
        "task": task,
        "checkpoint": str(Path(checkpoint).resolve()),
        "episodes": count,
        "seeds": seeds,
        "successes": successes,
        "success_rate": successes / max(1, count),
        "verified_two_finger_grasps": sum(x["verified_two_finger_grasp"] for x in per_episode),
        "held_lift_successes": sum(bool(x["held_lift_success"]) for x in per_episode) if task == "lift" else None,
        "lift_gate_required_successes": lift_gate,
        "lift_gate_passed": (successes >= lift_gate) if lift_gate is not None else None,
        "placement_gate": "report_only" if task == "place" else None,
        "mean_peak_lift_cm": sum(x["peak_lift_cm"] for x in per_episode) / max(1, count),
        "mean_table_slide_cm": sum(x["table_slide_cm"] for x in per_episode) / max(1, count),
        "minimum_clearance_mm": min(
            (x["minimum_clearance_mm"] for x in per_episode if x["minimum_clearance_mm"] is not None),
            default=None,
        ),
        "episode_results": per_episode,
        "note": "Reward is not task success. Review each video and the contact, hold, slide, and clearance measurements.",
    }


def training_processes():
    found = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return found
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if any(script in command for script in ("scripts/train.py", "scripts/train_simple_rewards.py", "scripts/train_curriculum.py")):
            found.append({"pid": int(entry.name), "command": command.strip()})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("lift", "place"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

    import torch
    from isaac_vision_rl.utils.real_policy import validate_real_checkpoint_metadata

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        validate_real_checkpoint_metadata(saved)
    except ValueError as error:
        parser.error(str(error))
    if saved.get("so101_reward_version") != "simple_rewards_v5_pure_ppo":
        parser.error("This evaluator is only for simple_rewards_v5_pure_ppo checkpoints")
    if saved.get("so101_training_goal") != args.task:
        parser.error(f"Checkpoint goal is {saved.get('so101_training_goal')!r}; requested evaluation is {args.task!r}")
    del saved

    active = training_processes()
    if active:
        parser.error("Training is running; finish or stop it before evaluation to avoid GPU/RAM contention")

    default_dir = ROOT / "outputs/simple_rewards/evaluations" / f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    folder = Path(args.output).resolve() if args.output else default_dir
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite evaluation directory: {folder}")

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    results = []
    evaluator = ROOT / "scripts/manual_simple_rewards.py"
    for index, seed in enumerate(seeds, start=1):
        output = folder / f"episode_{index:02d}_seed_{seed}.json"
        log_path = folder / f"episode_{index:02d}_seed_{seed}.log"
        command = [
            sys.executable, str(evaluator), "--mode", "evaluate", "--headless",
            "--checkpoint", str(checkpoint), "--seed", str(seed),
            "--record-video", "--output", str(output), "--device", args.device,
        ]
        print(f"[{index}/{len(seeds)}] Evaluating seed {seed}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        result["json"] = str(output)
        video_path = output.with_suffix(".mp4")
        result["video"] = str(video_path) if video_path.is_file() else None
        results.append(result)

    report = summarize(args.task, checkpoint, seeds, results)
    report_path = folder / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episode_results"}, indent=2, allow_nan=False))
    print(f"Full episode results and videos: {folder}")
    print(f"Summary: {report_path}")


if __name__ == "__main__":
    main()
