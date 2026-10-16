#!/usr/bin/env python3
"""Bounded RGB training followed by measured evaluation, without motor commands."""

import argparse
import json
import os
import subprocess
import signal
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--task", choices=["lift", "place"], default="lift")
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    parser.add_argument("--box-jitter", type=float, default=0.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--checkpoint", help="Resume from a saved compatible checkpoint")
    args = parser.parse_args()
    if min(args.envs, args.iterations, args.episodes) < 1:
        parser.error("Environment, iteration and episode counts must be positive")
    if not 0 <= args.cube_jitter <= .04 or not 0 <= args.box_jitter <= .04:
        parser.error("Training/evaluation reset jitter must be between 0 and 0.04 metres")
    curriculum = "fixed" if args.task == "lift" else "place"
    start_iteration = 0
    if args.checkpoint:
        import torch
        resume_path = Path(args.checkpoint).resolve()
        if not resume_path.is_file():
            parser.error("Resume checkpoint does not exist")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        start_iteration = int(saved["iter"])
        del saved
    session = ROOT / "outputs/training" / datetime.now().strftime("live_%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    log_root = ROOT / f"logs/rsl_rl/so101_pick{args.task}_rgb"
    before = set(log_root.glob("*"))
    state = {"session": str(session), "phase": "training", "pid": os.getpid(),
             "envs": args.envs, "iterations": args.iterations, "episodes": args.episodes,
             "hardware_commands": False, "task": args.task,
             "resume_checkpoint": args.checkpoint, "start_iteration": start_iteration,
             "cube_jitter_m": args.cube_jitter, "box_jitter_m": args.box_jitter}
    stop_requested = False
    active_child = None

    def stop_session(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if active_child is not None and active_child.poll() is None:
            active_child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop_session)
    signal.signal(signal.SIGTERM, stop_session)

    def status(**updates):
        state.update(updates)
        temporary = session / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(session / "status.json")
        print(json.dumps(state), flush=True)

    def run(command, filename):
        nonlocal active_child
        with (session / filename).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active_child = child
            status(child_pid=child.pid, command=command)
            result = child.wait()
            active_child = None
            return 130 if stop_requested else result

    status()
    training = [sys.executable, "-u", "scripts/train.py", "--task", args.task,
                "--curriculum", curriculum, "--policy", "rgb", "--envs", str(args.envs),
                "--iterations", str(args.iterations), "--device", "cuda:0", "--dashboard",
                "--cube-jitter", str(args.cube_jitter), "--box-jitter", str(args.box_jitter)]
    training += ["--headless"] if args.headless else ["--viz", "newton", "--max_visible_envs", "1"]
    if args.checkpoint:
        training += ["--checkpoint", str(resume_path)]
    code = run(training, "training.log")
    if code:
        status(phase="stopped_by_user" if stop_requested else "training_failed", returncode=code)
        return code
    created = [p for p in log_root.glob("*") if p.is_dir() and p not in before]
    if len(created) != 1:
        status(phase="checkpoint_selection_failed", new_runs=[str(p) for p in created])
        return 1
    checkpoints = list(created[0].glob("model_*.pt"))
    if not checkpoints:
        status(phase="checkpoint_missing", run_dir=str(created[0]))
        return 1
    checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
    if int(checkpoint.stem.split("_")[-1]) < start_iteration + args.iterations - 1:
        status(phase="training_incomplete", checkpoint=str(checkpoint), returncode=1)
        return 1
    status(phase="evaluation", run_dir=str(created[0]), checkpoint=str(checkpoint))
    evaluation = [sys.executable, "-u", "scripts/eval_pipeline.py", "--task", args.task,
                  "--curriculum", curriculum, "--policy", "rgb", "--checkpoint", str(checkpoint),
                  "--episodes", str(args.episodes), "--seed", "101",
                  "--cube-jitter", str(args.cube_jitter), "--box-jitter", str(args.box_jitter),
                  "--output", str(session / "evaluation.json")]
    if args.record_video:
        evaluation.append("--record-video")
    evaluation += ["--headless"] if args.headless else ["--viz", "kit"]
    code = run(evaluation, "evaluation.log")
    if code:
        status(phase="stopped_by_user" if stop_requested else "evaluation_failed", returncode=code)
        return code
    report = json.loads((session / "evaluation.json").read_text())
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(created[0] / "independent_evaluation")) as writer:
        step = int(checkpoint.stem.split("_")[-1])
        metric = "placement_success_rate" if args.task == "place" else "held_lift_success_rate"
        writer.add_scalar("Evaluation/" + metric, report["success_rate"], step)
        writer.add_scalar("Evaluation/mean_table_slide_cm", report["mean_table_slide_cm"], step)
        writer.add_scalar("Evaluation/mean_peak_lift_cm", report["mean_peak_lift_cm"], step)
        writer.add_scalar("Evaluation/episodes", report["episodes"], step)
    status(phase="complete", returncode=0, success_rate=report["success_rate"],
           mean_table_slide_cm=report["mean_table_slide_cm"],
           evaluation=str(session / "evaluation.json"), video=str(session / "evaluation.mp4") if args.record_video else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
