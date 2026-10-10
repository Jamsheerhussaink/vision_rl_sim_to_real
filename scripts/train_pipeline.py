#!/usr/bin/env python3
"""Evidence-gated pipeline: physical grasp -> easy grasp -> fixed reach -> randomized lift -> placement."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import isaaclab  # noqa: F401
except ImportError:
    interpreter = "/home/ai/miniconda3/envs/env_isaaclab/bin/python"
    if Path(interpreter).exists() and sys.executable != interpreter:
        os.execv(interpreter, [interpreter, *sys.argv])

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isaac_vision_rl.utils.pipeline import (  # noqa: E402
    SEEDS,
    STAGES,
    fingerprint,
    read_physical_report,
    require_camera_preflight,
    require_gate,
    write_gate,
)

OUTPUT = ROOT / "outputs/pipeline"


def run(command, log_name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    with (OUTPUT / log_name).open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if result:
        raise RuntimeError(f"Check failed (exit {result}); inspect {OUTPUT / log_name}. Pipeline stopped.")


def evaluate(stage, checkpoint, args):
    reports = []
    label = stage if args.policy == "vision" else f"{stage}_state"
    for seed in SEEDS:
        target = OUTPUT / f"{label}_eval_{seed}.json"
        run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/eval_pipeline.py"),
                "--task",
                "place" if stage == "place" else "lift",
                "--curriculum",
                stage,
                "--policy",
                args.policy,
                "--cube-jitter",
                str(args.cube_jitter),
                "--box-jitter",
                str(args.box_jitter),
                "--checkpoint",
                checkpoint,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--headless",
                "--record-video",
                "--output",
                str(target),
            ],
            f"{label}_eval_{seed}.log",
        )
        reports.append(json.loads(target.read_text()))
    gate = write_gate(stage, checkpoint, reports, args.episodes)
    print(
        f"{stage}: {'PASSED' if gate['passed'] else 'FAILED'}; success rates {[r['success_rate'] for r in reports]}",
        flush=True,
    )
    if not gate["passed"]:
        raise RuntimeError("Mastery gate failed. Inspect playback and diagnostics; no next-stage training was started.")


def train(stage, args):
    read_physical_report()
    require_camera_preflight()
    previous = STAGES[STAGES.index(stage) - 1] if stage != "grasp" else None
    inherited = require_gate(previous, args.policy) if previous else None
    checkpoint = args.checkpoint or inherited
    exp = f"so101_pickplace_{args.policy}" if stage == "place" else f"so101_picklift_{args.policy}"
    parent = ROOT / "logs/rsl_rl" / exp
    before = set(parent.iterdir()) if parent.exists() else set()
    training_fingerprint = fingerprint()
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train.py"),
        "--task",
        "place" if stage == "place" else "lift",
        "--curriculum",
        stage,
        "--envs",
        str(args.envs),
        "--iterations",
        str(args.place_iterations if stage == "place" else args.lift_iterations),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if checkpoint:
        command += ["--checkpoint", checkpoint]
    command += ["--viz", args.viz, "--max_visible_envs", str(args.max_visible_envs)] if args.viz else ["--headless"]
    if args.dashboard:
        command += ["--dashboard"]
    command += ["--policy", args.policy, "--cube-jitter", str(args.cube_jitter)]
    label = stage if args.policy == "vision" else f"{stage}_state"
    run(command, f"{label}_training.log")
    if fingerprint() != training_fingerprint:
        raise RuntimeError(
            "Training configuration changed during the run; evaluation is blocked. Restart with a fixed configuration."
        )
    runs = [p for p in parent.iterdir() if p not in before and p.is_dir() and list(p.glob("model_*.pt"))]
    if len(runs) != 1:
        raise RuntimeError(
            "Cannot identify exactly one new training run; no arbitrary latest checkpoint will be selected."
        )
    candidate = max(runs[0].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    evaluate(stage, str(candidate), args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=["preflight", "status", "grasp", "fixed", "lift", "place", "all", "evaluate"],
        default="preflight",
    )
    p.add_argument("--evaluate-stage", choices=STAGES, default="lift")
    p.add_argument("--checkpoint", help="Explicit candidate to evaluate or resume; previous-stage gates still apply")
    p.add_argument("--policy", choices=["vision", "state"], default="vision")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--lift-iterations", type=int, default=2000)
    p.add_argument("--place-iterations", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viz", choices=["kit", "newton", "newton_gl"])
    p.add_argument("--max_visible_envs", type=int, default=1, help="Limit visible copies; all --envs still train")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres, 0 to 0.04")
    args = p.parse_args()
    if not 0 <= args.cube_jitter <= 0.04:
        p.error("--cube-jitter must be between 0 and 0.04 metres")
    if args.episodes < 50 or args.envs < 1 or min(args.lift_iterations, args.place_iterations) < 1:
        p.error("Use at least 50 evaluation episodes per seed, positive envs and positive iteration counts.")
    if args.viz and args.headless:
        p.error("Choose a visualizer or --headless.")
    if args.max_visible_envs < 1:
        p.error("--max_visible_envs must be positive.")
    try:
        if args.stage == "status":
            try:
                read_physical_report()
                print("Physical grasp: PASSED")
            except ValueError as error:
                print("Physical grasp:", error)
            for stage in STAGES:
                try:
                    print(stage, "PASSED:", require_gate(stage))
                except ValueError as error:
                    print(stage, error)
            return 0
        if args.stage in ("preflight", "all"):
            run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], "unit_checks.log")
            run(
                [sys.executable, "-u", str(ROOT / "scripts/check_workcell.py"), "--device", args.device],
                "workcell_physics.log",
            )
            if not json.loads((OUTPUT / "workcell_physics.json").read_text()).get("passed"):
                raise RuntimeError("Table/box collision preflight failed. Training remains blocked.")
            run(
                [sys.executable, "-u", str(ROOT / "scripts/check_resets.py"), "--device", args.device],
                "reset_check.log",
            )
            if not json.loads((OUTPUT / "reset_check.json").read_text()).get("passed"):
                raise RuntimeError("Cube pose/velocity reset preflight failed. Training remains blocked.")
            run(
                [sys.executable, "-u", str(ROOT / "scripts/verify_grasp.py"), "--device", args.device],
                "physical_grasp.log",
            )
            read_physical_report()
            run(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/verify_camera_perception.py"),
                    "--headless",
                    "--device",
                    args.device,
                ],
                "camera_perception.log",
            )
            run(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/camera_pick_lift.py"),
                    "--headless",
                    "--grasp-assist",
                    "--record-video",
                    "--device",
                    args.device,
                ],
                "camera_reference_lift.log",
            )
            require_camera_preflight()
            print("Physical preflight passed. Ready for the grasp curriculum.", flush=True)
            if args.stage == "preflight":
                return 0
        if args.stage == "evaluate":
            read_physical_report()
            index = STAGES.index(args.evaluate_stage)
            if index:
                require_gate(STAGES[index - 1], args.policy)
            if not args.checkpoint:
                p.error("--stage evaluate requires an explicit --checkpoint")
            evaluate(args.evaluate_stage, str(Path(args.checkpoint).resolve()), args)
        elif args.stage == "all":
            if args.checkpoint:
                p.error("--stage all starts fresh; use an individual stage to resume.")
            for stage in STAGES:
                train(stage, args)
        else:
            train(args.stage, args)
        return 0
    except (ValueError, RuntimeError) as error:
        print(f"PIPELINE STOPPED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
