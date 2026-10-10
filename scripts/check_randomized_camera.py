#!/usr/bin/env python3
"""Test camera-guided reference grasps at seeded randomized cube reset poses."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--cube-jitter", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grasp-assist", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--viz", choices=["kit", "newton_gl"])
    parser.add_argument("--output", default="outputs/perception/randomized_camera")
    args = parser.parse_args()
    if args.cases < 1 or not 0 < args.cube_jitter <= 0.04:
        parser.error("Use positive cases and jitter in (0, 0.04] metres")
    rng = np.random.default_rng(args.seed)
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.cases):
        xy = rng.uniform(-args.cube_jitter, args.cube_jitter, 2)
        yaw = float(rng.uniform(-0.15, 0.15))
        target = folder / f"case_{index}" / "result.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/camera_pick_lift.py"),
            "--device",
            args.device,
            "--cube-offset",
            *map(str, xy),
            "--cube-yaw",
            str(yaw),
            "--output",
            str(target),
        ]
        command += ["--viz", args.viz] if args.viz else ["--headless"]
        if args.grasp_assist:
            command.append("--grasp-assist")
        if args.record_video:
            command.append("--record-video")
        with (folder / f"case_{index}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        report = json.loads(target.read_text()) if target.exists() else {}
        record = {
            "case": index,
            "cube_offset_m": xy.tolist(),
            "cube_yaw_rad": yaw,
            "passed": result.returncode == 0 and report.get("passed", False),
            "exit_code": result.returncode,
            "report": str(target),
            "bilateral_contact_fraction": report.get("bilateral_contact_fraction"),
        }
        records.append(record)
        print(record, flush=True)
    summary = {
        "controller": "camera-guided IK reference, not learned PPO",
        "seed": args.seed,
        "cube_jitter_m": args.cube_jitter,
        "cases": args.cases,
        "success_rate": sum(r["passed"] for r in records) / args.cases,
        "results": records,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(r["passed"] for r in records) else 2


if __name__ == "__main__":
    sys.exit(main())
