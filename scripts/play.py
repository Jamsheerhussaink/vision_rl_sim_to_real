#!/usr/bin/env python3
"""
Interactive 3D Visual Playback & Video Recording for Trained SO-101 Vision RL Policy.

Usage:
  python scripts/play.py                                 # Watch latest trained model in 3D
  python scripts/play.py --video                         # Record MP4 video clip of the policy
  python scripts/play.py --checkpoint <path/to/model.pt> # Watch specific checkpoint
"""

import argparse
import glob
import os
import sys

# Auto-detect and switch to env_isaaclab Python if executed in another conda environment (e.g. vision_rl)
try:
    import isaaclab  # noqa: F401
except ImportError:
    isaac_py = "/home/ai/miniconda3/envs/env_isaaclab/bin/python"
    if os.path.exists(isaac_py) and sys.executable != isaac_py:
        os.execv(isaac_py, [isaac_py] + sys.argv)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def find_latest_checkpoint(task_hint: str | None = None, policy_hint: str = "rgb") -> tuple[str, str]:
    """Find the most recent checkpoint and return (checkpoint_path, task_id)."""
    search_dirs = []
    if task_hint == "lift":
        search_dirs = [f"so101_picklift_{policy_hint}"]
    elif task_hint == "place":
        search_dirs = [f"so101_pickplace_{policy_hint}"]
    else:
        search_dirs = [f"so101_picklift_{policy_hint}", f"so101_pickplace_{policy_hint}"]

    all_checkpoints = []
    for exp_dir in search_dirs:
        pattern = os.path.join(ROOT_DIR, "logs", "rsl_rl", exp_dir, "*", "model_*.pt")
        found = glob.glob(pattern)
        for cp in found:
            task = "IsaacVisionRl-PickLift-SO101" if "lift" in exp_dir else "IsaacVisionRl-Pickplace-SO101"
            all_checkpoints.append((cp, task))

    if not all_checkpoints:
        raise FileNotFoundError(
            "No trained checkpoints found in logs/rsl_rl/RGB training folders."
        )

    all_checkpoints.sort(key=lambda x: os.path.getmtime(x[0]))
    return all_checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Watch trained SO-101 robot pick & lift / place in 3D Isaac Sim")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file (defaults to latest)")
    parser.add_argument(
        "--task", type=str, default=None, choices=["lift", "place"], help="Task variant (auto-detected if None)"
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of visualization environments (1 recommended)")
    parser.add_argument("--video", action="store_true", help="Record MP4 video of the episode")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video in steps")
    parser.add_argument("--headless", action="store_true", help="Run without a simulator window")
    parser.add_argument("--curriculum", choices=["grasp", "fixed", "lift", "place"], default=None)
    parser.add_argument(
        "--policy",
        choices=["rgb"],
        default="rgb",
        help="RGB images and joint/controller readings",
    )
    parser.add_argument(
        "--viz",
        type=str,
        default="kit",
        choices=["kit", "newton", "newton_gl"],
        help="Visualizer backend ('kit' for Omniverse Kit RTX, 'newton_gl' for lightweight Newton OpenGL)",
    )
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres, 0 to 0.04")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY reset jitter in metres, 0 to 0.04")
    args = parser.parse_args()
    if not 0 <= args.cube_jitter <= 0.04:
        parser.error("--cube-jitter must be between 0 and 0.04 metres")
    if not 0 <= args.box_jitter <= 0.04:
        parser.error("--box-jitter must be between 0 and 0.04 metres")
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum or ("place" if args.task == "place" else "lift")

    if args.checkpoint:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if args.task == "lift" or "lift" in checkpoint_path:
            task_id = "IsaacVisionRl-PickLift-SO101"
        else:
            task_id = "IsaacVisionRl-Pickplace-SO101"
    else:
        checkpoint_path, task_id = find_latest_checkpoint(args.task, args.policy)

    print(f"\n[INFO] Loading trained checkpoint: {checkpoint_path}")
    print(f"[INFO] Task environment: {task_id}\n")

    # Build isaaclab play arguments
    play_args = [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        task_id,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        checkpoint_path,
        "--real-time",
    ]

    if not args.video and not args.headless:
        play_args.extend(["--viz", args.viz])
    else:
        if args.video:
            play_args.extend(["--video", "--video_length", str(args.video_length)])

    from isaaclab.cli import cli

    sys.argv = [sys.argv[0]] + play_args
    sys.exit(cli())


if __name__ == "__main__":
    main()
