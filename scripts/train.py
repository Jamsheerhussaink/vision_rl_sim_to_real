#!/usr/bin/env python3
"""
Simple & Organized Training Launcher for SO-101 Vision RL.

Features:
- Prioritized Pick-and-Lift upwards with anti-drag penalties
- Uncluttered, in-place live Rich terminal dashboard (progress bar, rewards, losses, SPS, ETA)
- Supports both Stage 1 (Pick & Lift) and Stage 2 (Pick, Lift & Place)
- Automatic suppression of noisy Omniverse/PhysX/carb startup warnings
- Optional TensorBoard dashboard launcher (--dashboard)

Usage:
  python scripts/train.py                       # Train Priority 1: Pick & Lift (32 envs, 2000 iters)
  python scripts/train.py --task place          # Train Full Pick & Place (with gated placement)
  python scripts/train.py --task lift --viz     # Interactive 3D rendering window (Omniverse Kit)
  python scripts/train.py --dashboard           # Launch with TensorBoard web UI (http://localhost:6006)
"""

import argparse
import os
import subprocess
import sys

# Auto-detect and switch to env_isaaclab Python if executed in another conda environment (e.g. vision_rl)
try:
    import isaaclab  # noqa: F401
except ImportError:
    isaac_py = "/home/ai/miniconda3/envs/env_isaaclab/bin/python"
    if os.path.exists(isaac_py) and sys.executable != isaac_py:
        os.execv(isaac_py, [isaac_py] + sys.argv)

# Ensure isaac_vision_rl package is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Activate clean terminal UI
from isaac_vision_rl.utils.ui import setup_clean_terminal_ui  # noqa: E402

setup_clean_terminal_ui()


def launch_tensorboard(log_dir: str):
    """Launch TensorBoard daemon in background."""
    try:
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n[INFO] TensorBoard Web Dashboard launched at: http://localhost:6006 (PID: {proc.pid})")
    except Exception as e:
        print(f"[Warning] Could not start TensorBoard automatically: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch SO-101 Vision RL Training with Clean Terminal Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lift",
        choices=["lift", "place"],
        help="Task curriculum stage: 'lift' (Pick & Lift upwards, anti-drag) or 'place' (Full Pick & Place)",
    )
    parser.add_argument(
        "--envs",
        "--num_envs",
        type=int,
        default=32,
        dest="num_envs",
        help="Number of parallel environments to simulate",
    )
    parser.add_argument(
        "--iterations",
        "--max_iterations",
        type=int,
        default=2000,
        dest="max_iterations",
        help="Maximum PPO training iterations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--viz",
        type=str,
        nargs="?",
        const="kit",
        default=None,
        choices=["kit", "newton", "newton_gl", "rerun", "viser"],
        help="Launch with 3D visualizer ('kit' for Omniverse Kit, 'newton' for Newton OpenGL)",
    )
    parser.add_argument("--gui", action="store_true", help="Launch with 3D visualizer GUI (defaults to kit)")
    parser.add_argument("--newton", action="store_true", help="Launch with Newton OpenGL visualizer GUI")
    parser.add_argument(
        "--max_visible_envs",
        type=int,
        default=None,
        help="Maximum number of environments rendered in GUI visualizer (default: show all)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a model checkpoint (.pt)")
    parser.add_argument(
        "--dashboard", action="store_true", help="Launch background TensorBoard web dashboard (http://localhost:6006)"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device (e.g. cuda:0 or cpu)")
    parser.add_argument(
        "--headless", action="store_true", help="Run without a simulator window (camera rendering remains enabled)"
    )

    parser.add_argument(
        "--curriculum",
        choices=["grasp", "fixed", "lift", "place"],
        default=None,
        help="Initial-state difficulty; managed automatically by train_pipeline.py",
    )
    parser.add_argument(
        "--policy",
        choices=["vision", "state"],
        default="vision",
        help="Actor input: camera geometry plus proprioception; vision also adds RGB",
    )
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres, 0 to 0.05")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY reset jitter in metres, 0 to 0.05")
    parser.add_argument("--smooth-control", action="store_true", help="Experimental lift controller: bounded acceleration, slow wrist roll and ramped closure")
    parser.add_argument("--wrist-only-control", action="store_true", help="Keep baseline approach and closure; hold measured wrist roll after capture")
    args = parser.parse_args()
    if args.wrist_only_control and (args.smooth_control or args.task != "lift"):
        parser.error("--wrist-only-control requires lift and cannot be combined with --smooth-control")
    os.environ["SO101_WRIST_ONLY_CONTROL"] = "1" if args.wrist_only_control else "0"
    if args.smooth_control and args.task != "lift":
        parser.error("--smooth-control currently supports lift only")
    os.environ["SO101_SMOOTH_CONTROL"] = "1" if args.smooth_control else "0"
    if not 0 <= args.cube_jitter <= 0.05:
        parser.error("--cube-jitter must be between 0 and 0.05 metres")
    if not 0 <= args.box_jitter <= 0.05:
        parser.error("--box-jitter must be between 0 and 0.05 metres")
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum or ("place" if args.task == "place" else "lift")

    # Map task name to registered Gym environment
    if args.task == "lift":
        task_id = "IsaacVisionRl-PickLift-SO101"
        exp_name = f"so101_picklift_{args.policy}"
    else:
        task_id = "IsaacVisionRl-Pickplace-SO101"
        exp_name = f"so101_pickplace_{args.policy}"

    # Log directory root
    log_root = os.path.join(ROOT_DIR, "logs", "rsl_rl", exp_name)
    os.makedirs(log_root, exist_ok=True)

    if args.dashboard:
        launch_tensorboard(log_root)

    # Determine visualizer backend
    viz_backend = None
    if args.newton:
        viz_backend = "newton"
    elif args.viz is not None:
        viz_backend = args.viz
    elif args.gui:
        viz_backend = "kit"

    # Build isaaclab arguments
    train_args = [
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        task_id,
        "--experiment_name",
        exp_name,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    if viz_backend is not None:
        train_args.extend(["--viz", viz_backend])
        if args.max_visible_envs is not None:
            train_args.extend(["--max_visible_envs", str(args.max_visible_envs)])
        if viz_backend in ("newton", "newton_gl"):
            # Show the actual policy RGB in Newton's camera panel, beside its GL scene.
            train_args.extend(
                [
                    "env.sim.default_visualizer_cfg.streaming_view=true",
                    "env.sim.default_visualizer_cfg.streaming_envs=[0]",
                ]
            )
    # Isaac Lab infers headless execution when no Kit visualizer is selected.
    # Camera rendering is selected automatically because this task contains CameraCfg.

    if args.checkpoint:
        train_args.extend(["--checkpoint", args.checkpoint])

    # Run training using Isaac Lab CLI dispatcher
    from isaaclab.cli import cli

    sys.argv = [sys.argv[0]] + train_args
    sys.exit(cli())


if __name__ == "__main__":
    main()
