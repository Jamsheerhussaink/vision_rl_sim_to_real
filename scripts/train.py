#!/usr/bin/env python3
"""
Simple & Organized Training Launcher for SO-101 Vision RL.

Features:
- Prioritized Pick-and-Lift upwards with anti-drag penalties
- Standard default RSL-RL console logging table (optional Rich dashboard with --custom-ui)
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
import socket
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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)




def launch_tensorboard(log_dir: str):
    """Launch TensorBoard daemon in background."""
    try:
        # Keep an existing dashboard alive; use another port for this experiment.
        port = 6006
        while port < 6016:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) != 0:
                    break
            port += 1
        if port == 6016:
            raise RuntimeError("No free dashboard port in 6006-6015")
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n[INFO] TensorBoard Web Dashboard launched at: http://localhost:{port} (PID: {proc.pid})")
    except Exception as e:
        print(f"[Warning] Could not start TensorBoard automatically: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch SO-101 Vision RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-log",
        action="store_true",
        help="Use raw multi-line RSL-RL table dump instead of clean streaming progress bar UI",
    )
    parser.add_argument(
        "--profile", choices=["rgb64"], default=None,
        help="Fresh tuned RGB fixed-lift run: 64 environments and 2000 updates; explicit envs/iterations override these",
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
        default=None,
        dest="num_envs",
        help="Number of parallel environments to simulate",
    )
    parser.add_argument(
        "--iterations",
        "--max_iterations",
        type=int,
        default=None,
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
        "--adapt-checkpoint",
        action="store_true",
        help="Allow resuming/fine-tuning a checkpoint trained at a different control frequency (e.g. 60 Hz -> 30 Hz)",
    )
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
        choices=["rgb"],
        default="rgb",
        help="RGB images and joint/controller readings",
    )
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres, 0 to 0.05")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY reset jitter in metres, 0 to 0.05")
    parser.add_argument("--fps", type=int, choices=[25, 30, 60], default=30, help="Policy and camera rate in Hz (default: 30)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height in pixels (default: 720)")
    args = parser.parse_args()
    args.num_envs = args.num_envs if args.num_envs is not None else (64 if args.profile == "rgb64" else 32)
    args.max_iterations = args.max_iterations if args.max_iterations is not None else 2000
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--envs and --iterations must be positive")
    if args.profile == "rgb64":
        if args.task != "lift" or args.policy != "rgb" or args.curriculum not in (None, "fixed"):
            parser.error("--profile rgb64 requires --task lift --policy rgb --curriculum fixed")
        if args.checkpoint:
            parser.error("--profile rgb64 starts fresh; use the ordinary launcher to resume a checkpoint")
        args.curriculum = "fixed"
    if not 0 <= args.cube_jitter <= 0.05:
        parser.error("--cube-jitter must be between 0 and 0.05 metres")
    if not 0 <= args.box_jitter <= 0.05:
        parser.error("--box-jitter must be between 0 and 0.05 metres")
    os.environ["SO101_RGB_FPS"] = str(args.fps)
    os.environ["SO101_CAMERA_WIDTH"] = str(args.width)
    os.environ["SO101_CAMERA_HEIGHT"] = str(args.height)
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum or ("place" if args.task == "place" else "lift")
    os.environ["SO101_ALLOW_FPS_MISMATCH"] = "1" if args.adapt_checkpoint else "0"

    # Map task name to registered Gym environment
    if args.task == "lift":
        task_id = "IsaacVisionRl-PickLift-SO101"
        exp_name = f"so101_picklift_{args.policy}"
    else:
        task_id = "IsaacVisionRl-Pickplace-SO101"
        exp_name = f"so101_pickplace_{args.policy}"
    if args.profile:
        exp_name += "_tuned64"

    # Log directory root
    log_root = os.path.join(ROOT_DIR, "logs", "rsl_rl", exp_name)
    os.makedirs(log_root, exist_ok=True)
    if not args.raw_log:
        from isaac_vision_rl.utils.ui import setup_clean_terminal_ui

        setup_clean_terminal_ui()

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
                    "env.sim.default_visualizer_cfg.background_color=[0.04,0.06,0.04]",
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
