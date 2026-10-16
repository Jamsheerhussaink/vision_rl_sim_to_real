#!/usr/bin/env python3
"""Hardware-in-the-Loop Co-Execution Streamer for SO-101 Robotic Arm.

Runs policy inference inside Isaac Sim using synthetic RGB camera rendering,
applies actions to the virtual arm, and simultaneously streams 60 Hz joint position
targets over UDP to the physical hardware bridge.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from isaac_vision_rl.utils.calibration import JOINT_ORDER
from isaac_vision_rl.utils.hardware_client import HardwareStreamer


def find_latest_checkpoint(task: str = "place", policy: str = "rgb") -> str:
    """Find the most recent checkpoint compatible with the given task and policy."""
    import torch

    folder = f"so101_pick{task}_{policy}"
    patterns = [
        str(ROOT_DIR / f"logs/rsl_rl/{folder}/*/model_*.pt"),
        str(ROOT_DIR / f"logs/rsl_rl/so101_picklift_rgb/*/model_*.pt"),
        str(ROOT_DIR / f"logs/rsl_rl/so101_pickplace_rgb/*/model_*.pt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in logs/rsl_rl for task {task}")

    valid = []
    for cp in files:
        try:
            meta = torch.load(cp, map_location="cpu")
            if policy == "rgb":
                if meta.get("so101_rgb_profile") == "424x240_letterbox128_v1":
                    valid.append((cp, os.path.getmtime(cp), meta.get("iter", 0)))
            else:
                valid.append((cp, os.path.getmtime(cp), meta.get("iter", 0)))
        except Exception:
            pass

    if valid:
        valid.sort(key=lambda x: (x[1], x[2]))
        return valid[-1][0]

    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "place"], default="place", help="Task variant")
    parser.add_argument("--policy", choices=["rgb"], default="rgb", help="Policy type")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (.pt)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--device", default="cuda:0", help="Simulation device")
    parser.add_argument("--viz", choices=["newton", "kit", "none"], default="newton", help="Visualizer backend")
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Target hardware bridge UDP host")
    parser.add_argument("--udp-port", type=int, default=9870, help="Target hardware bridge UDP port")
    parser.add_argument("--rate", type=float, default=60.0, help="Control rate in Hz (default 60)")
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY jitter in metres")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY jitter in metres")
    args = parser.parse_args()

    # Set task environment variables before importing simulation libraries
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.task
    os.environ["SO101_RGB_FPS"] = str(int(args.rate))

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.task, args.policy)
    checkpoint = Path(checkpoint_path).resolve()

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg

    is_place = args.task == "place"
    task_id = "IsaacVisionRl-Pickplace-SO101" if is_place else "IsaacVisionRl-PickLift-SO101"
    cfg = PickplaceEnvCfg() if is_place else PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device

    agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    agent.device = args.device

    viz_choice = None if args.headless or args.viz == "none" else args.viz

    print("\n" + "=" * 70)
    print("SO-101 Sim-to-Real Digital Twin Streamer")
    print(f"Task:       {task_id}")
    print(f"Policy:     {args.policy.upper()} (Synthetic RGB Camera)")
    print(f"Checkpoint: {checkpoint}")
    print(f"Stream:     UDP {args.udp_host}:{args.udp_port} @ {args.rate:.1f} Hz")
    print(f"Visualizer: {args.viz}")
    print("=" * 70 + "\n")

    streamer = HardwareStreamer(
        host=args.udp_host,
        port=args.udp_port,
        joint_names=JOINT_ORDER,
        target_rate_hz=args.rate,
    )

    keep_running = True

    def sig_handler(sig, frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    with launch_simulation(cfg, {"device": args.device, "visualizer": viz_choice}):
        env = RslRlVecEnvWrapper(cast("ManagerBasedRLEnv", gym.make(task_id, cfg=cfg)), clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy()
            base = cast("ManagerBasedRLEnv", env.unwrapped)

            robot = base.scene["robot"]
            # Map joints to exact indices in Articulation
            joint_indices = [robot.find_joints(name)[0][0] for name in JOINT_ORDER]

            obs = env.get_observations()
            episode_count = 0
            step_count = 0

            streamer.reset_timer()
            print("[INFO] Beginning 60 Hz real-time inference & hardware stream...")

            with torch.inference_mode():
                # Initial target sent to hardware
                joint_pos_t = robot.data.joint_pos.torch[0, joint_indices]
                joint_angles_rad = joint_pos_t.cpu().numpy().tolist()
                streamer.send_joint_targets(joint_angles_rad, is_reset=False)
                last_valid_angles = list(joint_angles_rad)

                while keep_running and episode_count < args.episodes:
                    step_start = time.perf_counter()

                    # Policy forward pass (consumes synthetic RGB from Isaac Sim camera)
                    action = policy(obs)

                    # Step environment
                    obs, rewards, dones, extras = env.step(action)
                    step_count += 1

                    # Extract current joint positions in radians
                    joint_pos_t = robot.data.joint_pos.torch[0, joint_indices]
                    joint_angles_rad = joint_pos_t.cpu().numpy().tolist()

                    is_reset = bool(dones[0].item())
                    if is_reset:
                        episode_count += 1
                        print(f"\n[RESET] Episode {episode_count} ended. Executing smooth vertical retract & reset...")

                        # 1. First retract UP vertically to clear desk completely before translating
                        retract_pose = list(last_valid_angles)
                        retract_pose[1] = min(last_valid_angles[1], -1.15)  # Lift shoulder up and back away from desk
                        retract_pose[2] = min(last_valid_angles[2], 1.05)   # Unfold elbow to raise forearm up
                        retract_pose[3] = 0.90                              # Level wrist so fingers tilt UP away from desk
                        retract_pose[5] = 1.20                              # Open gripper

                        lift_steps = 60  # 1.0 second
                        for s in range(1, lift_steps + 1):
                            frac = s / lift_steps
                            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                            q_interp = [(1.0 - alpha) * a + alpha * b for a, b in zip(last_valid_angles, retract_pose)]
                            streamer.send_joint_targets(q_interp, is_reset=False)
                            streamer.pace(1.0 / 60.0)

                        # 2. Smoothly move from high clearance pose to new episode start pose
                        reset_target = list(joint_angles_rad)
                        move_steps = 75  # 1.25 seconds
                        for s in range(1, move_steps + 1):
                            frac = s / move_steps
                            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                            q_interp = [(1.0 - alpha) * a + alpha * b for a, b in zip(retract_pose, reset_target)]
                            streamer.send_joint_targets(q_interp, is_reset=False)
                            streamer.pace(1.0 / 60.0)

                        # 3. Hold stationary for 0.3s to allow physical vibration to settle
                        for _ in range(18):
                            streamer.send_joint_targets(reset_target, is_reset=False)
                            streamer.pace(1.0 / 60.0)

                        last_valid_angles = list(reset_target)
                        streamer.reset_timer()
                        continue

                    # Normal step: transmit joint angles over UDP to hardware bridge
                    streamer.send_joint_targets(joint_angles_rad, is_reset=False)
                    last_valid_angles = list(joint_angles_rad)

                    # Enforce strict 60 Hz wall-clock pacing
                    streamer.pace(base.step_dt)

                    if step_count % 30 == 0:
                        sys.stdout.write(
                            f"\r[SIM-STREAM] Ep {episode_count + 1}/{args.episodes} | "
                            f"Step {step_count:,} | Sent: {streamer.sent_count:,} pkts   "
                        )
                        sys.stdout.flush()

        finally:
            print("\n[INFO] Closing hardware streamer and simulation...", flush=True)
            try:
                from isaac_vision_rl.utils.calibration import FollowerArmCalibration, CALIB_FILE_PATH
                from isaac_vision_rl.utils.home_pose import HOME_TICKS
                calib = FollowerArmCalibration(CALIB_FILE_PATH)
                rest_rad = calib.ticks_to_rad(HOME_TICKS).tolist()
                start_q = joint_angles_rad if 'joint_angles_rad' in locals() else rest_rad

                # Check if arm is away from rest stance
                dist = max(abs(a - b) for a, b in zip(start_q, rest_rad))
                if dist > 0.15:
                    print("[INFO] Executing clearance-safe retract and park...", flush=True)
                    # Stage 1: Retract up
                    wp_lift = list(start_q)
                    wp_lift[1] = min(start_q[1], -1.15)
                    wp_lift[2] = min(start_q[2], 1.05)
                    wp_lift[3] = 0.90
                    wp_lift[5] = 1.20
                    for s in range(1, 61):
                        frac = s / 60
                        alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                        interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(start_q, wp_lift)]
                        streamer.send_joint_targets(interp_q, is_reset=False)
                        streamer.pace(1.0 / 60.0)

                    # Stage 2: Slew above home cradle
                    wp_clear = list(rest_rad)
                    wp_clear[2] -= 0.30
                    wp_clear[3] -= 0.15
                    wp_clear[5] = 0.50
                    for s in range(1, 49):
                        frac = s / 48
                        alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                        interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(wp_lift, wp_clear)]
                        streamer.send_joint_targets(interp_q, is_reset=False)
                        streamer.pace(1.0 / 60.0)

                    # Stage 3: Fold into home resting blocks
                    for s in range(1, 49):
                        frac = s / 48
                        alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                        interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(wp_clear, rest_rad)]
                        streamer.send_joint_targets(interp_q, is_reset=False)
                        streamer.pace(1.0 / 60.0)
                else:
                    # Settle directly into rest stance
                    for s in range(1, 49):
                        frac = s / 48
                        alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
                        interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(start_q, rest_rad)]
                        streamer.send_joint_targets(interp_q, is_reset=False)
                        streamer.pace(1.0 / 60.0)
                time.sleep(0.3)
                print("[INFO] Robot safely returned to home rest stance.", flush=True)
            except Exception as e:
                print(f"[WARN] Error during parking: {e}")
            streamer.close()
            env.close()

    print("[SUCCESS] Sim-to-Real streaming session completed.")


if __name__ == "__main__":
    main()

