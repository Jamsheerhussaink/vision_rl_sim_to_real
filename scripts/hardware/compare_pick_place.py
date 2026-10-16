#!/usr/bin/env python3
"""Sim-to-Real Full Pick, Lift & Place with Synchronized Video Comparison.

Executes a complete 10-phase Pick, Lift, Transport & Place trajectory in Isaac Sim,
simultaneously streams joint targets to the physical SO-101 robot arm over UDP,
records synchronized video feeds from both the RealSense D455 camera and the
simulation virtual camera, and generates a side-by-side comparison video with a
difference heatmap and SSIM / PSNR visual similarity metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from isaac_vision_rl.utils.calibration import CALIB_FILE_PATH, JOINT_ORDER, FollowerArmCalibration
from isaac_vision_rl.utils.hardware_client import HardwareStreamer


def compute_metrics(real_bgr: np.ndarray, sim_bgr: np.ndarray) -> dict[str, float]:
    """Compute PSNR, MSE, MAE, and SSIM between real and simulated BGR frames."""
    diff = np.abs(real_bgr.astype(np.float32) - sim_bgr.astype(np.float32))
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))
    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse)) if mse > 1e-6 else 99.0

    gray1 = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim = float(np.mean(ssim_map))

    return {"ssim": ssim, "psnr": psnr, "mae": mae, "mse": mse}


def create_comparison_tile(
    real_bgr: np.ndarray, sim_bgr: np.ndarray, metrics: dict[str, float], phase_name: str
) -> np.ndarray:
    """Create a 3-panel tile: [ Real Camera | Simulation Camera | Difference Heatmap ]."""
    h, w = real_bgr.shape[:2]

    diff = cv2.absdiff(real_bgr, sim_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    def add_label(img: np.ndarray, text: str, subtext: str = ""):
        canvas = img.copy()
        banner_h = max(32, int(h * 0.065))
        cv2.rectangle(canvas, (0, 0), (w, banner_h), (0, 0, 0), -1)
        scale1 = 0.45 if h <= 240 else 0.8
        scale2 = 0.35 if h <= 240 else 0.6
        y1 = int(banner_h * 0.45)
        y2 = int(banner_h * 0.85)
        cv2.putText(canvas, text, (int(w * 0.02), y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, (255, 255, 255), 1 if h <= 240 else 2, cv2.LINE_AA)
        if subtext:
            cv2.putText(canvas, subtext, (int(w * 0.02), y2), cv2.FONT_HERSHEY_SIMPLEX, scale2, (180, 255, 180), 1, cv2.LINE_AA)
        return canvas

    p1 = add_label(real_bgr, "RealSense D455 (Real)", phase_name)
    p2 = add_label(sim_bgr, "Isaac Sim (Virtual)", phase_name)
    p3 = add_label(
        heatmap,
        f"Diff Heatmap | SSIM: {metrics['ssim']:.3f}",
        f"PSNR: {metrics['psnr']:.1f} dB | MAE: {metrics['mae']:.1f}",
    )

    return np.hstack([p1, p2, p3])


def interpolate(start_q: list[float], end_q: list[float], fraction: float) -> list[float]:
    """Smooth cosine S-curve interpolation between two joint poses."""
    alpha = 0.5 * (1.0 - math.cos(fraction * math.pi))
    return [(1.0 - alpha) * s + alpha * e for s, e in zip(start_q, end_q)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=1, help="Number of full pick-and-place cycles to execute")
    parser.add_argument("--output-dir", default="outputs/comparison", help="Output directory for recordings and metrics")
    parser.add_argument("--device", default="cuda:0", help="Isaac Sim device")
    parser.add_argument("--viz", choices=["newton", "kit", "none"], default="newton", help="Visualizer backend")
    parser.add_argument("--headless", action="store_true", help="Run without visualizer window")
    parser.add_argument("--stream-to-robot", action="store_true", help="Stream UDP joint targets to hardware bridge")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Target hardware bridge UDP host")
    parser.add_argument("--udp-port", type=int, default=9870, help="Target hardware bridge UDP port")
    parser.add_argument("--no-camera", action="store_true", help="Run simulation-only recording without real camera")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate FPS (default: 30)")
    args = parser.parse_args()

    os.environ["SO101_POLICY"] = "rgb"
    os.environ["SO101_CURRICULUM"] = "fixed"
    os.environ["SO101_CUBE_JITTER"] = "0.0"
    os.environ["SO101_BOX_JITTER"] = "0.0"
    os.environ["SO101_RGB_FPS"] = str(args.fps)
    os.environ["SO101_CAMERA_WIDTH"] = str(args.width)
    os.environ["SO101_CAMERA_HEIGHT"] = str(args.height)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("SO-101 Full Pick, Lift & Place Sim-to-Real Video Comparison")
    print(f"Cycles:         {args.cycles}")
    print(f"Vision Profile: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Hardware:       {'Streaming to UDP ' + args.udp_host + ':' + str(args.udp_port) if args.stream_to_robot else 'Simulation only'}")
    print(f"Output Dir:     {out_dir}")
    print("=" * 75 + "\n")

    # 1. Initialize RealSense Camera
    pipe = None
    has_real_camera = False

    if not args.no_camera:
        try:
            import pyrealsense2 as rs

            print(f"[INFO] Connecting to Intel RealSense D455 Camera ({args.width}x{args.height} @ {args.fps} FPS)...")
            pipe = rs.pipeline()
            cfg_rs = rs.config()
            cfg_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            pipe.start(cfg_rs)
            has_real_camera = True
            print("[SUCCESS] RealSense D455 streaming actively.")
        except Exception as e:
            print(f"[WARNING] RealSense camera not available: {e}")
            print("          Continuing in simulation-only recording mode...")
            pipe = None
            has_real_camera = False

    # 2. Launch Isaac Sim Environment
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry

    cfg_sim = PickplaceEnvCfg()
    cfg_sim.scene.num_envs = 1
    cfg_sim.sim.device = args.device
    cfg_sim.scene.camera.width = args.width
    cfg_sim.scene.camera.height = args.height

    # Ensure robust gripper pinch force for picking the cube
    if "gripper" in cfg_sim.scene.robot.actuators:
        cfg_sim.scene.robot.actuators["gripper"].stiffness = 120.0
        cfg_sim.scene.robot.actuators["gripper"].damping = 2.0
        cfg_sim.scene.robot.actuators["gripper"].effort_limit = 2.0

    streamer = None
    if args.stream_to_robot:
        streamer = HardwareStreamer(
            host=args.udp_host,
            port=args.udp_port,
            joint_names=JOINT_ORDER,
            target_rate_hz=60.0,
        )

    viz_choice = None if args.headless or args.viz == "none" else args.viz

    keep_running = True

    def sig_handler(sig, frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("[INFO] Launching Isaac Sim Pick-and-Place scene...")
    with launch_simulation(cfg_sim, {"device": args.device, "visualizer": viz_choice}):
        env = gym.make("IsaacVisionRl-Pickplace-SO101", cfg=cfg_sim)
        try:
            env.reset()
            base = env.unwrapped
            robot = base.scene["robot"]
            cube = base.scene["cube"]
            sim_camera = base.scene["camera"]

            joint_indices = [robot.find_joints(name)[0][0] for name in JOINT_ORDER]

            # Solve Kinematics for Pick, Lift, Transport & Place Waypoints
            spawner = cfg_sim.scene.robot.spawn
            geometry = GraspGeometry(spawner.usd_path, cfg_sim.scene.robot.init_state.pos, cfg_sim.scene.robot.init_state.rot)
            cube_pos = np.array(cfg_sim.scene.cube.init_state.pos)
            box_pos = np.array(cfg_sim.scene.box.init_state.pos)

            home_arm = [cfg_sim.scene.robot.init_state.joint_pos[n] for n in ARM_NAMES]
            home_gripper = cfg_sim.scene.robot.init_state.joint_pos["gripper"]
            above_cube = geometry.solve(cube_pos + [0, 0, 0.07], home_arm).tolist()
            grasp_cube = geometry.solve(cube_pos + [0, 0, 0.015], above_cube).tolist()

            # Coordinated vertical lift kinematics: solves all joints (shoulder, elbow, wrist)
            # elevates vertically 10.0 cm above grasp pose (~14.1 cm above desk, >7 cm above box rim)
            # while folding the elbow inwards to minimize gravitational moment arm on shoulder_lift.
            lift_arm = geometry.solve(cube_pos + [0, 0, 0.12], grasp_cube).tolist()

            # Box placement waypoints:
            # Elevated elbow kinematics (shoulder_lift <= -0.05) ensuring hand & cube stay
            # >18 cm above table (>9 cm above box rim) during transit, then lowers into cavity.
            above_box = [-0.3812, -0.0500, -0.2141, 1.2320, 0.0054]
            place_box = [-0.3797, -0.0500, 0.7165, 0.2195, 0.0201]

            # Poses: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
            # POSE_HOME is the exact calibrated photo home pose
            POSE_HOME = home_arm + [home_gripper]
            POSE_CLEAR = list(POSE_HOME)
            POSE_CLEAR[2] -= 0.30  # Unfold elbow to raise hand clear of desk
            POSE_CLEAR[3] -= 0.15
            POSE_CLEAR_OPEN = list(POSE_CLEAR)
            POSE_CLEAR_OPEN[5] = 1.20

            GRIP_OPEN = 1.20
            # 0.20 rad (~2281 ticks) firmly pinches the 45 mm physical cube without motor stall
            GRIP_PINCH = 0.20

            POSE_ABOVE_CUBE = above_cube + [GRIP_OPEN]
            POSE_GRASP_OPEN = grasp_cube + [GRIP_OPEN]
            POSE_GRASP_CLOSED = grasp_cube + [GRIP_PINCH]
            POSE_LIFTED = lift_arm + [GRIP_PINCH]

            POSE_ABOVE_BOX = above_box + [GRIP_PINCH]
            POSE_PLACE_BOX = place_box + [GRIP_PINCH]
            POSE_RELEASE_BOX = place_box + [GRIP_OPEN]
            POSE_RETRACT_BOX = above_box + [GRIP_OPEN]

            # Trajectory phase segments: (start_pose, end_pose, duration_s, description)
            trajectory_segments = [
                (POSE_HOME, POSE_CLEAR, 1.5, "Raise clear of home"),
                (POSE_CLEAR, POSE_CLEAR_OPEN, 0.8, "Open gripper jaws"),
                (POSE_CLEAR_OPEN, POSE_ABOVE_CUBE, 2.5, "Approach above cube"),
                (POSE_ABOVE_CUBE, POSE_GRASP_OPEN, 1.8, "Descend around cube"),
                (POSE_GRASP_OPEN, POSE_GRASP_CLOSED, 1.5, "Close gripper on cube"),
                (POSE_GRASP_CLOSED, POSE_LIFTED, 2.2, "Lift cube vertically high"),
                (POSE_LIFTED, POSE_ABOVE_BOX, 3.0, "Transport high across to box"),
                (POSE_ABOVE_BOX, POSE_PLACE_BOX, 1.8, "Lower cube into box"),
                (POSE_PLACE_BOX, POSE_RELEASE_BOX, 1.2, "Open gripper to release"),
                (POSE_RELEASE_BOX, POSE_RETRACT_BOX, 1.8, "Retract arm out of box"),
                (POSE_RETRACT_BOX, POSE_CLEAR_OPEN, 3.0, "Return to high clearance"),
                (POSE_CLEAR_OPEN, POSE_CLEAR, 0.8, "Close gripper"),
                (POSE_CLEAR, POSE_HOME, 1.8, "Fold into home stance"),
            ]

            total_cycle_time = sum(dur for _, _, dur, _ in trajectory_segments)
            total_time = total_cycle_time * args.cycles
            fps = float(args.fps)
            print(f"[INFO] Trajectory: 13 phases ({total_cycle_time:.1f}s per cycle, {total_time:.1f}s total at {fps:.0f} FPS).")

            # Warm-up sensors
            print("[INFO] Warming up camera sensors...")
            for _ in range(20):
                if pipe is not None:
                    pipe.wait_for_frames(2000)
                base.sim.render()
                sim_camera.update(cfg_sim.sim.dt, force_recompute=True)

            # Initialize video writers
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            sim_video_path = out_dir / "sim_footage.mp4"
            real_video_path = out_dir / "real_footage.mp4"
            comparison_video_path = out_dir / "sim_vs_real_comparison.mp4"

            video_sim = cv2.VideoWriter(str(sim_video_path), fourcc, fps, (args.width, args.height))
            video_real = cv2.VideoWriter(str(real_video_path), fourcc, fps, (args.width, args.height)) if has_real_camera else None
            video_comparison = (
                cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (args.width * 3, args.height)) if has_real_camera else None
            )

            all_metrics = []
            frame_count = 0
            start_wall = time.time()

            print("\n[INFO] Starting execution, streaming & synchronized video recording...")

            for cycle in range(args.cycles):
                if not keep_running:
                    break
                print(f"\n--- CYCLE {cycle + 1}/{args.cycles} ---")

                for p_start, p_end, duration_s, phase_name in trajectory_segments:
                    if not keep_running:
                        break

                    steps = max(1, int(duration_s * fps))
                    sys.stdout.write(f"\r  Phase: {phase_name:<30} ({duration_s:.1f}s)...")
                    sys.stdout.flush()

                    for step_idx in range(steps):
                        if not keep_running:
                            break

                        frac = (step_idx + 1) / steps
                        target_q = interpolate(p_start, p_end, frac)

                        # 1. Apply command to virtual robot
                        q_tensor = robot.data.joint_pos.torch.clone()
                        q_tensor[0, joint_indices] = torch.tensor(target_q, device=args.device, dtype=torch.float32)
                        robot.set_joint_position_target(q_tensor)

                        # Step physics & render virtual camera
                        for physics_step in range(cfg_sim.decimation):
                            base.scene.write_data_to_sim()
                            base.sim.step(render=physics_step == cfg_sim.decimation - 1)
                            base.scene.update(cfg_sim.sim.dt)

                        sim_camera.update(cfg_sim.sim.dt, force_recompute=True)
                        sim_rgb_out = sim_camera.data.output["rgb"]
                        sim_rgb = sim_rgb_out.torch[0, :, :, :3] if hasattr(sim_rgb_out, "torch") else sim_rgb_out[0, :, :, :3]
                        sim_bgr = cv2.cvtColor(sim_rgb.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

                        # 2. Fetch real camera frame
                        real_bgr = None
                        if pipe is not None:
                            fs = pipe.wait_for_frames(2000)
                            real_bgr = np.asanyarray(fs.get_color_frame().get_data())

                        # 3. Stream joint command to physical robot over UDP
                        if streamer is not None:
                            streamer.send_joint_targets(target_q, is_reset=False)
                            streamer.pace(base.step_dt)

                        # 4. Write video frames
                        video_sim.write(sim_bgr)

                        if real_bgr is not None:
                            video_real.write(real_bgr)
                            metrics = compute_metrics(real_bgr, sim_bgr)
                            all_metrics.append(metrics)
                            tile = create_comparison_tile(real_bgr, sim_bgr, metrics, phase_name)
                            video_comparison.write(tile)

                        frame_count += 1

                # Settle firmly in calibrated home position
                if streamer is not None and keep_running:
                    sys.stdout.write("\n  Holding arm stationary in calibrated home position (1.5s)...")
                    sys.stdout.flush()
                    for _ in range(int(1.5 * fps)):
                        streamer.send_joint_targets(POSE_HOME, is_reset=False)
                        streamer.pace(base.step_dt)

            print("\n\n[INFO] Finalizing video recordings and metrics...")
            video_sim.release()
            if video_real is not None:
                video_real.release()
            if video_comparison is not None:
                video_comparison.release()

            # Generate Similarity Report
            if all_metrics:
                summary = {
                    "total_frames": len(all_metrics),
                    "duration_s": len(all_metrics) / fps,
                    "average_ssim": float(np.mean([x["ssim"] for x in all_metrics])),
                    "average_psnr_db": float(np.mean([x["psnr"] for x in all_metrics])),
                    "average_mae": float(np.mean([x["mae"] for x in all_metrics])),
                    "comparison_video": str(comparison_video_path),
                    "real_video": str(real_video_path),
                    "sim_video": str(sim_video_path),
                }
                report_file = out_dir / "similarity_report.json"
                report_file.write_text(json.dumps(summary, indent=2) + "\n")

                print("\n" + "=" * 70)
                print("SIM-TO-REAL VISUAL SIMILARITY REPORT")
                print("=" * 70)
                print(f"Recorded frames:    {summary['total_frames']} ({summary['duration_s']:.1f}s)")
                print(f"Average SSIM:       {summary['average_ssim']:.4f}  (Structural Similarity 0..1)")
                print(f"Average PSNR:       {summary['average_psnr_db']:.2f} dB (Peak Signal-to-Noise)")
                print(f"Average Pixel MAE:  {summary['average_mae']:.2f} / 255")
                print(f"Comparison Video:   {summary['comparison_video']}")
                print(f"Real Footage:       {summary['real_video']}")
                print(f"Sim Footage:        {summary['sim_video']}")
                print("=" * 70 + "\n")
            else:
                print(f"[SUCCESS] Virtual simulation video recorded: {sim_video_path}")

        finally:
            if pipe is not None:
                pipe.stop()
            if streamer is not None:
                streamer.close(send_stop=False)
            env.close()


if __name__ == "__main__":
    main()
