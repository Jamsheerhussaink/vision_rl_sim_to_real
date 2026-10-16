#!/usr/bin/env python3
"""Interactive Sim-to-Real Motion Mirroring Demo.

Executes a smooth, deterministic reach-grasp-lift trajectory in Isaac Sim
and streams it at 60 Hz to the physical robot over UDP so you can watch
the virtual arm in Newton Viewer and the physical arm on your desk move together.
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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from isaac_vision_rl.utils.calibration import JOINT_ORDER, FollowerArmCalibration
from isaac_vision_rl.utils.hardware_client import HardwareStreamer


def interpolate(start_q: list[float], end_q: list[float], fraction: float) -> list[float]:
    """Smooth cosine interpolation between two joint poses."""
    alpha = 0.5 * (1.0 - math.cos(fraction * math.pi))
    return [(1.0 - alpha) * s + alpha * e for s, e in zip(start_q, end_q)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viz", choices=["newton", "kit", "none"], default="newton", help="Visualizer backend")
    parser.add_argument("--device", default="cuda:0", help="Simulation device")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Hardware bridge host")
    parser.add_argument("--udp-port", type=int, default=9870, help="Hardware bridge port")
    parser.add_argument("--cycles", type=int, default=2, help="Number of demo cycles to execute")
    parser.add_argument("--sim-only", action="store_true", help="Run live simulation without opening a hardware UDP client")
    parser.add_argument("--report", default="outputs/hardware_mapping/home_clearance_check.json")
    args = parser.parse_args()

    os.environ["SO101_POLICY"] = "rgb"
    os.environ["SO101_CURRICULUM"] = "fixed"
    os.environ["SO101_CUBE_JITTER"] = "0.0"
    os.environ["SO101_BOX_JITTER"] = "0.0"

    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry
    from isaac_vision_rl.utils.table_clearance import HandTableClearance
    from isaac_vision_rl.utils.evaluation_mirror import checked_ticks

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device

    # Ensure virtual gripper has snappy response in Newton Viewer
    if "gripper" in cfg.scene.robot.actuators:
        cfg.scene.robot.actuators["gripper"].stiffness = 120.0
        cfg.scene.robot.actuators["gripper"].damping = 2.0
        cfg.scene.robot.actuators["gripper"].effort_limit = 2.0

    print("\n" + "=" * 70)
    print("SO-101 Sim-to-Real Digital Twin Motion Mirroring Demo")
    print(f"Cycles:     {args.cycles}")
    print(f"Visualizer: {args.viz}")
    print(f"Hardware:   UDP {args.udp_host}:{args.udp_port} @ 60 Hz")
    print("=" * 70 + "\n")

    streamer = None if args.sim_only else HardwareStreamer(
        host=args.udp_host,
        port=args.udp_port,
        joint_names=JOINT_ORDER,
        target_rate_hz=60.0,
    )
    calib = FollowerArmCalibration()

    keep_running = True

    def sig_handler(sig, frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    viz_choice = None if args.viz == "none" else args.viz

    with launch_simulation(cfg, {"device": args.device, "visualizer": viz_choice}):
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        try:
            env.reset()
            base = env.unwrapped
            robot = base.scene["robot"]

            joint_indices = [robot.find_joints(name)[0][0] for name in JOINT_ORDER]

            # Solve inverse kinematics for key waypoints
            spawner = cfg.scene.robot.spawn
            geometry = GraspGeometry(spawner.usd_path, cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot)
            hardware_lower, hardware_upper = calib.get_joint_limits_rad()
            geometry.lower = np.maximum(geometry.lower, hardware_lower[:5] + 0.01)
            geometry.upper = np.minimum(geometry.upper, hardware_upper[:5] - 0.01)
            cube_pos = np.array(cfg.scene.cube.init_state.pos)

            home_arm = [cfg.scene.robot.init_state.joint_pos[n] for n in ARM_NAMES]
            above_arm = geometry.solve(cube_pos + [0, 0, 0.05], home_arm).tolist()
            # The tool origin is not the fingertip: retain table clearance at grasp.
            grasp_arm = geometry.solve(cube_pos + [0, 0, 0.015], above_arm).tolist()
            lift_arm = geometry.solve(cube_pos + [0, 0, 0.08], grasp_arm).tolist()

            # Construct full 6-DoF poses: [5 arm joints, gripper]
            # gripper: 1.2 rad (fully open jaws), 0.0 rad (fully closed / pinch)
            POSE_HOME = home_arm + [cfg.scene.robot.init_state.joint_pos["gripper"]]
            POSE_CLEAR = list(POSE_HOME)
            POSE_CLEAR[2] -= 0.30  # Unfold elbow to raise fingers before reaching.
            POSE_CLEAR[3] -= 0.15
            POSE_ABOVE = above_arm + [1.2]
            POSE_GRASP_OPEN = grasp_arm + [1.2]
            POSE_GRASP_CLOSED = grasp_arm + [0.0]
            POSE_LIFTED = lift_arm + [0.0]
            phases = [
                (POSE_HOME, POSE_CLEAR, 2.0, "Raise clear of home"),
                (POSE_CLEAR, POSE_ABOVE, 4.0, "Approach above cube"),
                (POSE_ABOVE, POSE_GRASP_OPEN, 2.0, "Descend with fingertip clearance"),
                (POSE_GRASP_OPEN, POSE_GRASP_CLOSED, 2.0, "Close gripper"),
                (POSE_GRASP_CLOSED, POSE_LIFTED, 2.5, "Lift"),
                (POSE_LIFTED, POSE_GRASP_CLOSED, 2.0, "Lower cube"),
                (POSE_GRASP_CLOSED, POSE_GRASP_OPEN, 2.0, "Release"),
                (POSE_GRASP_OPEN, POSE_ABOVE, 2.0, "Retract vertically"),
                (POSE_ABOVE, POSE_CLEAR, 4.0, "Return above home"),
                (POSE_CLEAR, POSE_HOME, 2.0, "Fold into measured home"),
            ]
            clearance = HandTableClearance(spawner.usd_path, cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot)
            planned = []
            for start, end, duration, label in phases:
                # Check every reference that will actually be streamed, not just endpoints.
                minimum = float("inf")
                for step in range(int(duration / base.step_dt) + 1):
                    target = interpolate(start, end, step / int(duration / base.step_dt))
                    checked_ticks(calib, target)
                    minimum = min(minimum, clearance.minimum(target))
                if minimum < 0.005:
                    raise RuntimeError(f"Unsafe planned phase {label}: clearance {minimum:.4f} m")
                planned.append({"phase": label, "minimum_commanded_clearance_m": minimum})
            hand_ids = {name: robot.find_bodies(name)[0][0] for name in clearance.points}
            measured = []
            report = Path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"passed": False, "status": "running", "planned": planned}, indent=2))

            def execute_trajectory(start_pose: list[float], end_pose: list[float], duration_s: float, description: str):
                if not keep_running:
                    return
                ticks_start = calib.rad_to_ticks(start_pose)[5]
                ticks_end = calib.rad_to_ticks(end_pose)[5]
                print(f"  -> {description} ({duration_s:.1f}s) [Grip: {start_pose[5]:.2f} rad ({ticks_start}t) -> {end_pose[5]:.2f} rad ({ticks_end}t)]...")

                steps = max(1, int(duration_s / base.step_dt))
                for step_idx in range(steps):
                    if not keep_running:
                        break
                    frac = (step_idx + 1) / steps
                    target_q = interpolate(start_pose, end_pose, frac)

                    # Apply command to simulated robot
                    q_tensor = robot.data.joint_pos.torch.clone()
                    q_tensor[0, joint_indices] = torch.tensor(target_q, device=args.device, dtype=torch.float32)
                    robot.set_joint_position_target(q_tensor)
                    for physics_step in range(cfg.decimation):
                        base.scene.write_data_to_sim()
                        base.sim.step(render=physics_step == cfg.decimation - 1)
                        base.scene.update(cfg.sim.dt)
                        positions = {name: robot.data.body_pos_w.torch[0, i].cpu().numpy() for name, i in hand_ids.items()}
                        rotations = {name: robot.data.body_quat_w.torch[0, i].cpu().numpy() for name, i in hand_ids.items()}
                        gap = clearance.measured_minimum(positions, rotations)
                        measured.append({"phase": description, "clearance_m": gap})
                        if gap < 0.003:
                            report.write_text(json.dumps({"passed": False, "planned": planned, "measured": measured}, indent=2))
                            raise RuntimeError(f"Actual hand/table clearance below 3 mm: {gap:.4f} m; stopping")

                    # Stream desired trajectory target_q directly to physical robot
                    if streamer is not None:
                        streamer.send_joint_targets(target_q, is_reset=False)
                        streamer.pace(base.step_dt)
                    else:
                        time.sleep(base.step_dt)

            print("[INFO] Beginning mirroring demonstration across 60 Hz bridge...\n")
            current_pose = POSE_HOME

            for cycle in range(args.cycles):
                if not keep_running:
                    break
                print(f"[CYCLE {cycle + 1}/{args.cycles}]")

                for start, end, duration, label in phases:
                    execute_trajectory(start, end, duration, label)
                current_pose = POSE_HOME
                print()
            report.write_text(json.dumps({"passed": bool(measured) and keep_running, "sim_only": args.sim_only,
                "planned": planned, "minimum_measured_clearance_m": min((r["clearance_m"] for r in measured), default=None),
                "measured": measured}, indent=2))

        finally:
            if streamer is not None:
                print("[INFO] Sending explicit STOP packet to hardware bridge...")
                for _ in range(5):
                    streamer.send_stop()
                    time.sleep(0.02)
                streamer.close()
            env.close()
            print("[INFO] Simulation environment cleanly closed.")


if __name__ == "__main__":
    main()

