#!/usr/bin/env python3
"""Simulation-only camera-guided reference grasp. No cube pose is used for motion."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--viz", choices=["kit", "newton_gl"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs/perception/camera_lift.json")
    p.add_argument("--cube-offset", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"))
    p.add_argument("--cube-yaw", type=float, default=0.0, help="Test reset yaw in radians")
    p.add_argument(
        "--grasp-assist", action="store_true", help="Use camera-triggered closure instead of scripted closure"
    )
    p.add_argument("--record-video", action="store_true", help="Store global camera MP4 beside the result JSON")
    args = p.parse_args()
    if args.headless and args.viz:
        p.error("Choose --headless or --viz")
    os.environ["SO101_CURRICULUM"] = "fixed"
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from isaaclab.app import launch_simulation
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    from isaac_vision_rl.perception.cube import YellowCubeEstimator
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry
    from isaac_vision_rl.utils.pipeline import camera_fingerprint

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.scene.camera.width = 512
    cfg.scene.camera.height = 512
    cfg.scene.camera.data_types = ["rgb", "distance_to_image_plane"]
    initial_cube = cfg.scene.cube.init_state.pos
    cfg.scene.cube.init_state.pos = (
        initial_cube[0] + args.cube_offset[0],
        initial_cube[1] + args.cube_offset[1],
        initial_cube[2],
    )
    cfg.scene.cube.init_state.rot = tuple(Rotation.from_euler("z", args.cube_yaw).as_quat())
    if args.viz == "newton_gl":
        if cfg.sim.default_visualizer_cfg is None:
            raise ValueError("Newton camera streaming requires a visualizer configuration")
        cfg.sim.default_visualizer_cfg.streaming_view = True
        cfg.sim.default_visualizer_cfg.streaming_envs = [0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": False, "status": "running or interrupted"}) + "\n")
    video = None
    with launch_simulation(cfg, {"device": args.device, "visualizer": args.viz}):
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        try:
            base = cast("ManagerBasedRLEnv", env.unwrapped)
            env.reset()
            robot, camera = base.scene["robot"], base.scene["camera"]

            def tensor(value):
                return value.torch if hasattr(value, "torch") else value

            def matrix(pos, quat):
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(tensor(quat).cpu().numpy()).as_matrix()
                T[:3, 3] = tensor(pos).cpu().numpy()
                return T

            T_world_base = matrix(robot.data.root_pos_w[0], robot.data.root_quat_w[0])
            for _ in range(4):
                base.sim.render()
                camera.update(cfg.sim.dt, force_recompute=True)
            T_world_camera = matrix(camera.data.pos_w[0], camera.data.quat_w_ros[0])
            detector = YellowCubeEstimator(np.linalg.inv(T_world_base) @ T_world_camera)
            spawner = cfg.scene.robot.spawn
            if not isinstance(spawner, UsdFileCfg):
                raise TypeError("Expected USD robot")
            geometry = GraspGeometry(spawner.usd_path, cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot)
            ids = [robot.find_joints(n)[0][0] for n in ARM_NAMES + ["gripper"]]
            q = tensor(robot.data.joint_pos).clone()
            command = q[0, ids].cpu().numpy().copy()
            observations = []
            frame_count = 0
            if args.record_video:
                import cv2

                video = cv2.VideoWriter(
                    str(output.with_suffix(".mp4")), cv2.VideoWriter.fourcc(*"mp4v"), 1 / base.step_dt, (512, 512)
                )
                if not video.isOpened():
                    raise RuntimeError("Cannot open reference video encoder")

            def observe(stage):
                for _ in range(4):
                    base.sim.render()
                    camera.update(cfg.sim.dt, force_recompute=True)
                rgb = tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype("uint8")
                depth = tensor(camera.data.output["distance_to_image_plane"])[0].cpu().numpy().squeeze()
                K = tensor(camera.data.intrinsic_matrices)[0].cpu().numpy()
                estimate = detector.estimate_depth(rgb, depth, K)
                np.savez_compressed(
                    output.parent / f"camera_lift_{stage}.npz",
                    rgb=rgb,
                    depth_z_m=depth,
                    K=K,
                    T_base_camera=np.linalg.inv(T_world_base) @ T_world_camera,
                )
                observations.append({"stage": stage, **estimate.to_dict()})
                Image.fromarray(rgb).save(output.parent / f"camera_lift_{stage}.png")
                print(stage, estimate.to_dict(), flush=True)
                if not estimate.valid:
                    raise RuntimeError(f"Camera cannot locate cube at {stage}: {estimate.reason}")
                return T_world_base[:3, :3] @ np.array(estimate.center_base_m) + T_world_base[:3, 3]

            def move(target, seconds):
                nonlocal command, frame_count
                for _ in range(int(seconds / cfg.sim.dt)):
                    speed = np.array([1.5] * 5 + [0.8]) * cfg.sim.dt
                    command += np.clip(np.asarray(target) - command, -speed, speed)
                    if frame_count % cfg.decimation == 0:
                        base.sim.render()
                        if args.grasp_assist:
                            cast(Any, base)._camera_cube_cache = None
                            term = base.action_manager.get_term("gripper_action")
                            term.process_actions(torch.zeros(1, 1, device=args.device))
                    if args.grasp_assist:
                        command[-1] = float(base.action_manager.get_term("gripper_action").processed_actions[0, 0])
                    q[:, ids] = torch.tensor(command, device=args.device, dtype=torch.float32)
                    robot.set_joint_position_target(q)
                    base.scene.write_data_to_sim()
                    base.sim.step()
                    base.scene.update(cfg.sim.dt)
                    frame_count += 1
                    if video is not None and frame_count % cfg.decimation == 0:
                        camera.update(cfg.sim.dt, force_recompute=True)
                        rgb = tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype("uint8")
                        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        actual = float(tensor(robot.data.joint_pos)[0, ids[-1]])
                        cube_z = float(tensor(base.scene["cube"].data.root_pos_w)[0, 2])
                        cv2.putText(
                            frame,
                            f"gripper target {command[-1]:.2f} actual {actual:.2f} cube z {cube_z:.3f} m",
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (255, 255, 255),
                            1,
                        )
                        video.write(frame)
                    if not torch.isfinite(tensor(robot.data.joint_pos)).all():
                        raise FloatingPointError("Invalid robot dynamics")

            center = observe("initial")
            # Raise near the base before traversing toward the observed cube.
            # Targets use robot kinematics and camera estimates only.
            start = geometry.forward(command[:5])[:3, 3]
            previous = command[:5].copy()
            for height in np.linspace(start[2], start[2] + 0.15, 8)[1:]:
                previous = geometry.solve([start[0], start[1], height], previous, downward=False)
                move(np.r_[previous, 1.2], 0.3)
            center = observe("clearance")
            above = geometry.solve(center + [0, 0, 0.04], command[:5])
            move(np.r_[above, 1.2], 3.0)
            center = observe("above")
            grasp = geometry.solve(center, command[:5])
            move(np.r_[grasp, 1.2], 2.0)
            center = observe("grasp_open")
            # Reacquire after descent, correcting small object movements from contact.
            grasp = geometry.solve(center, command[:5])
            move(np.r_[grasp, 1.2], 0.5)
            move(np.r_[grasp, 0.0], 1.5)
            # Leave clearance for actuator tracking error under cube load.
            raised = geometry.solve(center + [0, 0, 0.08], command[:5])
            move(np.r_[raised, 0.0], 2.0)
            heights = []
            distances = []
            contacts = []
            from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact

            body = robot.find_bodies("gripper_frame_link")[0][0]
            for _ in range(100):
                move(np.r_[raised, 0.0], cfg.sim.dt)
                # Ground truth is used only for success scoring, never target generation.
                cube = tensor(base.scene["cube"].data.root_pos_w)[0]
                heights.append(float(cube[2]))
                distances.append(float(torch.norm(cube - tensor(robot.data.body_pos_w)[0, body])))
                contacts.append(bool(bilateral_cube_contact(base)[0]))
            passed = min(heights[-50:]) > 0.47 and max(distances[-50:]) < 0.06 and all(contacts[-50:])
            report = {
                "camera_fingerprint": camera_fingerprint(),
                "passed": passed,
                "controller": "camera-guided IK reference, not a learned PPO policy",
                "cube_ground_truth_used_for_motion": False,
                "scoring_ground_truth": True,
                "minimum_hold_height_m": min(heights[-50:]),
                "max_hold_distance_m": max(distances[-50:]),
                "bilateral_contact_fraction": sum(contacts[-50:]) / 50,
                "cube_offset_m": args.cube_offset,
                "cube_yaw_rad": args.cube_yaw,
                "grasp_assist_used": args.grasp_assist,
                "grasp_assist_latched": bool(
                    cast(Any, base.action_manager.get_term("gripper_action")).grasp_latched[0]
                ),
                "camera_observations": observations,
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(report, flush=True)
            return 0 if passed else 2
        finally:
            if video is not None:
                video.release()
            env.close()


if __name__ == "__main__":
    sys.exit(main())
