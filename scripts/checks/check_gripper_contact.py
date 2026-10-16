#!/usr/bin/env python3
"""Diagnostic tool to inspect SO-101 gripper CAD geometry and live physics contact."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def inspect_cad_profile(usd_path: str):
    """Inspect the CAD mesh geometry of the fingers without launching the simulator."""
    from pxr import Usd, UsdGeom

    print(f"Loading robot asset: {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Failed to open USD: {usd_path}")

    cache = UsdGeom.XformCache()

    # 1. Fixed Finger: wrist_roll_follower_so101_v1
    p_fixed = next(x for x in stage.Traverse() if x.GetName() == "wrist_roll_follower_so101_v1" and x.GetTypeName() == "Mesh")
    p_fixed_body = next(x for x in stage.Traverse() if x.GetName() == "gripper_link")
    m_fixed = np.array(cache.GetLocalToWorldTransform(p_fixed) * cache.GetLocalToWorldTransform(p_fixed_body).GetInverse()).T
    pts_fixed = np.array(UsdGeom.Mesh(p_fixed).GetPointsAttr().Get())
    pts_fixed_link = (np.c_[pts_fixed, np.ones(len(pts_fixed))] @ m_fixed.T)[:, :3]

    print("\n" + "=" * 60)
    print("FIXED FINGER CAD PROFILE (wrist_roll_follower_so101_v1 in gripper_link frame)")
    print("=" * 60)
    tip_fixed = pts_fixed_link[(pts_fixed_link[:, 2] < -0.065) & (pts_fixed_link[:, 2] > -0.104)]
    print(f"Total fingertip vertices: {len(tip_fixed)}")
    print(f"{'Height Z (m)':<15} | {'Inner Contact Surface X (mm)':<30} | {'Taper / Step-back (mm)':<22}")
    print("-" * 75)
    tip_x = tip_fixed[:, 0].max() * 1000.0  # reference at tip
    for z_cut in np.linspace(-0.100, -0.070, 7):
        pts_z = tip_fixed[(tip_fixed[:, 2] > z_cut - 0.003) & (tip_fixed[:, 2] < z_cut + 0.003)]
        if len(pts_z) > 0:
            max_x = pts_z[:, 0].max() * 1000.0
            step_back = tip_x - max_x
            note = "(Protruding Tip - Contact Area)" if abs(step_back) < 0.1 else f"({step_back:+.1f} mm gap from cube)"
            print(f"{z_cut:<15.3f} | {max_x:<30.2f} | {note}")

    # 2. Moving Jaw: moving_jaw_so101_v1
    p_mov = next(x for x in stage.Traverse() if x.GetName() == "moving_jaw_so101_v1" and x.GetTypeName() == "Mesh")
    p_mov_body = next(x for x in stage.Traverse() if x.GetName() == "moving_jaw_so101_v1_link")
    m_mov = np.array(cache.GetLocalToWorldTransform(p_mov) * cache.GetLocalToWorldTransform(p_mov_body).GetInverse()).T
    pts_mov = np.array(UsdGeom.Mesh(p_mov).GetPointsAttr().Get())
    pts_mov_link = (np.c_[pts_mov, np.ones(len(pts_mov))] @ m_mov.T)[:, :3]

    print("\n" + "=" * 60)
    print("MOVING JAW CAD PROFILE (moving_jaw_so101_v1 in moving_jaw_link frame)")
    print("=" * 60)
    tip_mov = pts_mov_link[pts_mov_link[:, 1] < -0.035]
    print(f"Total moving fingertip vertices: {len(tip_mov)}")
    print(f"Moving fingertip Y extent (length): [{tip_mov[:, 1].min():.4f}, {tip_mov[:, 1].max():.4f}] m")
    print(f"Moving fingertip X extent (thickness): [{tip_mov[:, 0].min():.4f}, {tip_mov[:, 0].max():.4f}] m")
    print(f"Moving fingertip Z extent (height): [{tip_mov[:, 2].min():.4f}, {tip_mov[:, 2].max():.4f}] m")
    print("=" * 60 + "\n")


def inspect_sim_contact(device: str = "cuda:0", viz: str | None = None, output_image: str = "outputs/pipeline/gripper_contact_check.png"):
    """Run a live simulation grasp and print detailed contact and distance metrics."""
    import gymnasium as gym
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from isaaclab.app import launch_simulation
    os.environ["SO101_POLICY"] = "rgb"
    os.environ["SO101_CURRICULUM"] = "fixed"
    os.environ["SO101_CUBE_JITTER"] = "0"
    os.environ["SO101_BOX_JITTER"] = "0"

    from isaaclab.utils.math import matrix_from_quat
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.scene.camera.width = 512
    cfg.scene.camera.height = 512
    if cfg.events.reset_cube_pose is not None:
        cfg.events.reset_cube_pose.params["pose_range"] = {}
        cfg.events.reset_cube_pose.params["velocity_range"] = {}
    cfg.events.randomize_cube_mass = None
    cfg.events.randomize_cube_friction = None

    with launch_simulation(cfg, {"device": device, "visualizer": viz}):
        geometry = GraspGeometry(cfg.scene.robot.spawn.usd_path, cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot)
        center = np.array(cfg.scene.cube.init_state.pos)
        center[2] = 0.425
        home = [cfg.scene.robot.init_state.joint_pos[n] for n in ARM_NAMES]
        above = geometry.solve(center + [0, 0, 0.04], home)
        grasp = geometry.solve(center, above)
        raised = geometry.solve(center + [0, 0, 0.08], grasp)
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(ARM_NAMES, above.tolist(), strict=True)))
        cfg.scene.robot.init_state.joint_pos["gripper"] = 1.2
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        env.reset()
        unwrapped = env.unwrapped
        robot, cube = unwrapped.scene["robot"], unwrapped.scene["cube"]

        def to_torch(x):
            return x.torch if hasattr(x, "torch") else x

        ids = [robot.find_joints(n)[0][0] for n in ARM_NAMES + ["gripper"]]
        q = to_torch(robot.data.joint_pos).clone()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))

        def step(target):
            q[:, ids] = torch.tensor(target, device=device, dtype=torch.float32)
            robot.set_joint_position_target(q)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step()
            unwrapped.scene.update(cfg.sim.dt)

        def move(start, end, seconds):
            for alpha in np.linspace(0, 1, int(seconds / cfg.sim.dt)):
                step((1 - alpha) * np.array(start) + alpha * np.array(end))

        opened = np.r_[above, 1.2]
        down = np.r_[grasp, 1.2]
        closed = np.r_[grasp, 0.0]
        up = np.r_[raised, 0.0]

        print("\nExecuting grasp and lift...")
        move(opened, opened, 0.5)
        move(opened, down, 2.0)
        move(down, closed, 1.0)
        move(closed, up, 2.0)
        for _ in range(25):
            step(up)

        # Contact and pose telemetry
        cube_pos = to_torch(cube.data.root_pos_w)[0].cpu().numpy()
        cube_quat = to_torch(cube.data.root_quat_w)
        R_cube = matrix_from_quat(cube_quat)[0].cpu().numpy()
        gripper_q = float(to_torch(robot.data.joint_pos)[0, ids[-1]])
        bilateral = bool(bilateral_cube_contact(unwrapped)[0].item())

        print("\n" + "=" * 60)
        print("SIMULATION GRASP TELEMETRY")
        print("=" * 60)
        print(f"Gripper Joint Angle (clamped): {gripper_q:.4f} rad ({np.rad2deg(gripper_q):.1f} deg)")
        print(f"Cube World Position:            [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}] m")
        print(f"Cube Lifted Height Above Table: {(cube_pos[2] - 0.400) * 100:.2f} cm")
        print(f"Bilateral Fingertip Grasp:     {'PASSED' if bilateral else 'FAILED'}")

        for sensor_name, body_name, lower, upper in (
            ("fixed_finger_contact", "gripper_link", (-0.028, -0.010, -0.106), (-0.005, 0.010, -0.064)),
            ("moving_finger_contact", "moving_jaw_so101_v1_link", (-0.0133, -0.083, -0.0061), (0.011, -0.040, 0.0439)),
        ):
            sensor = unwrapped.scene[sensor_name]
            forces = to_torch(sensor.data.normal_force_matrix_w)[0].reshape(-1, 3)
            pts = to_torch(sensor.data.contact_pos_w)[0].reshape(-1, 3)
            body_id = robot.find_bodies(body_name)[0][0]
            body_pos = to_torch(robot.data.body_pos_w)[0, body_id]
            body_quat = to_torch(robot.data.body_quat_w)[0, body_id]
            rot = matrix_from_quat(body_quat.unsqueeze(0))[0]
            local_pts = ((pts - body_pos) @ rot).cpu().numpy()

            valid = (forces.abs().sum(dim=-1) > 0.01).cpu().numpy()
            force_mag = torch.linalg.vector_norm(forces.sum(dim=0)).item()

            print("-" * 60)
            print(f"Sensor: {sensor_name} (Body: {body_name})")
            print(f"  Total Normal Force:     {force_mag:.3f} N")
            if valid.any():
                v_pts = pts[valid].cpu().numpy()[0]
                v_local = local_pts[valid][0]
                # Distance to cube faces (cube half-size = 0.025m)
                cube_local_pt = (v_pts - cube_pos) @ R_cube
                face_axis = int(np.argmax(np.abs(cube_local_pt)))
                face_dist_mm = (np.abs(cube_local_pt[face_axis]) - 0.025) * 1000.0
                face_name = ["X", "Y", "Z"][face_axis]

                inside_cad = all(lower[i] <= v_local[i] <= upper[i] for i in range(3))
                print(f"  Contact World Pos:      [{v_pts[0]:.4f}, {v_pts[1]:.4f}, {v_pts[2]:.4f}] m")
                print(f"  Contact Link-Local Pos: [{v_local[0]:.4f}, {v_local[1]:.4f}, {v_local[2]:.4f}] m")
                print(f"  Inside Fingertip CAD:   {inside_cad}")
                print(f"  Point on Cube Local:    [{cube_local_pt[0]:.4f}, {cube_local_pt[1]:.4f}, {cube_local_pt[2]:.4f}] m")
                print(f"  Contact on Cube Face:   {face_name} (Deviation from face: {face_dist_mm:+.2f} mm)")
            else:
                print("  No active contact detected.")

        print("=" * 60)

        # Save snapshot
        target = Path(output_image)
        target.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            unwrapped.sim.render()
            unwrapped.scene["camera"].update(cfg.sim.dt, force_recompute=True)
        rgb = to_torch(unwrapped.scene["camera"].data.output["rgb"])[0, :, :, :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(str(target))
        print(f"Camera snapshot saved to: {target}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cad-profile", action="store_true", help="Inspect only the CAD mesh surfaces without launching simulation")
    p.add_argument("--viz", choices=["kit", "newton"], default=None, help="Visualizer backend")
    p.add_argument("--device", default="cuda:0", help="Computation device (e.g. cuda:0, cpu)")
    p.add_argument("--asset", default=str(ROOT / "usd/robot_grasp_surface_v3.usda"), help="USD robot asset path")
    p.add_argument("--output", default="outputs/pipeline/gripper_contact_check.png", help="Path to save camera snapshot")
    args = p.parse_args()

    if args.cad_profile:
        inspect_cad_profile(args.asset)
    else:
        inspect_sim_contact(device=args.device, viz=args.viz, output_image=args.output)


if __name__ == "__main__":
    main()
