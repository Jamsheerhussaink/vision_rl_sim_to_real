#!/usr/bin/env python3
"""Gate zero: solve a grasp, then verify closure and sustained lifting in physics."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/pipeline/physical_grasp.json")
    p.add_argument("--viz", choices=["kit", "newton"])
    p.add_argument("--policy", choices=["rgb"], default="rgb")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = "fixed"
    os.environ["SO101_CUBE_JITTER"] = "0"
    os.environ["SO101_BOX_JITTER"] = "0"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": False, "status": "preflight running or interrupted"}) + "\n")
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    from isaaclab.app import launch_simulation
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    if cfg.events.reset_cube_pose is not None:
        cfg.events.reset_cube_pose.params["pose_range"] = {}
        cfg.events.reset_cube_pose.params["velocity_range"] = {}
    cfg.events.randomize_cube_mass = None
    cfg.events.randomize_cube_friction = None
    spawner = cfg.scene.robot.spawn
    if not isinstance(spawner, UsdFileCfg):
        raise TypeError("Grasp preflight requires a USD robot asset")
    with launch_simulation(cfg, {"device": args.device, "visualizer": args.viz}):
        geometry = GraspGeometry(spawner.usd_path, cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot)
        center = np.array(cfg.scene.cube.init_state.pos)
        center[2] = 0.425
        home = [cfg.scene.robot.init_state.joint_pos[n] for n in ARM_NAMES]
        above = geometry.solve(center + [0, 0, 0.04], home)
        grasp = geometry.solve(center, above)
        raised = geometry.solve(center + [0, 0, 0.08], grasp)
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(ARM_NAMES, above.tolist(), strict=True)))
        # Open before spawning: opening from a closed pose beside the cube can
        # inject large contact impulses before the descent test even begins.
        cfg.scene.robot.init_state.joint_pos["gripper"] = 1.2
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        try:
            env.reset()
            unwrapped = cast("ManagerBasedRLEnv", env.unwrapped)
            robot, cube = unwrapped.scene["robot"], unwrapped.scene["cube"]

            def tensor(x):
                return x.torch if hasattr(x, "torch") else x

            ids = [robot.find_joints(n)[0][0] for n in ARM_NAMES + ["gripper"]]
            q = tensor(robot.data.joint_pos).clone()
            body = robot.find_bodies("gripper_frame_link")[0][0]
            print(
                "INITIAL ROOT",
                tensor(robot.data.root_pos_w).tolist(),
                tensor(robot.data.root_quat_w).tolist(),
                flush=True,
            )
            print("INITIAL Q", tensor(robot.data.joint_pos).tolist(), flush=True)
            print(
                "INITIAL TOOL",
                tensor(robot.data.body_pos_w)[0, body].tolist(),
                "FK",
                geometry.forward(above)[:3, 3].tolist(),
                flush=True,
            )
            print("FIXED BASE", robot.is_fixed_base, flush=True)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            records = []

            def step(target):
                q[:, ids] = torch.tensor(target, device=args.device, dtype=torch.float32)
                robot.set_joint_position_target(q)
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step()
                unwrapped.scene.update(cfg.sim.dt)
                actual = tensor(robot.data.joint_pos)
                if not torch.isfinite(actual).all() or actual.abs().max() > 4:
                    raise RuntimeError("Physical joint state diverged; grasp gate failed before policy training.")

            def move(start, end, seconds):
                for alpha in np.linspace(0, 1, int(seconds / cfg.sim.dt)):
                    step((1 - alpha) * np.array(start) + alpha * np.array(end))

            def snapshot(name):
                print(
                    name,
                    "TOOL",
                    tensor(robot.data.body_pos_w)[0, body].tolist(),
                    "CUBE",
                    tensor(cube.data.root_pos_w)[0].tolist(),
                    "Q",
                    tensor(robot.data.joint_pos)[0].tolist(),
                    flush=True,
                )
                for _ in range(4):
                    unwrapped.sim.render()
                    unwrapped.scene["camera"].update(cfg.sim.dt, force_recompute=True)
                rgb = tensor(unwrapped.scene["camera"].data.output["rgb"])[0, :, :, :3]
                target = output.with_name(output.stem + "_" + name + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb.cpu().numpy().astype("uint8")).save(target)

            opened = np.r_[above, 1.2]
            down = np.r_[grasp, 1.2]
            closed = np.r_[grasp, 0.0]
            up = np.r_[raised, 0.0]
            fk_error = float(
                torch.norm(
                    tensor(robot.data.body_pos_w)[0, body]
                    - torch.tensor(geometry.forward(above)[:3, 3], device=args.device)
                )
            )
            move(opened, opened, 0.5)
            move(opened, down, 2.0)
            snapshot("grasp_open")
            move(down, closed, 1.0)
            snapshot("grasp_closed")
            move(closed, up, 2.0)
            hold_steps = math.ceil(0.5 / cfg.sim.dt)
            for _ in range(2 * hold_steps):
                step(up)
                pos = tensor(cube.data.root_pos_w)[0]
                tool = tensor(robot.data.body_pos_w)[0, body]
                records.append(
                    {
                        "cube_z": float(pos[2]),
                        "distance": float(torch.norm(pos - tool)),
                        "gripper": float(tensor(robot.data.joint_pos)[0, ids[-1]]),
                        "xy_separation_m": float(torch.norm((pos - tool)[:2])),
                        "z_separation_m": float(torch.abs((pos - tool)[2])),
                        "cube_speed_m_s": float(torch.norm(tensor(cube.data.root_lin_vel_w)[0])),
                        "bilateral_contact": bool(bilateral_cube_contact(unwrapped)[0].item()),
                    }
                )
            snapshot("grasp_raised")
            passed = fk_error < 0.005 and all(
                r["cube_z"] > 0.47 and r["distance"] < 0.06 and r["bilateral_contact"]
                for r in records[-hold_steps:]
            )
            strict_held_lift = all(
                r["cube_z"] > 0.47 and r["xy_separation_m"] < 0.035
                and r["z_separation_m"] < 0.045 and r["gripper"] < 1.10
                and r["cube_speed_m_s"] < 0.25 and r["bilateral_contact"]
                for r in records[-hold_steps:]
            )
            report = {
                "passed": passed,
                "strict_held_lift_passed": strict_held_lift,
                "fk_error_m": fk_error,
                "minimum_hold_height_m": min(r["cube_z"] for r in records[-hold_steps:]),
                "maximum_hold_distance_m": max(r["distance"] for r in records[-hold_steps:]),
                "hold_duration_s": hold_steps * cfg.sim.dt,
                "policy": args.policy,
                "final_gripper_rad": records[-1]["gripper"],
                "above_joint_pos": dict(zip(ARM_NAMES, above.tolist(), strict=True)),
                "above_action": (
                    2
                    * (
                        torch.tensor(above, device=args.device)
                        - tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 0]
                    )
                    / (
                        tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 1]
                        - tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 0]
                    )
                    - 1
                )
                .cpu()
                .tolist(),
                "grasp_joint_pos": dict(zip(ARM_NAMES, grasp.tolist(), strict=True)),
                "grasp_action": (
                    2
                    * (
                        torch.tensor(grasp, device=args.device)
                        - tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 0]
                    )
                    / (
                        tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 1]
                        - tensor(robot.data.soft_joint_pos_limits)[0, ids[:-1], 0]
                    )
                    - 1
                )
                .cpu()
                .tolist(),
                "base_pos": cfg.scene.robot.init_state.pos,
                "base_rot": cfg.scene.robot.init_state.rot,
                "robot_asset": spawner.usd_path,
                "asset_hash": file_hash(spawner.usd_path),
                "source_asset_hash": file_hash(ROOT / "usd" / "robot.usd"),
                "env_config_hash": file_hash(ROOT / "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py"),
                "records": records,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)
            return 0 if passed else 2
        finally:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
