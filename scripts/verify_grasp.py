#!/usr/bin/env python3
"""Gate zero: solve a grasp, then verify closure and sustained lifting in physics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/pipeline/physical_grasp.json")
    p.add_argument("--viz", choices=["kit"])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
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
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES, GraspGeometry

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.camera.width = 512
    cfg.scene.camera.height = 512
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
        raised = geometry.solve(center + [0, 0, 0.06], grasp)
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(ARM_NAMES, above.tolist(), strict=True)))
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
                target = ROOT / "outputs" / "pipeline" / (name + ".png")
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
            for _ in range(100):
                step(up)
                pos = tensor(cube.data.root_pos_w)[0]
                tool = tensor(robot.data.body_pos_w)[0, body]
                records.append(
                    {
                        "cube_z": float(pos[2]),
                        "distance": float(torch.norm(pos - tool)),
                        "gripper": float(tensor(robot.data.joint_pos)[0, ids[-1]]),
                    }
                )
            snapshot("grasp_raised")
            passed = fk_error < 0.005 and all(r["cube_z"] > 0.47 and r["distance"] < 0.06 for r in records[-50:])
            report = {
                "passed": passed,
                "fk_error_m": fk_error,
                "minimum_hold_height_m": min(r["cube_z"] for r in records[-50:]),
                "maximum_hold_distance_m": max(r["distance"] for r in records[-50:]),
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
