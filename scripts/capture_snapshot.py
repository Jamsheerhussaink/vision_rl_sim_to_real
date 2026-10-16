"""Capture the actual task camera in the nominal ready pose."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "place"], default="lift")
    parser.add_argument("--output", default="outputs/training/scene_preview.png")
    parser.add_argument("--width", type=int, default=424)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--policy", choices=["rgb"], default="rgb")
    parser.add_argument("--curriculum", choices=["fixed", "lift", "place"], default="fixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", choices=["kit"], help="Also verify the GUI uses the task camera")
    args = parser.parse_args()
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum
    if args.headless and args.viz:
        parser.error("Choose --headless or --viz kit")
    import gymnasium as gym
    from PIL import Image

    from isaaclab.app import launch_simulation

    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg

    cfg = PickLiftEnvCfg() if args.task == "lift" else PickplaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.scene.camera.width = args.width
    cfg.scene.camera.height = args.height
    cfg.events.reset_robot_joints = None
    if cfg.events.reset_cube_pose is not None:
        cfg.events.reset_cube_pose.params["pose_range"] = {}
        cfg.events.reset_cube_pose.params["velocity_range"] = {}
    task = "IsaacVisionRl-PickLift-SO101" if args.task == "lift" else "IsaacVisionRl-Pickplace-SO101"
    with launch_simulation(cfg, {"device": args.device, "visualizer": args.viz}):
        env = gym.make(task, cfg=cfg)
        try:
            env.reset()
            if args.viz:
                from omni.kit.viewport.utility import get_active_viewport

                viewport = get_active_viewport()
                expected = "/World/envs/env_0/Camera"
                if viewport is None or str(viewport.camera_path) != expected:
                    raise RuntimeError("GUI viewport is not using the task camera")
                print(f"Verified GUI camera: {viewport.camera_path}")
            robot = env.unwrapped.scene["robot"]
            positions = robot.data.default_joint_pos
            velocities = robot.data.default_joint_vel
            if hasattr(positions, "torch"):
                positions, velocities = positions.torch, velocities.torch
            robot.write_joint_state_to_sim(positions, velocities)
            robot.set_joint_position_target(positions)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step()
            env.unwrapped.scene.update(cfg.sim.dt)
            for _ in range(8):
                env.unwrapped.sim.render()
                env.unwrapped.scene["camera"].update(cfg.sim.dt, force_recompute=True)
            rgb = env.unwrapped.scene["camera"].data.output["rgb"][0, :, :, :3]
            if hasattr(rgb, "torch"):
                rgb = rgb.torch
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb.cpu().numpy().astype("uint8")).save(output)
            print(f"Snapshot saved: {output.resolve()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
