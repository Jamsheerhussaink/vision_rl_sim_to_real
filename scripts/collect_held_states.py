"""Save good "cube held at lift height" states from a trained policy, for the held-start practice.

Run it once (headless) with a checkpoint that can hold the cube. The result is read by the HeldStartReset event,
which is switched on with SO101_HELD_START (see the printed training command).
"""

import argparse
import importlib.metadata
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Saved policy that can lift and hold the cube")
    parser.add_argument("--output", default="outputs/held_states/place.npz")
    parser.add_argument("--count", type=int, default=300, help="How many held states to save")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--min-height", type=float, default=0.526, help="Cube centre height in metres (10 cm lift)")
    parser.add_argument("--min-wrist-cos", type=float, default=0.80, help="1 = hand straight down")
    parser.add_argument("--max-twist-deg", type=float, default=30.0, help="Cube turn away from the jaw faces")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.environ["SO101_POLICY"] = "rgb"
    os.environ["SO101_CURRICULUM"] = "fixed"
    os.environ["SO101_CUBE_JITTER"] = "0"
    os.environ["SO101_BOX_JITTER"] = "0"
    os.environ.pop("SO101_HELD_START", None)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict
    from isaaclab.utils.math import matrix_from_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact

    cfg = PickplaceEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    agent.device = args.device
    agent.seed = args.seed

    joints, cube_pos, cube_quat, wrist_cos, twist = [], [], [], [], []
    pan_values, tool_angles = [], []
    last_saved = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with launch_simulation(cfg, {"device": args.device, "visualizer": None}):
        raw_env = gym.make("IsaacVisionRl-Pickplace-SO101", cfg=cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
            runner.load(str(checkpoint))
            base = raw_env.unwrapped
            robot, cube = base.scene["robot"], base.scene["cube"]
            tool_id = robot.find_bodies("gripper_frame_link")[0][0]
            jaw_id = robot.find_joints("gripper")[0][0]
            pan_id = robot.find_joints("shoulder_pan")[0][0]
            origins = base.scene.env_origins
            origins = origins.torch if hasattr(origins, "torch") else origins
            last_saved = torch.full((args.envs,), -100, device=base.device)
            obs = env.get_observations()
            with torch.inference_mode():
                for step in range(args.max_steps):
                    action = runner.alg.actor(obs, stochastic_output=True)
                    obs, _, _, _ = env.step(action)
                    tool_pos = robot.data.body_pos_w.torch[:, tool_id]
                    base_xy = robot.data.root_pos_w.torch[:, :2]
                    pan = robot.data.joint_pos.torch[:, pan_id]
                    if step % 4 == 0:
                        rel = tool_pos[:, :2] - base_xy
                        pan_values.append(pan.cpu().numpy())
                        tool_angles.append(torch.atan2(rel[:, 1], rel[:, 0]).cpu().numpy())
                    pos = cube.data.root_pos_w.torch
                    quat = cube.data.root_quat_w.torch
                    rotation = matrix_from_quat(robot.data.body_quat_w.torch[:, tool_id])
                    cos_wrist = torch.clamp(-rotation[:, 2, 2], -1.0, 1.0)
                    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                    yaw = torch.rad2deg(torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) % 90.0
                    off = torch.minimum(yaw, 90.0 - yaw)
                    held = (
                        bilateral_cube_contact(base)
                        & (pos[:, 2] >= args.min_height)
                        & (robot.data.joint_pos.torch[:, jaw_id] < 1.10)
                        & (torch.norm(pos - tool_pos, dim=-1) < 0.052)
                        & (torch.norm(cube.data.root_lin_vel_w.torch, dim=-1) < 0.15)
                        & (cos_wrist >= args.min_wrist_cos)
                        & (off <= args.max_twist_deg)
                        & (step - last_saved >= 5)
                    )
                    for i in torch.nonzero(held).flatten().tolist():
                        joints.append(robot.data.joint_pos.torch[i].cpu().numpy())
                        cube_pos.append((pos[i] - origins[i]).cpu().numpy())
                        cube_quat.append(quat[i].cpu().numpy())
                        wrist_cos.append(float(cos_wrist[i]))
                        twist.append(float(off[i]))
                        last_saved[i] = step
                    if len(joints) >= args.count:
                        break
                    if step % 200 == 0:
                        print(f"[collect] step {step}: {len(joints)}/{args.count} held states", flush=True)
        finally:
            env.close()

    if not joints:
        raise SystemExit("No held states found. Try --min-wrist-cos 0.6 or a better checkpoint.")
    pan_values = np.concatenate(pan_values)
    tool_angles = np.concatenate(tool_angles)
    pan_values -= pan_values.mean()
    angle_change = np.unwrap(tool_angles) - np.unwrap(tool_angles).mean()
    pan_sign = 1.0 if float((pan_values * angle_change).sum()) >= 0 else -1.0
    np.savez(
        output,
        joint_pos=np.array(joints, dtype=np.float32),
        cube_pos_local=np.array(cube_pos, dtype=np.float32),
        cube_quat=np.array(cube_quat, dtype=np.float32),
        pan_sign=np.float32(pan_sign),
        pan_joint_index=np.int64(pan_id),
    )
    print(
        f"[collect] saved {len(joints)} held states to {output}; mean wrist cos {np.mean(wrist_cos):.2f}, "
        f"mean cube twist {np.mean(twist):.1f} deg, pan sign {pan_sign:+.0f}"
    )


if __name__ == "__main__":
    main()
