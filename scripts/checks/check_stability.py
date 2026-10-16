#!/usr/bin/env python3
"""Exercise saved-policy or bounded random actions and identify invalid simulation fields."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--curriculum", choices=["grasp", "fixed", "lift"], default="grasp")
    parser.add_argument("--output", default="outputs/pipeline/stability.json")
    parser.add_argument(
        "--fast", action="store_true", help="Cache RGB between 16-step renders for faster physics stress"
    )
    parser.add_argument("--policy", choices=["rgb"], default="rgb")
    args = parser.parse_args()
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum
    import importlib.metadata

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.utils.rl_agent import ExplorationPPO

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.fast:
        cfg.sim.render_interval = cfg.decimation * 16
        cfg.scene.camera.update_period = cfg.sim.dt * cfg.decimation * 16
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": False, "status": "running or interrupted"}) + "\n")
    with launch_simulation(cfg, {"device": args.device, "visualizer": None}):
        env = RslRlVecEnvWrapper(
            cast("ManagerBasedRLEnv", gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)), clip_actions=1.0
        )
        try:
            runner = None
            obs = env.get_observations()
            if args.checkpoint:
                agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
                runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
                runner.load(str(Path(args.checkpoint).resolve()))
                policy = runner.get_inference_policy()

            def tensor(x):
                return x.torch if hasattr(x, "torch") else x

            def invalid_fields(obs):
                robot, cube = env.unwrapped.scene["robot"], env.unwrapped.scene["cube"]
                fields = {
                    **{f"obs/{k}": v for k, v in obs.items()},
                    "robot/joint_pos": tensor(robot.data.joint_pos),
                    "robot/joint_vel": tensor(robot.data.joint_vel),
                    "robot/body_pos": tensor(robot.data.body_pos_w),
                    "robot/body_quat": tensor(robot.data.body_quat_w),
                    "cube/root_pos": tensor(cube.data.root_pos_w),
                    "cube/root_quat": tensor(cube.data.root_quat_w),
                }
                bad = {}
                for name, value in fields.items():
                    mask = ~torch.isfinite(value).reshape(args.envs, -1).all(dim=-1)
                    if mask.any():
                        ids = torch.nonzero(mask).flatten()
                        bad[name] = {"env_ids": ids.cpu().tolist(), "values": value[ids[:2]].cpu().tolist()}
                return bad

            with torch.inference_mode():
                for step in range(args.steps):
                    if runner:
                        actions = policy(obs) + cast(
                            ExplorationPPO, runner.alg
                        ).exploration_distribution().log_std_param.exp() * torch.randn(args.envs, 6, device=args.device)
                    else:
                        actions = 2 * torch.rand(args.envs, 6, device=args.device) - 1
                    obs, rewards, dones, _ = env.step(actions)
                    bad = invalid_fields(obs)
                    if bad or not torch.isfinite(rewards).all():
                        output.write_text(
                            json.dumps(
                                {
                                    "passed": False,
                                    "step": step,
                                    "bad_fields": bad,
                                    "rewards_finite": bool(torch.isfinite(rewards).all()),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        print("FAILED", step, list(bad), flush=True)
                        return 2
                    if (step + 1) % 500 == 0:
                        print(f"Finite: {step + 1}/{args.steps} steps x {args.envs} environments", flush=True)
            report = {
                "passed": True,
                "steps": args.steps,
                "envs": args.envs,
                "transitions": args.steps * args.envs,
                "checkpoint": args.checkpoint,
                "seed": args.seed,
                "cached_rgb": args.fast,
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(report, flush=True)
            return 0
        finally:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
