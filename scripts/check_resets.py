#!/usr/bin/env python3
"""Verify cube position, orientation and velocity restoration for all and partial resets."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/pipeline/reset_check.json")
    args = parser.parse_args()
    os.environ["SO101_CURRICULUM"] = "fixed"
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation

    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"passed": False, "status": "running or interrupted"}) + "\n")
    with launch_simulation(cfg, {"device": args.device, "visualizer": None}):
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        try:
            base = cast("ManagerBasedRLEnv", env.unwrapped)
            cube = base.scene["cube"]

            def tensor(x):
                return x.torch if hasattr(x, "torch") else x

            env.reset()
            expected = tensor(cube.data.default_root_state).clone()
            expected[:, :3] += base.scene.env_origins
            cases = []
            for trial in range(5):
                displaced = expected.clone()
                displaced[:, :3] += torch.tensor([0.12, 0.08, 0.04], device=args.device)
                displaced[:, 3:7] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=args.device)
                displaced[:, 7:] = 0.2
                cube.write_root_state_to_sim(displaced)
                base.sim.forward()
                base.scene.update(cfg.sim.dt)
                before = tensor(cube.data.root_state_w).clone()
                ids = torch.tensor([0, 2], device=args.device, dtype=torch.int32)
                # Runtime vector reset accepts tensor IDs; the SDK annotates Sequence[int].
                base.reset(env_ids=cast(Any, ids))
                partial = tensor(cube.data.root_state_w).clone()
                torch.testing.assert_close(partial[ids, :3], expected[ids, :3], atol=1e-5, rtol=0)
                torch.testing.assert_close(partial[ids, 3:7], expected[ids, 3:7], atol=1e-5, rtol=0)
                torch.testing.assert_close(partial[ids, 7:], expected[ids, 7:], atol=1e-5, rtol=0)
                others = torch.tensor([1, 3], device=args.device)
                torch.testing.assert_close(partial[others], before[others], atol=1e-5, rtol=0)
                base.reset()
                after = tensor(cube.data.root_state_w).clone()
                torch.testing.assert_close(after, expected, atol=1e-5, rtol=0)
                cases.append(
                    {
                        "trial": trial,
                        "max_position_error_m": float((after[:, :3] - expected[:, :3]).abs().max()),
                        "max_velocity_error": float((after[:, 7:] - expected[:, 7:]).abs().max()),
                    }
                )
            target.write_text(
                json.dumps(
                    {"passed": True, "envs": 4, "trials": 5, "partial_and_all_resets": True, "cases": cases}, indent=2
                )
                + "\n"
            )
            print("PASS: five repeated full and partial pose/rotation/velocity resets", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
