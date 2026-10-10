#!/usr/bin/env python3
"""PhysX checks for cube support on the table, box floor, and box walls."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs/pipeline/workcell_physics.json")
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": False, "status": "running or interrupted"}) + "\n")
    os.environ["SO101_CURRICULUM"] = "lift"
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg

    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # A minimal scene avoids allocating robot and camera assets for collision checks.
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=cfg.scene.env_spacing)
    for name in ("table", "cube", "box"):
        setattr(scene_cfg, name, getattr(cfg.scene, name))
    spawner = cfg.scene.box.spawn
    if not isinstance(spawner, UsdFileCfg) or spawner.collision_props is None:
        raise TypeError("Workcell check requires explicit box collision settings")
    collisions = spawner.collision_props
    if not isinstance(collisions, PhysxCollisionPropertiesCfg):
        raise TypeError("Box collision check requires PhysX collision properties")
    with launch_simulation(cfg, {"device": args.device, "visualizer": None}):
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        cube, box = scene["cube"], scene["box"]

        def tensor(x):
            return x.torch if hasattr(x, "torch") else x

        def reset_cube(position, velocity=(0, 0, 0)):
            state = tensor(cube.data.default_root_state).clone()
            state[0, :3] = torch.tensor(position, device=args.device)
            state[0, 7:10] = torch.tensor(velocity, device=args.device)
            cube.write_root_state_to_sim(state)
            scene.write_data_to_sim()

        def settle(steps=180, history=None):
            for _ in range(steps):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(cfg.sim.dt)
                if history is not None:
                    history.append(float(tensor(cube.data.root_pos_w)[0, 0]) - box_pos[0])
            return tensor(cube.data.root_pos_w)[0].cpu().tolist()

        # A freely dropped cube must rest on the tabletop, centered at 0.425 m.
        reset_cube([0.05, 0.02, 0.50])
        table = settle()
        box_pos = tensor(box.data.root_pos_w)[0].cpu().tolist()
        # Drop inside the box: center should be floor top + cube half-height.
        reset_cube([box_pos[0], box_pos[1], box_pos[2] + 0.18])
        floor = settle()
        expected_floor_z = box_pos[2] + 0.004 + 0.025
        # Launch toward an interior wall below its top: the cube cannot pass through.
        reset_cube([box_pos[0] + 0.055, box_pos[1], expected_floor_z + 0.001], [1.0, 0, 0])
        wall_history = []
        wall = settle(history=wall_history)
        passed = (
            abs(table[2] - 0.425) < 0.003
            and abs(floor[2] - expected_floor_z) < 0.003
            and abs(wall[0] - box_pos[0]) < 0.073
            and abs(wall[1] - box_pos[1]) < 0.048
            and abs(wall[2] - expected_floor_z) < 0.004
            and 0.065 < max(wall_history) < 0.075
        )
        report = {
            "passed": passed,
            "table_cube_center": table,
            "box_floor_cube_center": floor,
            "expected_box_floor_center_z": expected_floor_z,
            "wall_test_cube_center": wall,
            "box_position": box_pos,
            "max_wall_relative_x": max(wall_history),
            "box_collisions_enabled": collisions.collision_enabled,
        }
        report["env_config_hash"] = hashlib.sha256(
            (ROOT / "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py").read_bytes()
        ).hexdigest()
        report["box_reference_hash"] = hashlib.sha256((ROOT / "usd/box_reference.usda").read_bytes()).hexdigest()
        report["box_source_hash"] = hashlib.sha256((ROOT / "usd/box.usd").read_bytes()).hexdigest()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
