#!/usr/bin/env python3
"""Validate the RGB/depth estimator against scored simulation poses; no policy training."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--viz", choices=["kit", "newton_gl"])
    p.add_argument("--robot-visible", action="store_true", help="Also test visibility around the robot")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs/perception/sim_validation.json")
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

    from isaac_vision_rl.perception.cube import YellowCubeEstimator
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    from isaac_vision_rl.utils.pipeline import camera_fingerprint

    cfg = PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.scene.camera.height = 512
    cfg.scene.camera.width = 512
    cfg.scene.camera.data_types = ["rgb", "distance_to_image_plane"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, {"device": args.device, "visualizer": args.viz}):
        env = gym.make("IsaacVisionRl-PickLift-SO101", cfg=cfg)
        try:
            from pxr import UsdGeom

            base = cast("ManagerBasedRLEnv", env.unwrapped)
            env.reset()
            camera, cube, robot = base.scene["camera"], base.scene["cube"], base.scene["robot"]

            def tensor(x):
                return x.torch if hasattr(x, "torch") else x

            def transform(pos, quat):
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(tensor(quat).cpu().numpy()).as_matrix()
                T[:3, 3] = tensor(pos).cpu().numpy()
                return T

            for _ in range(4):
                base.sim.render()
                camera.update(cfg.sim.dt, force_recompute=True)
            T_world_base = transform(robot.data.root_pos_w[0], robot.data.root_quat_w[0])
            T_world_camera = transform(camera.data.pos_w[0], camera.data.quat_w_ros[0])
            T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera
            (output.parent / "sim_camera_to_base.json").write_text(
                json.dumps(
                    {
                        "source": "simulation",
                        "units": "metres",
                        "camera_frame": "color_optical",
                        "base_z_up": True,
                        "T_base_camera": T_base_camera.tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            detector = YellowCubeEstimator(T_base_camera)
            # Isolate perception geometry from robot occlusion in these scored cases.
            # Physics is unchanged; this is not a learned-policy or grasp-success test.
            imageable = UsdGeom.Imageable(base.sim.stage.GetPrimAtPath(base.scene.env_prim_paths[0] + "/Robot"))
            visibility = imageable.GetVisibilityAttr()
            if visibility is None:
                raise RuntimeError("Robot has no visibility attribute")
            if not args.robot_visible:
                visibility.Set("invisible")
            records = []
            for index, (dx, dy, dz, yaw) in enumerate(
                [(0, 0, 0, 0), (0.025, -0.02, 0, 0.35), (-0.025, 0.02, 0, -0.35), (0, 0, 0.06, 0.25)]
            ):
                state = tensor(cube.data.default_root_state).clone()
                state[:, :3] += base.scene.env_origins
                state[:, 0] += dx
                state[:, 1] += dy
                state[:, 2] += dz
                state[:, 3:7] = torch.tensor(
                    Rotation.from_euler("z", yaw).as_quat(), device=args.device, dtype=torch.float32
                )
                cube.write_root_state_to_sim(state)
                base.scene.write_data_to_sim()
                # Advance physics once so RTX/Fabric receives the new object pose.
                # Scoring uses the resulting actual pose, including this gravity step.
                base.sim.step()
                base.scene.update(cfg.sim.dt)
                for _ in range(8):
                    base.sim.render()
                    camera.update(cfg.sim.dt, force_recompute=True)
                rgb = tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype("uint8")
                depth = tensor(camera.data.output["distance_to_image_plane"])[0].cpu().numpy().squeeze()
                K = tensor(camera.data.intrinsic_matrices)[0].cpu().numpy()
                estimate = detector.estimate_depth(rgb, depth, K)
                # Simulator cube pose is accessed ONLY here, for error scoring.
                world = tensor(cube.data.root_pos_w)[0].cpu().numpy()
                truth = T_world_base[:3, :3].T @ (world - T_world_base[:3, 3])
                error = float(np.linalg.norm(np.array(estimate.center_base_m) - truth)) if estimate.valid else None
                records.append(
                    {"case": index, "estimate": estimate.to_dict(), "truth_base_m": truth.tolist(), "error_m": error}
                )
                Image.fromarray(rgb).save(output.parent / f"camera_case_{index}.png")
                np.savez_compressed(
                    output.parent / f"camera_case_{index}.npz",
                    rgb=rgb,
                    depth_z_m=depth,
                    K=K,
                    T_base_camera=T_base_camera,
                )
                print(records[-1], flush=True)
            passed = all(r["error_m"] is not None and r["error_m"] < 0.01 for r in records)
            report = {
                "camera_fingerprint": camera_fingerprint(),
                "passed": passed,
                "ground_truth_used_for_estimation": False,
                "ground_truth_used_for_scoring": True,
                "render_only_pose_cases": True,
                "robot_hidden_for_clear_view_test": not args.robot_visible,
                "position_tolerance_m": 0.01,
                "cases": records,
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print("Camera perception validation:", passed, flush=True)
            return 0 if passed else 2
        finally:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
