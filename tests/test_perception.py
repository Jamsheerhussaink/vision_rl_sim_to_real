"""Synthetic calibration and depth checks independent of simulation ground truth."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_vision_rl.perception.cube import (
    YellowCubeEstimator,
    fit_camera_to_base,
    load_calibration,
    validate_transform,
)


class PerceptionTests(unittest.TestCase):
    def scene(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:] = [20, 100, 25]
        depth = np.full((480, 640), 0.6)
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        T = np.diag([1.0, -1.0, -1.0, 1.0])
        T[:3, 3] = [0, 0, 1]
        rgb[160:215, 402:457] = [245, 220, 15]
        depth[160:215, 402:457] = 0.55
        return rgb, depth, K, T

    def test_top_surface_is_corrected_to_cube_center(self):
        rgb, depth, K, T = self.scene()
        estimate = YellowCubeEstimator(T).estimate_depth(rgb, depth, K)
        self.assertTrue(estimate.valid)
        self.assertAlmostEqual(estimate.center_base_m[2], 0.425, places=6)
        self.assertAlmostEqual(estimate.center_base_m[0], 0.10, delta=0.002)

    def test_empty_or_bad_depth_does_not_fabricate_a_position(self):
        rgb, depth, K, T = self.scene()
        depth[:] = 0
        estimate = YellowCubeEstimator(T).estimate_depth(rgb, depth, K)
        self.assertFalse(estimate.valid)
        self.assertIsNone(estimate.center_base_m)

    def test_reflection_is_not_a_calibrated_rotation(self):
        T = np.eye(4)
        T[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transform(T)

    def test_measured_transform_and_degenerate_landmarks(self):
        points = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1], [0.1, 0.1, 1], [0.05, 0.02, 1], [0.02, 0.08, 1.0]])
        R = np.diag([1.0, -1.0, -1.0])
        t = np.array([0.2, 0.3, 1.0])
        T, error = fit_camera_to_base(points, points @ R.T + t)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-10)
        self.assertLess(max(error), 1e-10)
        with self.assertRaises(ValueError):
            fit_camera_to_base(np.ones((6, 3)), np.ones((6, 3)))

    def test_live_camera_rejects_simulation_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "calibration.json"
            p.write_text(
                json.dumps(
                    {
                        "source": "simulation",
                        "units": "metres",
                        "camera_frame": "color_optical",
                        "base_z_up": True,
                        "T_base_camera": np.eye(4).tolist(),
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "measured"):
                load_calibration(p)

    def test_two_yellow_cubes_are_ambiguous(self):
        rgb, depth, K, T = self.scene()
        rgb[160:215, 210:265] = [245, 220, 15]
        depth[160:215, 210:265] = 0.55
        estimate = YellowCubeEstimator(T).estimate_depth(rgb, depth, K)
        self.assertFalse(estimate.valid)
        self.assertIn("ambiguous", estimate.reason)

    def test_actor_camera_input_never_reads_simulator_cube(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock

        import torch

        from isaac_vision_rl.tasks.pickplace.mdp.observations import camera_cube_observation

        rgb, depth, K, T = self.scene()
        camera = NS(
            update=Mock(),
            data=NS(
                output={"rgb": torch.from_numpy(rgb[None]), "distance_to_image_plane": torch.from_numpy(depth[None])},
                intrinsic_matrices=torch.from_numpy(K[None]),
                pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
                quat_w_ros=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            ),
        )
        robot = NS(
            find_bodies=lambda name: ([0], [name]),
            data=NS(
                root_pos_w=torch.zeros(1, 3),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                body_pos_w=torch.zeros(1, 1, 3),
            ),
        )
        # There is deliberately no cube entry in this scene.
        env = NS(
            scene={"camera": camera, "robot": robot},
            num_envs=1,
            device="cpu",
            step_dt=0.04,
            common_step_counter=0,
            episode_length_buf=torch.zeros(1, dtype=torch.long),
        )
        value = camera_cube_observation(env)
        self.assertEqual(value.shape, (1, 7))
        self.assertEqual(float(value[0, 6]), 1.0)
        self.assertAlmostEqual(float(value[0, 2]), 0.425, places=6)
        self.assertIs(value, camera_cube_observation(env))
        camera.data.output["distance_to_image_plane"].zero_()
        env.common_step_counter += 1
        self.assertTrue(torch.equal(camera_cube_observation(env), torch.zeros(1, 7)))
