"""Verify RGB training never supplies depth geometry or capture assistance."""
import os
import unittest
from unittest.mock import patch


class RGBPolicyTest(unittest.TestCase):
    def test_rgb_actor_and_controller(self):
        with patch.dict(os.environ, {"SO101_POLICY": "rgb", "SO101_CURRICULUM": "fixed",
                                    "SO101_CUBE_JITTER": "0", "SO101_BOX_JITTER": "0",
                                    "SO101_WRIST_ONLY_CONTROL": "0", "SO101_SMOOTH_CONTROL": "0"}):
            from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
            cfg = PickLiftEnvCfg()
            self.assertEqual(cfg.scene.camera.data_types, ["rgb"])
            self.assertEqual((cfg.scene.camera.width, cfg.scene.camera.height), (1280, 720))
            self.assertFalse(hasattr(cfg.observations.policy, "camera_cube"))
            self.assertFalse(hasattr(cfg.observations.policy, "assist_state"))
            self.assertEqual(cfg.actions.gripper_action.class_type.rsplit(":", 1)[-1], "PolicyGripperAction")
            self.assertEqual(cfg.rewards.reach_progress.params, {"time_normalized": True})
            self.assertAlmostEqual(cfg.sim.dt, 1 / 120)
            self.assertAlmostEqual(cfg.sim.dt * cfg.decimation, 1 / 30)
            self.assertEqual(cfg.actions.gripper_action.scale, 0.6)
            self.assertEqual(cfg.actions.gripper_action.offset, 0.6)
            self.assertEqual(cfg.events.reset_cube_pose.params["pose_range"], {})

