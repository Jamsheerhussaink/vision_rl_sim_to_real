"""Both vision tasks must refresh camera data after episode resets."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class CameraResetTests(unittest.TestCase):

    def test_lift_and_place_refresh_without_changing_policy_timing(self):
        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg

        with patch.dict(os.environ, {"SO101_CURRICULUM": "lift", "SO101_CUBE_JITTER": "0.01", "SO101_RGB_FPS": "30"}):
            lift = PickLiftEnvCfg()
        with patch.dict(os.environ, {"SO101_CURRICULUM": "place", "SO101_CUBE_JITTER": "0.01", "SO101_RGB_FPS": "30"}):
            place = PickplaceEnvCfg()
        for cfg in (lift, place):
            self.assertEqual(cfg.num_rerenders_on_reset, 1)
            self.assertAlmostEqual(cfg.decimation * cfg.sim.dt, 1 / 30)
            self.assertEqual(cfg.episode_length_s, 6.0)
