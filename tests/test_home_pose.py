"""Photo home calibration and no-motion bridge shutdown regression checks."""

import unittest
from unittest.mock import Mock

from isaac_vision_rl.utils.calibration import FollowerArmCalibration, JOINT_ORDER
from isaac_vision_rl.utils.home_pose import HOME_TICKS, home_joint_positions
from isaac_vision_rl.utils.table_clearance import HandTableClearance
from pathlib import Path
import importlib.util
from scripts.hardware.hardware_bridge import HardwareBridgeDaemon


RETURN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hardware/return_policy_home.py"
return_spec = importlib.util.spec_from_file_location("return_policy_home_under_test", RETURN_SCRIPT)
return_module = importlib.util.module_from_spec(return_spec)
return_spec.loader.exec_module(return_module)


class HomePoseTests(unittest.TestCase):
    def test_clearance_rejects_downward_sweep_and_accepts_unfolding(self):
        asset = Path(__file__).resolve().parents[1] / "usd/robot_grasp_surface_v3.usda"
        guard = HandTableClearance(asset, [0, 0, 0.405], [0, 0, 0, 1])
        home = [home_joint_positions()[name] for name in JOINT_ORDER]
        unsafe = list(home)
        unsafe[1] += 0.30
        with self.assertRaises(ValueError):
            guard.check_segment(home, unsafe, margin=0.005)
        raised = list(home)
        raised[2] -= 0.30
        raised[3] -= 0.15
        self.assertGreater(guard.check_segment(home, raised, margin=0.005), 0.005)

    def test_measured_home_round_trip_does_not_clip(self):
        calib = FollowerArmCalibration()
        angles = [home_joint_positions()[name] for name in JOINT_ORDER]
        self.assertEqual(calib.rad_to_ticks(angles).tolist(), list(HOME_TICKS))
        lower, upper = calib.get_joint_limits_rad()
        self.assertTrue(all(lo <= q <= hi for lo, q, hi in zip(lower, angles, upper)))

    def test_shutdown_does_not_move_arm_home(self):
        daemon = HardwareBridgeDaemon.__new__(HardwareBridgeDaemon)
        daemon.bus = Mock()
        daemon.sock = Mock()
        daemon.release_on_exit = False
        daemon.shutdown()
        daemon.bus.park_home_smoothly.assert_not_called()
        daemon.bus.close.assert_called_once_with(disable_torque=False)

    def test_photo_home_return_uses_all_six_measured_ticks(self):
        checkpoint = [2245, 2091, 2320, 2757, 3096, 3545]
        target, source = return_module.select_home_ticks(
            checkpoint, HOME_TICKS, photo_home=True, closed_gripper=False
        )
        self.assertEqual(target, list(HOME_TICKS))
        self.assertEqual(source, "photo_calibrated")

    def test_checkpoint_return_can_close_only_the_gripper(self):
        checkpoint = [2245, 2091, 2320, 2757, 3096, 3545]
        target, source = return_module.select_home_ticks(
            checkpoint, HOME_TICKS, photo_home=False, closed_gripper=True
        )
        self.assertEqual(target[:5], checkpoint[:5])
        self.assertEqual(target[5], HOME_TICKS[5])
        self.assertEqual(source, "checkpoint_with_closed_gripper")

    def test_standalone_photo_home_scene_matches_calibrated_pipeline_layout(self):
        self.assertEqual(return_module.PHOTO_HOME_ROBOT_POSITION, (-0.25, 0.015, 0.405))
        self.assertEqual(return_module.PHOTO_HOME_ROBOT_ROTATION, (0.0, 0.0, 0.0, 1.0))
        self.assertTrue(return_module.PHOTO_HOME_ROBOT_ASSET.is_file())


if __name__ == "__main__":
    unittest.main()
