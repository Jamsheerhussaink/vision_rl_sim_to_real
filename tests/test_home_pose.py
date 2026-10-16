"""Photo home calibration and no-motion bridge shutdown regression checks."""

import unittest
from unittest.mock import Mock

from isaac_vision_rl.utils.calibration import FollowerArmCalibration, JOINT_ORDER
from isaac_vision_rl.utils.home_pose import HOME_TICKS, home_joint_positions
from isaac_vision_rl.utils.table_clearance import HandTableClearance
from pathlib import Path
from scripts.hardware.hardware_bridge import HardwareBridgeDaemon


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


if __name__ == "__main__":
    unittest.main()
