#!/usr/bin/env python3
"""Unit tests for Sim-to-Real Hardware Bridge, Calibration, and UDP Streamer."""

import json
import socket
import unittest
from pathlib import Path

import numpy as np

from isaac_vision_rl.utils.calibration import (
    CALIB_FILE_PATH,
    JOINT_ORDER,
    SIM_GRIPPER_MAX_RAD,
    FollowerArmCalibration,
)
from isaac_vision_rl.utils.hardware_client import HardwareStreamer


class TestHardwareCalibration(unittest.TestCase):
    def setUp(self):
        self.calib = FollowerArmCalibration(CALIB_FILE_PATH)

    def test_calibration_loaded(self):
        self.assertEqual(len(self.calib.servo_ids), 6)
        self.assertEqual(len(self.calib.range_mins), 6)
        self.assertEqual(len(self.calib.range_maxs), 6)
        self.assertTrue((self.calib.range_maxs > self.calib.range_mins).all())

    def test_gripper_closed_and_open_mapping(self):
        # 0.0 rad in sim = fully closed -> maps to hardware range_mins[5] (closed)
        zero_angles = np.zeros(6)
        ticks_closed = self.calib.rad_to_ticks(zero_angles)
        self.assertEqual(ticks_closed[5], int(self.calib.range_mins[5]))

        # 1.2 rad in sim = fully open -> maps to hardware range_maxs[5] (open)
        open_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, SIM_GRIPPER_MAX_RAD])
        ticks_open = self.calib.rad_to_ticks(open_angles)
        self.assertEqual(ticks_open[5], int(self.calib.range_maxs[5]))

    def test_ticks_clipped_to_calibrated_limits(self):
        # Extreme angles should be strictly clipped
        huge_angles = np.array([100.0, -100.0, 50.0, -50.0, 100.0, 100.0])
        ticks = self.calib.rad_to_ticks(huge_angles)
        for i in range(6):
            self.assertGreaterEqual(ticks[i], self.calib.range_mins[i])
            self.assertLessEqual(ticks[i], self.calib.range_maxs[i])

    def test_dict_conversion(self):
        sample_dict = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.25,
            "elbow_flex": 1.60,
            "wrist_flex": 0.85,
            "wrist_roll": 0.0,
            "gripper": 0.95,
        }
        ticks_dict = self.calib.dict_to_ticks(sample_dict)
        self.assertEqual(len(ticks_dict), 6)
        for name in JOINT_ORDER:
            self.assertIn(name, ticks_dict)
            idx = JOINT_ORDER.index(name)
            self.assertGreaterEqual(ticks_dict[name], self.calib.range_mins[idx])
            self.assertLessEqual(ticks_dict[name], self.calib.range_maxs[idx])


class TestHardwareStreamerUDP(unittest.TestCase):
    def test_udp_loopback(self):
        # Bind a temporary test receiver
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        test_port = recv_sock.getsockname()[1]
        recv_sock.settimeout(1.0)

        streamer = HardwareStreamer(host="127.0.0.1", port=test_port)
        sample_angles = [0.01, -1.20, 1.55, 0.80, 0.05, 0.50]

        try:
            sent = streamer.send_joint_targets(sample_angles, is_reset=False)
            self.assertTrue(sent)

            data, _ = recv_sock.recvfrom(2048)
            msg = json.loads(data.decode("utf-8"))

            self.assertEqual(msg["seq"], 1)
            self.assertFalse(msg["is_reset"])
            self.assertIn("joints", msg)
            self.assertAlmostEqual(msg["joints"]["shoulder_pan"], 0.01, places=3)
            self.assertAlmostEqual(msg["joints"]["gripper"], 0.50, places=3)

            # Test stop packet
            streamer.send_stop()
            data, _ = recv_sock.recvfrom(2048)
            stop_msg = json.loads(data.decode("utf-8"))
            self.assertTrue(stop_msg.get("stop", False))

        finally:
            streamer.close()
            recv_sock.close()


if __name__ == "__main__":
    unittest.main()

