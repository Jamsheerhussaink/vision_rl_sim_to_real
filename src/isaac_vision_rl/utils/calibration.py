"""Hardware Calibration Bridge for SO-101 Robotic Arm (Feetech STS3215 Servos).

Loads real hardware calibration from my_awesome_follower_arm.json and provides
bidirectional conversion between radians (Isaac Sim) and raw servo ticks (0..4095).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

# Default calibration file path
CALIB_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../my_awesome_follower_arm.json"))

# Feetech STS3215 resolution: 4096 ticks = 360 deg = 2 * pi radians
ENCODER_RESOLUTION = 4096.0
TICKS_PER_RAD = ENCODER_RESOLUTION / (2.0 * np.pi)
RAD_PER_TICK = (2.0 * np.pi) / ENCODER_RESOLUTION

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Isaac Sim nominal gripper max open angle (radians)
SIM_GRIPPER_MAX_RAD = 1.2


class FollowerArmCalibration:
    """Manages Feetech STS3215 servo calibration parameters."""

    def __init__(self, calib_path: str = CALIB_FILE_PATH):
        self.calib_path = calib_path
        self.data: dict[str, dict] = {}
        self.servo_ids: list[int] = []
        self.homing_offsets: np.ndarray = np.zeros(6, dtype=np.float64)
        self.range_mins: np.ndarray = np.zeros(6, dtype=np.float64)
        self.range_maxs: np.ndarray = np.zeros(6, dtype=np.float64)
        self.drive_modes: np.ndarray = np.zeros(6, dtype=np.float64)
        self.load_calibration()

    def load_calibration(self) -> None:
        """Loads and parses calibration JSON."""
        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"Calibration file not found at: {self.calib_path}")

        with open(self.calib_path) as f:
            self.data = json.load(f)

        ids, offsets, mins, maxs, modes, zeros = [], [], [], [], [], []
        for joint in JOINT_ORDER:
            info = self.data.get(joint, {})
            ids.append(info.get("id", 1))
            offsets.append(info.get("homing_offset", 2048))
            mins.append(info.get("range_min", 0))
            maxs.append(info.get("range_max", 4095))
            modes.append(info.get("drive_mode", 0))
            zeros.append(info.get("zero_ticks", 2048.0))

        self.servo_ids = ids
        self.homing_offsets = np.array(offsets, dtype=np.float64)
        self.range_mins = np.array(mins, dtype=np.float64)
        self.range_maxs = np.array(maxs, dtype=np.float64)
        self.drive_modes = np.array(modes, dtype=np.float64)
        self.zero_ticks = np.array(zeros, dtype=np.float64)

    def rad_to_ticks(self, q_rad: Sequence[float] | np.ndarray) -> np.ndarray:
        """Convert radians to ticks using each joint's calibrated safe range and zero center.

        For 5 arm joints: converts relative to calibrated zero_ticks (default 2048 ticks).
        For gripper (index 5): maps 0.0 rad (closed) -> range_mins[5] (~2029 ticks)
        and 1.2 rad (open) -> range_maxs[5] (~2910 ticks) matching real SO-101 hardware.
        All values are strictly clipped to [range_min, range_max].
        """
        q_rad = np.asarray(q_rad, dtype=np.float64)
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        ticks = self.zero_ticks + direction * (q_rad * TICKS_PER_RAD)

        # Gripper dedicated linear mapping:
        # On SO-101 physical hardware, range_mins[5] (~2029) is CLOSED and range_maxs[5] (~2910) is OPEN.
        # In Isaac Sim: 0.0 rad is CLOSED, 1.2 rad (SIM_GRIPPER_MAX_RAD) is OPEN.
        if len(q_rad) >= 6:
            grip_rad = np.clip(q_rad[5], 0.0, SIM_GRIPPER_MAX_RAD)
            grip_frac = grip_rad / SIM_GRIPPER_MAX_RAD
            ticks[5] = self.range_mins[5] + grip_frac * (self.range_maxs[5] - self.range_mins[5])

        return np.clip(np.round(ticks), self.range_mins, self.range_maxs).astype(int)

    def ticks_to_rad(self, ticks: Sequence[int] | np.ndarray) -> np.ndarray:
        """Converts raw encoder ticks to joint angles (rad)."""
        ticks = np.asarray(ticks, dtype=np.float64)
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        q_rad = (ticks - self.zero_ticks) / (direction * TICKS_PER_RAD)

        # Gripper inverse mapping: range_mins[5] (closed) -> 0.0 rad, range_maxs[5] (open) -> 1.2 rad
        if len(ticks) >= 6:
            grip_span = max(1.0, self.range_maxs[5] - self.range_mins[5])
            grip_frac = (ticks[5] - self.range_mins[5]) / grip_span
            q_rad[5] = np.clip(grip_frac * SIM_GRIPPER_MAX_RAD, 0.0, SIM_GRIPPER_MAX_RAD)

        return q_rad.astype(np.float64)

    def dict_to_ticks(self, joint_dict: dict[str, float]) -> dict[str, int]:
        """Convert a dictionary of {joint_name: angle_rad} to {joint_name: ticks}."""
        angles = [joint_dict.get(name, 0.0) for name in JOINT_ORDER]
        ticks = self.rad_to_ticks(angles)
        return {name: int(t) for name, t in zip(JOINT_ORDER, ticks)}

    def ticks_to_dict(self, ticks: Sequence[int]) -> dict[str, float]:
        """Convert a sequence of raw ticks to {joint_name: angle_rad}."""
        angles = self.ticks_to_rad(ticks)
        return {name: float(a) for name, a in zip(JOINT_ORDER, angles)}

    def get_joint_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower_limits_rad, upper_limits_rad) derived from real calibration."""
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        lim_a = (self.range_mins - self.zero_ticks) / (direction * TICKS_PER_RAD)
        lim_b = (self.range_maxs - self.zero_ticks) / (direction * TICKS_PER_RAD)
        lower = np.minimum(lim_a, lim_b)
        upper = np.maximum(lim_a, lim_b)


        # Explicitly enforce safe physical wrist_roll bounds [-2.70, +1.48] rad
        lower[4] = -2.70
        upper[4] = 1.48

        # Set gripper bounds
        lower[5] = 0.0
        upper[5] = SIM_GRIPPER_MAX_RAD
        return lower, upper

    def summary(self) -> str:
        """Returns human-readable calibration table."""
        lower, upper = self.get_joint_limits_rad()
        lines = [
            f"{'Joint':<15} {'ID':<4} {'Homing':<8} {'Raw Range':<14} {'Rad Range':<22} {'Deg Range'}",
            "-" * 80,
        ]
        for i, joint in enumerate(JOINT_ORDER):
            raw_str = f"[{int(self.range_mins[i])}, {int(self.range_maxs[i])}]"
            rad_str = f"[{lower[i]:.2f}, {upper[i]:.2f}]"
            deg_str = f"[{np.rad2deg(lower[i]):.1f}°, {np.rad2deg(upper[i]):.1f}°]"
            lines.append(
                f"{joint:<15} {self.servo_ids[i]:<4} {int(self.homing_offsets[i]):<8} "
                f"{raw_str:<14} {rad_str:<22} {deg_str}"
            )
        return "\n".join(lines)


# Singleton instance
calibration = FollowerArmCalibration()
