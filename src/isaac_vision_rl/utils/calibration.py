"""Hardware Calibration Bridge for SO-101 Robotic Arm (Feetech STS3215 Servos).
Loads real hardware calibration from my_awesome_follower_arm.json and provides
bidirectional conversion between radians and raw servo ticks (0..4095).
"""

import json
import os

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

        ids, offsets, mins, maxs, modes = [], [], [], [], []
        for joint in JOINT_ORDER:
            info = self.data.get(joint, {})
            ids.append(info.get("id", 1))
            offsets.append(info.get("homing_offset", 2048))
            mins.append(info.get("range_min", 0))
            maxs.append(info.get("range_max", 4095))
            modes.append(info.get("drive_mode", 0))

        self.servo_ids = ids
        self.homing_offsets = np.array(offsets, dtype=np.float64)
        self.range_mins = np.array(mins, dtype=np.float64)
        self.range_maxs = np.array(maxs, dtype=np.float64)
        self.drive_modes = np.array(modes, dtype=np.float64)

    def rad_to_ticks(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert radians to ticks using each joint's calibrated safe range.

        The JSON file stores servo range limits, while its signed homing offsets are
        controller calibration metadata rather than a universal zero-angle encoder
        value.  The midpoint of each validated range is therefore used as the
        simulation zero and commands are clipped to the real joint limits.
        """
        q_rad = np.asarray(q_rad, dtype=np.float64)
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        zero_ticks = (self.range_mins + self.range_maxs) / 2.0
        ticks = zero_ticks + direction * (q_rad * TICKS_PER_RAD)
        return np.clip(np.round(ticks), self.range_mins, self.range_maxs).astype(int)

    def ticks_to_rad(self, ticks: np.ndarray) -> np.ndarray:
        """Converts raw encoder ticks to joint angles (rad)."""
        ticks = np.asarray(ticks, dtype=np.float64)
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        zero_ticks = (self.range_mins + self.range_maxs) / 2.0
        q_rad = (ticks - zero_ticks) / (direction * TICKS_PER_RAD)
        return q_rad.astype(np.float64)

    def get_joint_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower_limits_rad, upper_limits_rad) derived from real calibration."""
        direction = np.where(self.drive_modes == 0, 1.0, -1.0)
        zero_ticks = (self.range_mins + self.range_maxs) / 2.0
        lim_a = (self.range_mins - zero_ticks) / (direction * TICKS_PER_RAD)
        lim_b = (self.range_maxs - zero_ticks) / (direction * TICKS_PER_RAD)
        lower = np.minimum(lim_a, lim_b)
        upper = np.maximum(lim_a, lim_b)
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
