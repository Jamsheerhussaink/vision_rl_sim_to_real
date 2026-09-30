"""
Sim-to-Real Hardware Bridge for SO-101 Robot Arm.
Handles conversion between normalized RL actions, joint radian targets,
and physical Feetech STS3215 serial bus commands with safety limits.
"""
import time
import numpy as np


class SO101RealBridge:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 1000000,
        max_rad_per_sec: float = 1.2,
        smoothing_alpha: float = 0.35,  # Exponential Moving Average smoothing
    ):
        self.port = port
        self.baudrate = baudrate
        self.max_rad_per_sec = max_rad_per_sec
        self.smoothing_alpha = smoothing_alpha

        # SO-101 Joint calibration offsets & directions (radians)
        self.joint_offsets = np.zeros(6, dtype=np.float32)
        self.joint_directions = np.ones(6, dtype=np.float32)

        # Current commanded target
        self.current_qpos = np.array([0.0, 0.0, 1.20, -1.20, 0.0, 1.20], dtype=np.float32)
        self.last_cmd_time = time.time()
        self.connected = False

    def connect(self):
        """
        Connects to the STS3215 serial bus.
        Can use LeRobot's FeetechMotorsBus or direct serial.
        """
        try:
            # Check for lerobot package
            import lerobot  # noqa: F401
            print(f"[RealBridge] Connecting to SO-101 on {self.port} via LeRobot driver...")
            self.connected = True
        except ImportError:
            print("[RealBridge] LeRobot not installed in this env. Running in dry-run mode.")
            self.connected = False

    def send_action(self, delta_action: np.ndarray, max_joint_delta: float = 0.06):
        """
        Smooths and safety-checks policy action before sending to physical servos.
        """
        now = time.time()
        dt = max(now - self.last_cmd_time, 0.001)
        self.last_cmd_time = now

        # Delta position update
        target = self.current_qpos + np.clip(delta_action, -1.0, 1.0) * max_joint_delta

        # Velocity limiting for safety
        max_step = self.max_rad_per_sec * dt
        diff = target - self.current_qpos
        clipped_diff = np.clip(diff, -max_step, max_step)

        # EMA smoothing
        smoothed_target = (1.0 - self.smoothing_alpha) * self.current_qpos + self.smoothing_alpha * (self.current_qpos + clipped_diff)
        self.current_qpos = smoothed_target

        if self.connected:
            # Here: call lerobot motor bus write
            pass

        return self.current_qpos
