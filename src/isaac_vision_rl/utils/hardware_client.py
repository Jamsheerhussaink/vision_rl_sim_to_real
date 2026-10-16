"""Non-blocking UDP stream client for real-time Hardware-in-the-Loop co-execution.

Streams 60 Hz simulated joint references to the external hardware bridge daemon,
with precise wall-clock pacing to synchronize simulation and real physical time.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class HardwareStreamer:
    """Streams 6-DoF robot joint targets over UDP to an external hardware driver."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9870,
        joint_names: Sequence[str] = DEFAULT_JOINT_ORDER,
        target_rate_hz: float = 60.0,
    ):
        self.host = host
        self.port = port
        self.joint_names = list(joint_names)
        self.target_rate_hz = target_rate_hz
        self.target_dt = 1.0 / target_rate_hz
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._last_step_time: float | None = None
        self._sent_count = 0

    @property
    def sent_count(self) -> int:
        return self._sent_count

    def send_joint_targets(
        self,
        joint_angles_rad: Sequence[float] | dict[str, float],
        is_reset: bool = False,
    ) -> bool:
        """Encode and transmit joint position targets in radians."""
        now = time.time()
        self._seq += 1

        if isinstance(joint_angles_rad, dict):
            joints_dict = {k: float(joint_angles_rad[k]) for k in self.joint_names if k in joint_angles_rad}
        else:
            joints_dict = {
                name: float(angle) for name, angle in zip(self.joint_names, joint_angles_rad)
            }

        payload = {
            "seq": self._seq,
            "timestamp": now,
            "dt": self.target_dt,
            "joints": joints_dict,
            "is_reset": bool(is_reset),
        }

        data = json.dumps(payload).encode("utf-8")
        try:
            self._sock.sendto(data, (self.host, self.port))
            self._sent_count += 1
            return True
        except OSError as e:
            logger.warning(f"HardwareStreamer UDP send failed: {e}")
            return False

    def send_stop(self) -> None:
        """Send explicit stop / disconnect packet to trigger graceful hardware park."""
        now = time.time()
        self._seq += 1
        payload = {
            "seq": self._seq,
            "timestamp": now,
            "dt": self.target_dt,
            "joints": {},
            "is_reset": True,
            "stop": True,
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            self._sock.sendto(data, (self.host, self.port))
        except OSError:
            pass

    def pace(self, step_dt: float | None = None) -> float:
        """Sleep remainder of control period to lock simulation to real wall-clock time."""
        dt = step_dt if step_dt is not None else self.target_dt
        now = time.perf_counter()

        if self._last_step_time is None:
            self._last_step_time = now
            return 0.0

        elapsed = now - self._last_step_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_step_time = time.perf_counter()
        return max(0.0, sleep_time)

    def reset_timer(self) -> None:
        self._last_step_time = time.perf_counter()

    def close(self, send_stop: bool = False) -> None:
        try:
            if send_stop:
                self.send_stop()
            self._sock.close()
        except Exception:
            pass

