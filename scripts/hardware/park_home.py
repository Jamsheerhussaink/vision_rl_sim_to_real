#!/usr/bin/env python3
"""Smoothly and safely bring the SO-101 robot arm back to its home parked stance on the desk."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isaac_vision_rl.utils.home_pose import HOME_TICKS

try:
    import scservo_sdk as scs
except ImportError:
    print("scservo_sdk not found, please run with conda run -n lerobot python")
    sys.exit(1)

PORT = "/dev/ttyACM0"
BAUD = 1_000_000

# Measured photo home. Arbitrary direct returns are blocked below.
TARGET_HOME = list(HOME_TICKS)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def park_robot(port: str = PORT, baud: int = BAUD, leave_torque: bool = True):
    port_handler = scs.PortHandler(port)

    def _patch_set_packet_timeout(ph, packet_length):
        ph.packet_start_time = ph.getCurrentTime()
        ph.packet_timeout = (ph.tx_time_per_byte * packet_length) + (ph.tx_time_per_byte * 3.0) + 50

    port_handler.setPacketTimeout = _patch_set_packet_timeout.__get__(port_handler, scs.PortHandler)
    packet_handler = scs.PacketHandler(0)

    if not port_handler.openPort():
        print(f"Failed to open port {port}")
        sys.exit(1)
    if not port_handler.setBaudRate(baud):
        print(f"Failed to set baudrate {baud}")
        port_handler.closePort()
        sys.exit(1)

    print(f"Connected to SO-101 on {port} at {baud} baud.")

    # 1. Read current positions
    current_pos = []
    for m in range(1, 7):
        pos, comm, err = packet_handler.read2ByteTxRx(port_handler, m, 56)
        if comm != scs.COMM_SUCCESS:
            print(f"Error reading motor {m}: comm={comm}, err={err}")
            port_handler.closePort()
            sys.exit(1)
        current_pos.append(pos)
        print(f"  Motor {m} ({JOINT_NAMES[m-1]:<15}): current={pos} ticks -> target={TARGET_HOME[m-1]} ticks")

    # 2. Lock current positions before enabling torque (prevents jump)
    for m in range(1, 7):
        packet_handler.write2ByteTxRx(port_handler, m, 42, current_pos[m - 1])
        time.sleep(0.005)
        packet_handler.write1ByteTxRx(port_handler, m, 40, 1)
        time.sleep(0.005)

    print("\nTorque engaged smoothly at current positions. Moving gently to safe table rest pose...")

    max_err = max(abs(a - b) for a, b in zip(current_pos, TARGET_HOME))
    if max_err <= 30:
        # Already at or resting near table cradle
        steps = 48
        for s in range(1, steps + 1):
            frac = s / steps
            smooth_frac = 0.5 * (1.0 - math.cos(frac * math.pi))
            for m in range(1, 7):
                target = int(current_pos[m - 1] + smooth_frac * (TARGET_HOME[m - 1] - current_pos[m - 1]))
                packet_handler.write2ByteTxRx(port_handler, m, 42, target)
            time.sleep(0.016)
    else:
        # 3-Stage Clearance Homing
        # Stage 1: Retract/lift end-effector up
        wp_lift = list(current_pos)
        wp_lift[2] = min(current_pos[2], 2600)  # Unfold elbow to raise hand
        wp_lift[3] = min(current_pos[3], 2500)  # Level wrist
        wp_lift[5] = 2300  # Relax gripper

        # Stage 2: Clear pose directly above home cradle
        clear_pose = [TARGET_HOME[0], TARGET_HOME[1], 2900, 2699, TARGET_HOME[4], 2300]

        # A: Lift away from desk (1.0s)
        steps_a = 60
        for s in range(1, steps_a + 1):
            frac = s / steps_a
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            for m in range(1, 7):
                target = int(current_pos[m - 1] + alpha * (wp_lift[m - 1] - current_pos[m - 1]))
                packet_handler.write2ByteTxRx(port_handler, m, 42, target)
            time.sleep(0.016)

        # B: Slew to clear pose above cradle (0.8s)
        steps_b = 48
        for s in range(1, steps_b + 1):
            frac = s / steps_b
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            for m in range(1, 7):
                target = int(wp_lift[m - 1] + alpha * (clear_pose[m - 1] - wp_lift[m - 1]))
                packet_handler.write2ByteTxRx(port_handler, m, 42, target)
            time.sleep(0.016)

        # C: Gently lower into table rest blocks (0.8s)
        steps_c = 48
        for s in range(1, steps_c + 1):
            frac = s / steps_c
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            for m in range(1, 7):
                target = int(clear_pose[m - 1] + alpha * (TARGET_HOME[m - 1] - clear_pose[m - 1]))
                packet_handler.write2ByteTxRx(port_handler, m, 42, target)
            time.sleep(0.016)

    time.sleep(0.1)
    print("\nArrived at table rest stance.")
    for m in range(1, 7):
        final_p, _, _ = packet_handler.read2ByteTxRx(port_handler, m, 56)
        print(f"  Motor {m} ({JOINT_NAMES[m-1]:<15}): final={final_p} ticks")

    # 4. Torque management: holding by default, only unpowered if requested
    if not leave_torque:
        print("\nReleasing torque as requested. Support the arm before it drops...")
        for m in range(1, 7):
            packet_handler.write1ByteTxRx(port_handler, m, 40, 0)
            time.sleep(0.005)
        print("[SUCCESS] Robot motors unpowered.")
    else:
        print("[SUCCESS] Robot in safe elevated home stance. Motors holding position safely above desk.")

    port_handler.closePort()


if __name__ == "__main__":
    release_t = "--release" in sys.argv
    park_robot(leave_torque=not release_t)
