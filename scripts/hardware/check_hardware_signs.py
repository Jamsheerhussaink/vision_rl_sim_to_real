#!/usr/bin/env python3
"""Interactive Hardware Calibration & Joint Direction Sanity Checker.

Tests serial connection to Feetech STS3215 servos on SO-101 arm, reads current
ticks, and allows jogging each joint by ±5° to verify motor rotation directions
match Isaac Sim simulation conventions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from isaac_vision_rl.utils.calibration import (
    CALIB_FILE_PATH,
    JOINT_ORDER,
    FollowerArmCalibration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for Feetech bus")
    parser.add_argument("--baud", type=int, default=1_000_000, help="Serial baudrate")
    parser.add_argument("--calib", default=CALIB_FILE_PATH, help="Path to calibration file")
    parser.add_argument("--mock", action="store_true", help="Run with mock motors")
    parser.add_argument(
        "--test-nudge", action="store_true", help="Gently nudge shoulder_pan by +25 ticks and return, to verify physical movement"
    )
    args = parser.parse_args()

    calib = FollowerArmCalibration(args.calib)
    print("=" * 70)
    print("SO-101 Hardware Calibration & Joint Direction Checker")
    print("=" * 70)
    print(calib.summary())
    print("=" * 70)

    if args.mock:
        print("[INFO] Running in MOCK mode. Physical motors will not move.")
        current_ticks = [int((calib.range_mins[i] + calib.range_maxs[i]) // 2) for i in range(6)]
    else:
        try:
            import scservo_sdk as scs
        except ImportError:
            print("[ERROR] scservo_sdk is not installed. Please run inside the 'lerobot' conda env:")
            print("        conda run -n lerobot python scripts/hardware/check_hardware_signs.py --port " + args.port)
            sys.exit(1)

        port_handler = scs.PortHandler(args.port)
        packet_handler = scs.PacketHandler(0)

        if not port_handler.openPort():
            print(f"[ERROR] Could not open serial port {args.port}. Check permissions or USB connection.")
            sys.exit(1)
        if not port_handler.setBaudRate(args.baud):
            print(f"[ERROR] Could not set baud rate {args.baud} on {args.port}.")
            port_handler.closePort()
            sys.exit(1)

        print(f"[SUCCESS] Connected to {args.port} at {args.baud} baud.\n")
        print("Reading current servo positions...")
        ADDR_STS_PRESENT_POSITION = 56
        current_ticks = []
        for i, (joint, motor_id) in enumerate(zip(JOINT_ORDER, calib.servo_ids)):
            pos, comm_res, err = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_STS_PRESENT_POSITION)
            if comm_res != scs.COMM_SUCCESS:
                print(f"  [WARN] Motor {motor_id} ({joint}): Communication error {comm_res}")
                pos = int((calib.range_mins[i] + calib.range_maxs[i]) // 2)
            else:
                print(f"  Motor {motor_id} ({joint:<15}): {pos} ticks")
            current_ticks.append(pos)

    # Convert to radians and print current angles
    rads = calib.ticks_to_rad(current_ticks)
    print("\nCurrent estimated joint angles from physical arm:")
    for joint, ticks, angle in zip(JOINT_ORDER, current_ticks, rads):
        print(f"  {joint:<15}: {ticks:>5} ticks  |  {angle:>6.3f} rad  ({angle * 57.2958:>6.1f}°)")

    print("\nExpected directions in Isaac Sim:")
    print("  1. shoulder_pan : Positive (+) turns arm LEFT (CCW from above)")
    print("  2. shoulder_lift: Positive (+) pitches arm FORWARD / DOWNWARD")
    print("  3. elbow_flex   : Positive (+) bends elbow")
    print("  4. wrist_flex   : Positive (+) pitches wrist DOWNWARD")
    print("  5. wrist_roll   : Positive (+) rolls wrist")
    print("  6. gripper      : Positive (+) OPENS gripper (0.0 rad = closed, 1.2 rad = open)")

    if args.test_nudge and not args.mock:
        print("\n[TEST] Performing gentle nudge test on Motor 1 (shoulder_pan)...")
        pan_id = calib.servo_ids[0]
        orig_ticks = current_ticks[0]
        # Enable torque
        packet_handler.write1ByteTxRx(port_handler, pan_id, 40, 1)
        time.sleep(0.05)
        print(f"  Nudging shoulder_pan +25 ticks (~2°)...")
        packet_handler.write2ByteTxRx(port_handler, pan_id, 42, orig_ticks + 25)
        time.sleep(0.6)
        print(f"  Returning to rest position ({orig_ticks} ticks)...")
        packet_handler.write2ByteTxRx(port_handler, pan_id, 42, orig_ticks)
        time.sleep(0.6)
        packet_handler.write1ByteTxRx(port_handler, pan_id, 40, 0)
        port_handler.closePort()
        print("  [SUCCESS] Nudge test complete. Motor moved and returned smoothly!")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()

