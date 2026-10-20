#!/usr/bin/env python3
"""Forcefully return robot arm to safe table rest stance and release torque.

Usage:
    python scripts/park_home.py
    conda run -n env_isaaclab python scripts/park_home.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/hardware"))

from hardware_bridge import FeetechServoBus
from isaac_vision_rl.utils.calibration import FollowerArmCalibration, CALIB_FILE_PATH, JOINT_ORDER
from isaac_vision_rl.utils.home_pose import HOME_TICKS


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for Feetech servos")
    parser.add_argument("--calib", default=CALIB_FILE_PATH, help="Calibration JSON file")
    parser.add_argument("--duration", type=float, default=2.5, help="Parking duration in seconds")
    parser.add_argument("--leave-torque", action="store_true", help="Keep torque enabled at resting pose")
    args = parser.parse_args()

    calib = FollowerArmCalibration(args.calib)
    print(f"[INFO] Connecting to robot at {args.port}...", flush=True)
    bus = FeetechServoBus(args.port, 1_000_000, calib)
    measured = bus.read_current_positions()
    if measured is None:
        raise RuntimeError("Cannot read motor positions from serial bus")
    diffs = {name: m - h for name, m, h in zip(JOINT_ORDER, measured, HOME_TICKS)}
    print(f"[INFO] Measured joint ticks: {measured}", flush=True)
    print(f"[INFO] Offset from home rest blocks: {diffs}", flush=True)

    print("[INFO] Enabling torque with holding protection...", flush=True)
    bus.enable_torque(allow_holding=True)

    print(f"[INFO] Smoothly parking arm using 3-stage clearance trajectory ({args.duration:.1f}s)...", flush=True)
    bus.park_home_smoothly(current_ticks=measured, target_home=list(HOME_TICKS), duration_s=args.duration)

    bus.close(disable_torque=not args.leave_torque)
    status = "holding" if args.leave_torque else "unpowered (torque disabled)"
    print(f"[INFO] Robot safely parked on table rest blocks! Stance is {status}.", flush=True)


if __name__ == "__main__":
    main()

