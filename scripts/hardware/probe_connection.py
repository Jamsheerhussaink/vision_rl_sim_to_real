#!/usr/bin/env python3
"""Read all six SO-101 encoders repeatedly. Never write targets or torque registers."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hardware_bridge import FeetechServoBus
from isaac_vision_rl.utils.calibration import CALIB_FILE_PATH, JOINT_ORDER, FollowerArmCalibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--calib", default=CALIB_FILE_PATH)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    calib = FollowerArmCalibration(args.calib)
    bus = FeetechServoBus(args.port, args.baud, calib)
    failures = 0
    try:
        for sample in range(args.samples):
            readings = {}
            for name, motor in zip(JOINT_ORDER, calib.servo_ids):
                try:
                    # retries=1 reveals intermittent failures instead of hiding them.
                    readings[name] = {
                        "position": bus._read_checked(motor, 56, retries=1),
                        "goal_position": bus._read_checked(motor, 42, retries=1),
                        "torque": bus._read_checked(motor, 40, 1, retries=1),
                        "fault": bus._read_checked(motor, 65, 1, retries=1),
                    }
                except IOError as error:
                    failures += 1
                    readings[name] = {"error": str(error)}
            print(json.dumps({"sample": sample + 1, "motors": readings}), flush=True)
            time.sleep(0.1)
    finally:
        # bus.close() disables torque; this diagnostic must leave torque untouched.
        bus.port_handler.closePort()
        bus.connected = False
    print(f"Read-only probe complete: {failures} failed motor samples; no motor registers written.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
