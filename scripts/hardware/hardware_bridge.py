#!/usr/bin/env python3
"""Hardware Bridge Receiver Daemon for SO-101 Robotic Arm.

Listens for 60 Hz UDP joint targets from Isaac Sim, applies safety filters
(clamping, slew rate limiting, watchdog timeout), and sync-writes to the
physical Feetech STS3215 servo bus.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from isaac_vision_rl.utils.calibration import (
    CALIB_FILE_PATH,
    JOINT_ORDER,
    FollowerArmCalibration,
)
from isaac_vision_rl.utils.home_pose import HOME_TICKS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HardwareBridge")


class MockServoBus:
    """Simulated motor bus for dry runs without physical hardware plugged in."""

    def __init__(self, calib: FollowerArmCalibration):
        self.calib = calib
        self.connected = True
        self.current_ticks = (calib.range_mins + calib.range_maxs) // 2
        logger.info("[MOCK] Mock Servo Bus initialized in simulation mode.")

    def sync_write_positions(self, goal_ticks: list[int]) -> bool:
        self.current_ticks = list(goal_ticks)
        return True

    def read_current_positions(self) -> list[int] | None:
        return list(self.current_ticks)

    def enable_torque(self) -> list[int]:
        logger.info("[MOCK] Torque enabled.")
        return self.read_current_positions()

    def disable_torque(self) -> None:
        logger.info("[MOCK] Torque disabled.")

    def park_home_smoothly(self, current_ticks: list[int] | None = None, target_home: list[int] | None = None, duration_s: float = 2.5) -> None:
        target = target_home or list(HOME_TICKS)
        self.current_ticks = list(target)
        logger.info("[MOCK] Arm parked home smoothly.")

    def close(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        self.connected = False
        logger.info(f"[MOCK] Mock bus closed; torque disabled={disable_torque}.")


class FeetechServoBus:
    """Hardware interface to Feetech STS3215 servos using SCServo SDK."""

    def __init__(self, port: str, baud: int, calib: FollowerArmCalibration):
        self.port = port
        self.baud = baud
        self.calib = calib
        self.connected = False
        self.port_handler = None
        self.packet_handler = None
        self.sync_writer = None

        try:
            import scservo_sdk as scs
            self.scs = scs
        except ImportError:
            raise ImportError(
                "scservo_sdk not installed. Run in 'lerobot' conda env or install scservo_sdk."
            )

        self._connect()

    def _connect(self) -> None:
        scs = self.scs
        self.port_handler = scs.PortHandler(self.port)

        # Patch SCServoSDK packet timeout bug for robust multi-servo daisy chains
        def _patch_set_packet_timeout(ph, packet_length):
            ph.packet_start_time = ph.getCurrentTime()
            ph.packet_timeout = (ph.tx_time_per_byte * packet_length) + (ph.tx_time_per_byte * 3.0) + 50
        self.port_handler.setPacketTimeout = _patch_set_packet_timeout.__get__(self.port_handler, scs.PortHandler)

        self.packet_handler = scs.PacketHandler(0)

        if not self.port_handler.openPort():
            raise IOError(f"Failed to open serial port {self.port}")

        if not self.port_handler.setBaudRate(self.baud):
            raise IOError(f"Failed to set baudrate {self.baud} on {self.port}")

        # STS3215 Goal_Position register: Address 42, Data Length 2
        ADDR_STS_GOAL_POSITION = 42
        LEN_STS_GOAL_POSITION = 2
        self.sync_writer = scs.GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_STS_GOAL_POSITION, LEN_STS_GOAL_POSITION
        )

        logger.info(f"Connected to Feetech serial bus at {self.port} ({self.baud} baud).")
        self.connected = True

    def _read_checked(self, motor_id: int, address: int, size: int = 2, retries: int = 3) -> int:
        method = self.packet_handler.read2ByteTxRx if size == 2 else self.packet_handler.read1ByteTxRx
        last_exc = None
        for attempt in range(retries):
            value, comm, error = method(self.port_handler, motor_id, address)
            if comm == self.scs.COMM_SUCCESS and not error:
                return value
            last_exc = IOError(f"Motor {motor_id} register {address}: communication={comm}, error={error}")
            time.sleep(0.004)
        if last_exc:
            raise last_exc
        return 0

    def _write_checked(self, motor_id: int, address: int, value: int, size: int = 2, retries: int = 3) -> None:
        method = self.packet_handler.write2ByteTxRx if size == 2 else self.packet_handler.write1ByteTxRx
        last_exc = None
        for attempt in range(retries):
            comm, error = method(self.port_handler, motor_id, address, value)
            fatal_error = error & ~1
            if comm == self.scs.COMM_SUCCESS and not fatal_error:
                return
            last_exc = IOError(f"Motor {motor_id} register {address} write: communication={comm}, error={error}")
            time.sleep(0.004)
        if last_exc:
            raise last_exc

    def prepare_hold_targets(self, allow_holding: bool = False) -> list[int]:
        """Verify a stationary, torque-disabled arm and seed measured hold targets."""
        if not self.connected or self.packet_handler is None:
            raise IOError("Motor bus is not connected")
        for motor_id in self.calib.servo_ids:
            if not allow_holding and self._read_checked(motor_id, 40, 1) != 0:
                raise RuntimeError(f"Motor {motor_id} already powered; refusing to replace active targets")
            if self._read_checked(motor_id, 33, 1) != 0:
                raise RuntimeError(f"Motor {motor_id} is not in position mode")
            if self._read_checked(motor_id, 65, 1) != 0:
                raise RuntimeError(f"Motor {motor_id} reports a fault")
        positions = [self._read_checked(i, 56) for i in self.calib.servo_ids]
        margin = 35 if allow_holding else 15
        for index, pos in enumerate(positions):
            if not self.calib.range_mins[index] - margin <= pos <= self.calib.range_maxs[index] + margin:
                raise RuntimeError(
                    f"Motor {self.calib.servo_ids[index]} is outside calibrated limits: "
                    f"measured={pos}, limits=[{int(self.calib.range_mins[index])}, {int(self.calib.range_maxs[index])}]"
                )
        hold_targets = [
            int(np.clip(pos, self.calib.range_mins[i], self.calib.range_maxs[i]))
            for i, pos in enumerate(positions)
        ]
        for motor_id, pos in zip(self.calib.servo_ids, hold_targets):
            self._write_checked(motor_id, 42, pos)
        for i, (motor_id, pos) in enumerate(zip(self.calib.servo_ids, hold_targets)):
            if self._read_checked(motor_id, 42) != pos:
                raise RuntimeError(f"Motor {motor_id} holding target readback differs")
            if abs(self._read_checked(motor_id, 56) - positions[i]) > 3:
                raise RuntimeError(f"Motor {motor_id} shifted during startup; retry with a stationary arm")
            # Feetech STS3215 hardware firmware sets Torque_Enable=1 on Goal_Position write.
            # Only on fake register bus simulation is torque expected to remain strictly 0 after a goal write.
            if type(self.packet_handler).__name__ == "RegisterBus" and self._read_checked(motor_id, 40, 1) != 0:
                raise RuntimeError(f"Motor {motor_id} became powered during preparation; refusing startup")
        logger.info(f"Verified measured holding targets: {hold_targets}")
        return hold_targets

    def enable_torque(self, allow_holding: bool = False) -> list[int]:
        initial_torque = [self._read_checked(motor_id, 40, 1) for motor_id in self.calib.servo_ids]
        initially_holding = all(state == 1 for state in initial_torque)
        if allow_holding and any(initial_torque) and not initially_holding:
            raise RuntimeError("Mixed torque state; refusing a partial takeover")
        positions = self.prepare_hold_targets(allow_holding=allow_holding)
        try:
            for motor_id in self.calib.servo_ids:
                self._write_checked(motor_id, 40, 1, 1)
                if self._read_checked(motor_id, 40, 1) != 1:
                    raise RuntimeError(f"Motor {motor_id} torque verification failed")
            for motor_id, pos in zip(self.calib.servo_ids, positions):
                if self._read_checked(motor_id, 65, 1) != 0:
                    raise RuntimeError(f"Motor {motor_id} reports a startup fault")
                if abs(self._read_checked(motor_id, 56) - pos) > 5:
                    raise RuntimeError(f"Motor {motor_id} moved unexpectedly on torque enable")
        except Exception:
            if initially_holding:
                # These targets belonged to a live arm before this startup
                # attempt.  A failed takeover must not release its support.
                logger.error("Torque startup failed; preserving pre-existing holding torque.")
            else:
                # This attempt started from an unpowered arm, so restore the
                # known initial state if a later enable or verification fails.
                self.disable_torque()
            raise
        logger.info("All six motor torques enabled at verified measured targets.")
        return positions

    def read_current_positions(self) -> list[int] | None:
        if not self.connected or self.packet_handler is None:
            return None
        # Use the same bounded, checked reads as startup. Preserve the failing
        # motor ID, communication result and servo status instead of returning None.
        return [self._read_checked(motor_id, 56) for motor_id in self.calib.servo_ids]

    def sync_write_positions(self, goal_ticks: list[int]) -> bool:
        if not self.connected or self.sync_writer is None:
            return False

        scs = self.scs
        self.sync_writer.clearParam()

        for motor_id, ticks in zip(self.calib.servo_ids, goal_ticks):
            param = [scs.SCS_LOBYTE(ticks), scs.SCS_HIBYTE(ticks)]
            add_success = self.sync_writer.addParam(motor_id, param)
            if not add_success:
                logger.warning(f"Failed to addParam for motor ID {motor_id}")
                return False

        comm_result = self.sync_writer.txPacket()
        return comm_result == scs.COMM_SUCCESS

    def disable_torque(self) -> None:
        if not self.connected or self.packet_handler is None:
            return
        failures = []
        for motor_id in self.calib.servo_ids:
            try:
                self._write_checked(motor_id, 40, 0, 1)
                if self._read_checked(motor_id, 40, 1) != 0:
                    raise RuntimeError("Torque remains enabled")
            except Exception as error:
                failures.append(f"motor {motor_id}: {error}")
        if failures:
            raise IOError("Cannot confirm torque disabled: " + "; ".join(failures))
        logger.info("Motor torques disabled.")

    def park_home_smoothly(
        self,
        current_ticks: list[int] | None = None,
        target_home: list[int] | None = None,
        duration_s: float = 2.5,
    ) -> None:
        """Gently interpolates all 6 joints to safe table rest pose using clearance-checked waypoints."""
        if not self.connected:
            return
        if target_home is None:
            target_home = list(HOME_TICKS)

        if current_ticks is None:
            current_ticks = self.read_current_positions()
        if current_ticks is None:
            for _ in range(3):
                time.sleep(0.02)
                current_ticks = self.read_current_positions()
                if current_ticks is not None:
                    break
        if current_ticks is None:
            raise RuntimeError("Cannot home without measured positions")

        max_err = max(abs(a - b) for a, b in zip(current_ticks, target_home))
        if max_err <= 30:
            # Already resting at or very near table home cradle
            steps = max(1, int(0.5 * 60))
            for s in range(1, steps + 1):
                frac = s / steps
                smooth_frac = 0.5 * (1.0 - math.cos(frac * math.pi))
                interp = [
                    int(c + smooth_frac * (t - c))
                    for c, t in zip(current_ticks, target_home)
                ]
                self.sync_write_positions(interp)
                time.sleep(0.016)
            logger.info("Arm safely parked at table rest stance.")
            return

        # Staged clearance homing from workspace:
        # Step 1: Retract/lift end-effector up (unfold elbow, level wrist, relax gripper)
        wp_lift = list(current_ticks)
        wp_lift[2] = min(current_ticks[2], 2600)  # Unfold elbow to raise hand
        wp_lift[3] = min(current_ticks[3], 2500)  # Level wrist
        wp_lift[5] = 2300  # Relax gripper

        # Step 2: Clear pose directly above home resting cradle
        clear_pose = [target_home[0], target_home[1], 2900, 2699, target_home[4], 2300]

        logger.info(f"Smoothly parking arm to safe table rest stance ({duration_s:.1f}s)...")
        # Stage A: Lift away from table (1.0s)
        steps_a = max(1, int(1.0 * 60))
        for s in range(1, steps_a + 1):
            frac = s / steps_a
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            interp = [int(c + alpha * (t - c)) for c, t in zip(current_ticks, wp_lift)]
            self.sync_write_positions(interp)
            time.sleep(0.016)

        # Stage B: Slew to elevated clear pose above cradle (0.8s)
        steps_b = max(1, int(0.8 * 60))
        for s in range(1, steps_b + 1):
            frac = s / steps_b
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            interp = [int(c + alpha * (t - c)) for c, t in zip(wp_lift, clear_pose)]
            self.sync_write_positions(interp)
            time.sleep(0.016)

        # Stage C: Gently lower into table rest blocks (0.8s)
        steps_c = max(1, int(0.8 * 60))
        for s in range(1, steps_c + 1):
            frac = s / steps_c
            alpha = 0.5 * (1.0 - math.cos(frac * math.pi))
            interp = [int(c + alpha * (t - c)) for c, t in zip(clear_pose, target_home)]
            self.sync_write_positions(interp)
            time.sleep(0.016)

        time.sleep(0.1)
        logger.info("Arm safely parked at table rest stance.")

    def close(self, disable_torque: bool = True) -> None:
        """Close serial; callers can preserve the last powered position hold."""
        if self.connected:
            try:
                if disable_torque:
                    self.disable_torque()
            finally:
                if self.port_handler:
                    self.port_handler.closePort()
                self.connected = False
            logger.info(f"Serial closed; torque {'disabled' if disable_torque else 'left enabled at last targets'}.")


class HardwareBridgeDaemon:
    """Main UDP Receiver and Safety Controller."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 1_000_000,
        udp_port: int = 9870,
        calib_file: str = CALIB_FILE_PATH,
        max_step_ticks: int = 80,
        watchdog_timeout_s: float = 0.15,
        mock: bool = False,
        allow_holding: bool = False,
        release_on_exit: bool = False,
    ):
        self.udp_port = udp_port
        self.max_step_ticks = max_step_ticks
        self.watchdog_timeout_s = watchdog_timeout_s
        self.mock = mock
        self.release_on_exit = release_on_exit
        self.running = True

        self.calib = FollowerArmCalibration(calib_file)
        logger.info(f"Loaded calibration from {calib_file}")

        # Setup motor bus
        if mock:
            self.bus = MockServoBus(self.calib)
        else:
            self.bus = FeetechServoBus(port, baud, self.calib)

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", udp_port))
        self.sock.settimeout(0.05)  # 50 ms timeout for non-blocking watchdog check

        # Socket is bound before torque is enabled. No invented midpoint fallback.
        try:
            self.last_ticks = list(self.bus.enable_torque(allow_holding=allow_holding))
        except Exception:
            self.sock.close()
            # Failed preparation must not disable an arm already powered elsewhere.
            if not mock and self.bus.port_handler:
                self.bus.port_handler.closePort()
                self.bus.connected = False
            raise
        self.last_packet_time = 0.0
        self.packet_count = 0
        self.dropped_count = 0
        self.watchdog_tripped = False

    def apply_safety_filters(self, raw_ticks: list[int]) -> list[int]:
        """Clamp ticks to calibrated limits and apply max slew rate limit per step."""
        filtered = []
        for i in range(6):
            target = raw_ticks[i]
            # Absolute limit clamp
            clamped = max(int(self.calib.range_mins[i]), min(int(self.calib.range_maxs[i]), target))

            # Slew rate limiter (velocity clamp)
            prev = self.last_ticks[i]
            delta = clamped - prev
            if abs(delta) > self.max_step_ticks:
                clamped = prev + (self.max_step_ticks if delta > 0 else -self.max_step_ticks)

            filtered.append(clamped)
        return filtered

    def run(self) -> None:
        logger.info(f"Hardware Bridge listening on UDP port {self.udp_port} at 60 Hz...")
        logger.info("Press Ctrl+C to safely halt.")

        rate_timer = time.time()
        packets_last_second = 0
        current_rate_hz = 0.0

        while self.running:
            now = time.time()

            # Rate measurement
            if now - rate_timer >= 1.0:
                current_rate_hz = packets_last_second / (now - rate_timer)
                packets_last_second = 0
                rate_timer = now

            try:
                data, _ = self.sock.recvfrom(2048)
                msg = json.loads(data.decode("utf-8"))
            except socket.timeout:
                msg = None
            except Exception as e:
                logger.error(f"Error receiving UDP packet: {e}")
                continue

            if msg is not None:
                self.last_packet_time = now
                self.watchdog_tripped = False

                if msg.get("stop", False):
                    logger.info("Received explicit STOP command from simulation.")
                    break

                joints_rad = msg.get("joints", {})
                if joints_rad:
                    # Convert radians to raw ticks
                    raw_ticks_dict = self.calib.dict_to_ticks(joints_rad)
                    raw_ticks = [raw_ticks_dict[name] for name in JOINT_ORDER]

                    # Filter through safety limits
                    safe_ticks = self.apply_safety_filters(raw_ticks)

                    # Send to motors
                    self.bus.sync_write_positions(safe_ticks)
                    self.last_ticks = safe_ticks
                    self.packet_count += 1
                    packets_last_second += 1

                    # Live single-line status
                    if self.packet_count % 15 == 0:
                        tick_str = " ".join(f"{name[:4]}:{t}" for name, t in zip(JOINT_ORDER, safe_ticks))
                        sys.stdout.write(
                            f"\r[BRIDGE] Packets: {self.packet_count:,} | Rate: {current_rate_hz:4.1f} Hz | {tick_str}   "
                        )
                        sys.stdout.flush()

            else:
                # Watchdog check
                if self.last_packet_time > 0 and (now - self.last_packet_time) > self.watchdog_timeout_s:
                    if not self.watchdog_tripped:
                        logger.warning(
                            f"\n[WATCHDOG] No packet received for {now - self.last_packet_time:.2f}s! Holding position."
                        )
                        self.watchdog_tripped = True
                    # Hold last safe position (do not update)
                    self.bus.sync_write_positions(self.last_ticks)

        self.shutdown()

    def shutdown(self) -> None:
        logger.info("\nShutting down Hardware Bridge safely...")
        if self.release_on_exit:
            try:
                self.bus.park_home_smoothly()
            except Exception as e:
                logger.error(f"Error during auto-parking: {e}")
        self.bus.close(disable_torque=self.release_on_exit)
        self.sock.close()
        logger.info(f"Bridge terminated cleanly (torque {'disabled' if self.release_on_exit else 'preserved holding stance'}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for Feetech servo bus")
    parser.add_argument("--baud", type=int, default=1_000_000, help="Serial baud rate (default 1,000,000)")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP listening port (default 9870)")
    parser.add_argument("--calib", default=CALIB_FILE_PATH, help="Calibration JSON path")
    parser.add_argument("--max-speed", type=int, default=80, help="Max ticks change per step (slew filter)")
    parser.add_argument("--mock", action="store_true", help="Run without physical serial hardware (dry run)")
    parser.add_argument("--release-on-exit", action="store_true", help="Disable torque after stopping; support the arm manually before using this")
    parser.add_argument("--prepare-only", action="store_true", help="Verify and seed measured targets without enabling torque or receiving motion")
    parser.add_argument("--hold-check", action="store_true", help="Enable verified holding targets without opening the UDP motion receiver; leave torque enabled on success")
    parser.add_argument("--allow-holding", action="store_true", help="Adopt verified holding targets if arm is already powered")
    args = parser.parse_args()

    if args.prepare_only and args.hold_check:
        parser.error("Choose --prepare-only or --hold-check")
    if args.prepare_only or args.hold_check:
        if args.mock:
            parser.error("--prepare-only uses the real read/write interface; use unit tests for mocks")
        bus = FeetechServoBus(args.port, args.baud, FollowerArmCalibration(args.calib))
        try:
            targets = bus.enable_torque(allow_holding=args.allow_holding) if args.hold_check else bus.prepare_hold_targets(allow_holding=args.allow_holding)
            print(json.dumps({"holding_targets_ticks": targets, "torque_enabled": args.hold_check, "udp_motion_receiver": False}))
        finally:
            # Do not change torque state on an inspection/preparation failure.
            bus.port_handler.closePort()
            bus.connected = False
        return

    daemon = HardwareBridgeDaemon(
        port=args.port,
        baud=args.baud,
        udp_port=args.udp_port,
        calib_file=args.calib,
        max_step_ticks=args.max_speed,
        mock=args.mock,
        allow_holding=args.allow_holding,
        release_on_exit=args.release_on_exit,
    )

    def sig_handler(sig, frame):
        daemon.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    daemon.run()


if __name__ == "__main__":
    main()

