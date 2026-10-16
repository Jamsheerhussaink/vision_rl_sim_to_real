"""Hardware worker for copying simulated joint positions, without simulator imports."""
from __future__ import annotations

import threading
import time

import numpy as np

from isaac_vision_rl.utils.calibration import JOINT_ORDER, FollowerArmCalibration


def _joint_diff(a, b):
    """Compute a physical encoder difference without crossing a wrist limit.

    The wrist encoder is 12-bit, but the calibrated wrist interval is not a
    continuous rotary joint.  Wrapping its error across 0/4095 can command a
    path outside that physical interval, so every calibrated joint uses its
    direct tick difference.
    """
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def checked_ticks(calib: FollowerArmCalibration, angles) -> list[int]:
    """Reject invalid/unreachable poses instead of silently clipping the trajectory."""
    q = np.asarray(angles, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected six finite joint angles in JOINT_ORDER")
    lower, upper = calib.get_joint_limits_rad()
    if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
        bad = [name for i, name in enumerate(JOINT_ORDER) if q[i] < lower[i] - 1e-6 or q[i] > upper[i] + 1e-6]
        raise ValueError(f"Simulated pose exceeds hardware calibration: {bad}")
    return calib.rad_to_ticks(q).tolist()


def mirror_reference(sim_angles, gripper_target=None, calib=None):
    """Copy measured arm angles but use the processed jaw command, not contact deflection."""
    q = list(sim_angles)
    if len(q) != 6:
        raise ValueError("Expected six simulated joint angles")
    if calib is not None:
        lower, upper = calib.get_joint_limits_rad()
        q = np.clip(q, lower, upper).tolist()
    if gripper_target is not None:
        if not np.isfinite(gripper_target):
            raise ValueError("Gripper reference must be finite")
        q[5] = float(np.clip(gripper_target, 0.0, 1.2))
    return q


class MirrorWorker:
    """Own the serial bus in one thread; stop on stale targets or bad feedback.

    Startup approaches the first simulated pose at a bounded speed. After startup,
    target age is measured independently of the simulation thread. Shutdown uses
    a serial-only close by default, preserving torque at the last sent targets.
    Normal completion explicitly requests a bounded return to the initial pose.
    """

    def __init__(self, bus, calib, initial_angles, *, rate_hz=60.0, speed=400.0,
                 watchdog=0.25, tracking_limit=200, alignment_timeout=30.0, alignment_tolerance=5,
                 gripper_speed=600.0, gripper_tracking_limit=None, allow_holding=False, release_on_exit=False, park_on_exit=False,
                 close_gripper_on_home=False):
        if not all(np.isfinite(v) and v > 0 for v in
                   (rate_hz, speed, watchdog, tracking_limit, alignment_timeout, alignment_tolerance, gripper_speed)):
            raise ValueError("Worker rates, limits and timeouts must be finite and positive")
        self.bus, self.calib = bus, calib
        self.period, self.speed = 1 / rate_hz, speed
        self.speeds = np.asarray([speed] * 5 + [gripper_speed], dtype=float)
        self.allow_holding, self.release_on_exit, self.park_on_exit = allow_holding, release_on_exit, park_on_exit
        self.close_gripper_on_home = close_gripper_on_home
        self.shutdown_state = "not_started"
        self.watchdog, self.tracking_limit = watchdog, tracking_limit
        if gripper_tracking_limit is None:
            # Gripper contact is not a reason to ignore its complete encoder
            # error.  Callers that have a validated object-contact envelope may
            # explicitly choose a wider value.
            gripper_tracking_limit = tracking_limit
        self.tracking_limits = np.asarray([tracking_limit] * 5 + [gripper_tracking_limit], dtype=float)
        if alignment_tolerance > tracking_limit:
            raise ValueError("Alignment tolerance cannot exceed the runtime tracking limit")
        self.alignment_tolerance = alignment_tolerance
        self.alignment_timeout = alignment_timeout
        self.target = checked_ticks(calib, initial_angles)
        self.home_ticks = list(self.target)
        self.return_requested = threading.Event()
        self.home_ready = threading.Event()
        self.return_deadline = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.error = None
        self.updated = time.monotonic()
        self.feedback = {}
        self.thread = threading.Thread(target=self._run, name="so101-mirror", daemon=True)

    def start(self, cancelled=lambda: False):
        self.thread.start()
        while not self.ready.wait(0.05):
            if cancelled():
                self.close()
                raise KeyboardInterrupt("Interrupted during initial pose alignment")
            self.check()
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Motor mirror stopped: {self.error}") from self.error

    def publish(self, angles):
        ticks = checked_ticks(self.calib, angles)
        self.check()
        with self.lock:
            if self.return_requested.is_set():
                raise RuntimeError("Cannot publish policy targets while returning home")
            self.target, self.updated = ticks, time.monotonic()
            return dict(self.feedback)

    def return_home(self, cancelled=lambda: False, close_gripper=None):
        """Ask the bus-owning thread to approach home, with feedback and a timeout."""
        self.check()
        if not self.ready.is_set() or not self.thread.is_alive():
            raise RuntimeError("Cannot return home before successful startup or after shutdown")
        should_close = self.close_gripper_on_home if close_gripper is None else bool(close_gripper)
        with self.lock:
            # A previous successful return leaves this event set.  Clear it
            # before requesting a new return so this call waits for fresh
            # encoder settling at the current home target.
            self.home_ready.clear()
            self.target = list(self.home_ticks)
            if should_close:
                self.target[5] = int(self.calib.range_mins[5])
            self.return_deadline = time.monotonic() + self.alignment_timeout
            self.return_requested.set()
        while not self.home_ready.wait(0.05):
            if cancelled():
                self.close()
                raise KeyboardInterrupt("Home return interrupted; holding last commanded pose")
            self.check()
        self.check()
        with self.lock:
            self.return_requested.clear()
            self.updated = time.monotonic()

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Serial worker has not stopped; use the physical power cutoff")
        self.check()

    def _run(self):
        owns_torque = False
        try:
            # Existing bus refuses takeover and seeds targets from measured encoders.
            current = (self.bus.enable_torque(allow_holding=True) if self.allow_holding
                       else self.bus.enable_torque())
            owns_torque = True
            self.shutdown_state = "powered"
            command = np.asarray(current, dtype=float)
            alignment_deadline = time.monotonic() + self.alignment_timeout
            previous = time.monotonic()
            next_alignment_log = previous
            settled_since = None
            was_returning = False
            while not self.stop_event.is_set():
                now = time.monotonic()
                dt = min(now - previous, self.period * 2)
                previous = now
                with self.lock:
                    target, updated = np.asarray(self.target), self.updated
                    returning = self.return_requested.is_set()
                if returning and not was_returning:
                    settled_since = None
                    next_alignment_log = now
                    was_returning = True
                elif not returning and was_returning:
                    settled_since = None
                    was_returning = False
                if self.ready.is_set() and not returning and now - updated > self.watchdog:
                    raise TimeoutError("Simulation joint targets became stale")
                measured = self.bus.read_current_positions()
                if measured is None or len(measured) != 6:
                    raise IOError("Cannot read all six motor positions")
                measured = np.asarray(measured)
                if np.any(measured < self.calib.range_mins - 15) or np.any(measured > self.calib.range_maxs + 15):
                    details = "; ".join(
                        f"{name}: measured={int(m)} (limits [{int(lo)}, {int(hi)}])"
                        for name, m, lo, hi in zip(JOINT_ORDER, measured, self.calib.range_mins, self.calib.range_maxs)
                        if m < lo - 15 or m > hi + 15
                    )
                    raise RuntimeError(f"Measured motor position outside calibration: {details}")
                # Compare against the integer goal actually sent, not its fractional accumulator.
                previous_sent = np.rint(command).astype(int)
                diff_sent = _joint_diff(measured, previous_sent)
                errors = np.abs(diff_sent)
                if np.any(errors > self.tracking_limits):
                    details = "; ".join(
                        f"{name}: command={int(c)}, measured={int(m)}, error={int(err)} ticks"
                        for name, c, m, lim, err in zip(JOINT_ORDER, previous_sent, measured, self.tracking_limits, diff_sent)
                        if abs(err) > lim
                    )
                    raise RuntimeError(f"Motor tracking error exceeded configured tick limit: {details}")
                if hasattr(self.bus, "_read_checked"):
                    for motor in self.calib.servo_ids:
                        if self.bus._read_checked(motor, 65, 1):
                            raise RuntimeError(f"Motor {motor} reports a fault")
                delta_cmd = _joint_diff(target, command)
                command += np.clip(delta_cmd, -self.speeds * dt, self.speeds * dt)
                sent = np.rint(command).astype(int).tolist()
                if not self.bus.sync_write_positions(sent):
                    raise IOError("Motor position write failed")
                t_err = _joint_diff(target, measured)
                with self.lock:
                    self.feedback = {"measured_monotonic_s": now, "command_ticks": sent, "measured_ticks": measured.tolist(),
                                     "target_ticks": target.tolist(),
                                     "max_target_error_ticks": int(np.max(np.abs(t_err))),
                                     "target_error_ticks": t_err.tolist()}
                if not self.ready.is_set() or (returning and not self.home_ready.is_set()):
                    details = "; ".join(
                        f"{name}: target={int(t)}, measured={int(m)}, error={int(err)} ticks"
                        for name, t, m, err in zip(JOINT_ORDER, target, measured, t_err)
                    )
                    if now >= next_alignment_log:
                        print(f"[{'HOME' if returning else 'ALIGN'}] {details}", flush=True)
                        next_alignment_log = now + 1.0
                    within_tolerance = np.max(np.abs(t_err)) <= self.alignment_tolerance
                    c_delta = _joint_diff(target, command)
                    command_arrived = np.max(np.abs(c_delta)) <= 0.5
                    if within_tolerance and command_arrived:
                        if settled_since is None:
                            settled_since = now
                        if now - settled_since >= 0.2:
                            with self.lock:
                                self.updated = time.monotonic()
                            if returning:
                                self.home_ready.set()
                            else:
                                self.ready.set()
                    else:
                        settled_since = None
                    phase_done = self.home_ready.is_set() if returning else self.ready.is_set()
                    phase_deadline = self.return_deadline if returning else alignment_deadline
                    if not phase_done and now > phase_deadline:
                        raise TimeoutError(
                            f"Real arm did not reach the {'home pose' if returning else 'initial simulation pose'} within "
                            f"{self.alignment_timeout:g}s (required <={self.alignment_tolerance:g} ticks "
                            f"per joint for 0.2s after command arrival). {details}"
                        )
                self.stop_event.wait(max(0, self.period - (time.monotonic() - now)))
        except Exception as exc:
            self.error = exc
        finally:
            try:
                if owns_torque:
                    # Fault cleanup must never start a trajectory or remove support.
                    # Legacy parking has no verified-rest result, so it cannot
                    # authorize a torque release.
                    release = self.release_on_exit and self.error is None and not self.park_on_exit
                    if self.park_on_exit:
                        print("[WARN] Automatic parking skipped: recovery route is not verified; preserving torque.", flush=True)
                    self.bus.close(disable_torque=release)
                    self.shutdown_state = "torque_disabled" if release else "torque_left_enabled"
                elif getattr(self.bus, "port_handler", None) is not None:
                    # Failed startup must not disable an arm we refused to adopt.
                    self.bus.port_handler.closePort()
                    self.bus.connected = False
            except Exception as exc:
                self.shutdown_state = "shutdown_unconfirmed"
                self.error = RuntimeError(f"{self.error or 'Shutdown'}; serial/torque shutdown failed: {exc}")
            self.ready.set()
