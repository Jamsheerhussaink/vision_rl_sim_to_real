"""Exercise the motor worker without a simulator, serial device, or motor writes."""
import time
import unittest

from isaac_vision_rl.utils.calibration import FollowerArmCalibration
from isaac_vision_rl.utils.evaluation_mirror import MirrorWorker, checked_ticks, mirror_reference
from isaac_vision_rl.utils.motor_limits import GRIPPER_MIN_CLOSE_RAD


class FakeBus:
    def __init__(self, ticks):
        self.current = list(ticks)
        self.writes = []
        self.closed = False
        self.fail_write = False
        self.stuck = False
        self.torque_enabled = False
        self.allow_holding = None

    def enable_torque(self, allow_holding=False):
        self.allow_holding = allow_holding
        self.torque_enabled = True
        return list(self.current)

    def read_current_positions(self):
        return list(self.current)

    def sync_write_positions(self, ticks):
        self.writes.append(list(ticks))
        if not self.stuck:
            self.current = list(ticks)
        return not self.fail_write

    def close(self, disable_torque=True):
        if disable_torque:
            self.torque_enabled = False
        self.closed = True


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.calib = FollowerArmCalibration()
        self.q = [0, 0, 0, 0, 0, 0.5]
        self.bus = FakeBus(checked_ticks(self.calib, self.q))

    def make_worker(self, **kwargs):
        return MirrorWorker(self.bus, self.calib, self.q, rate_hz=200, **kwargs)

    def wait_failure(self, worker):
        self.assertTrue(worker.ready.wait(1))
        worker.thread.join(1)
        self.assertFalse(worker.thread.is_alive())
        with self.assertRaises(RuntimeError):
            worker.close()
        self.assertTrue(self.bus.closed)

    def test_invalid_pose_rejected_before_any_write(self):
        for q in ([0] * 5, [float("nan")] * 6, [float("inf")] * 6, [100] * 6):
            with self.assertRaises(ValueError):
                checked_ticks(self.calib, q)
        self.assertEqual(self.bus.writes, [])

    def test_watchdog_stops_even_when_simulation_thread_does_nothing(self):
        worker = self.make_worker(watchdog=0.04)
        worker.start()
        self.wait_failure(worker)
        self.assertIsInstance(worker.error, TimeoutError)

    def test_write_failure_stops_and_closes_bus(self):
        self.bus.fail_write = True
        worker = self.make_worker()
        with self.assertRaises(RuntimeError):
            worker.start()
        self.wait_failure(worker)

    def test_bounded_initial_approach_and_feedback(self):
        self.bus.current[0] -= 40
        worker = self.make_worker(speed=400, watchdog=0.5)
        worker.start()
        feedback = worker.publish(self.q)
        worker.close()
        self.assertTrue(self.bus.closed)
        trajectory = [checked_ticks(self.calib, self.q)[0] - 40] + [q[0] for q in self.bus.writes]
        self.assertLessEqual(max(abs(b - a) for a, b in zip(trajectory, trajectory[1:])), 5)
        self.assertLessEqual(feedback["max_target_error_ticks"], 5)

    def test_stuck_motor_tracking_error_stops(self):
        worker = self.make_worker(speed=10000, tracking_limit=10, watchdog=0.5)
        worker.start()
        self.bus.stuck = True
        q = list(self.q)
        q[0] = 0.15
        worker.publish(q)
        self.wait_failure(worker)
        self.assertIn("tracking error", str(worker.error))

    def test_alignment_timeout_identifies_small_gripper_error(self):
        self.bus.current[5] += 6
        self.bus.stuck = True
        worker = self.make_worker(alignment_timeout=0.03)
        with self.assertRaisesRegex(RuntimeError, "gripper: target=.*error=-6 ticks"):
            worker.start()
        self.wait_failure(worker)
        self.assertEqual(worker.feedback["target_error_ticks"][5], -6)
        self.assertIsInstance(worker.error, TimeoutError)

    def test_twelve_tick_alignment_accepts_observed_offsets_after_settling(self):
        self.bus.current[3] += 10
        self.bus.current[5] += 6
        self.bus.stuck = True
        worker = self.make_worker(alignment_tolerance=12, watchdog=0.5)
        started = time.monotonic()
        worker.start()
        self.assertGreaterEqual(time.monotonic() - started, 0.2)
        self.assertEqual(worker.feedback["max_target_error_ticks"], 10)
        worker.close()
        self.assertTrue(self.bus.closed)

    def test_twelve_tick_alignment_still_rejects_larger_offset(self):
        self.bus.current[3] += 13
        self.bus.stuck = True
        worker = self.make_worker(alignment_tolerance=12, alignment_timeout=0.04)
        with self.assertRaisesRegex(RuntimeError, "required <=12 ticks"):
            worker.start()
        self.wait_failure(worker)

    def test_alignment_tolerance_cannot_bypass_tracking_limit(self):
        with self.assertRaises(ValueError):
            self.make_worker(alignment_tolerance=201)

    def test_gripper_reference_uses_policy_closure_not_contact_angle(self):
        q = [0, 0, 0, 0, 0, 0.8]
        ref = mirror_reference(q, 0.0)
        self.assertEqual(ref[:5], q[:5])
        # A closed command (0.0) is raised to the shared floor so a cube-blocked jaw is never squeezed to overload
        self.assertEqual(ref[5], GRIPPER_MIN_CLOSE_RAD)
        self.assertGreater(checked_ticks(self.calib, ref)[5], int(self.calib.range_mins[5]))
        self.assertEqual(q[5], 0.8)
        with self.assertRaises(ValueError):
            mirror_reference(q, float("nan"))

    def test_gripper_floor_keeps_cube_stall_gap_under_tracking_limit(self):
        # Model 900 mirror run: jaw stalled on the cube at about 2440 ticks while 0.0 rad commanded 2029 ticks
        stalled_ticks = 2440
        old_gap = stalled_ticks - int(checked_ticks(self.calib, [0, 0, 0, 0, 0, 0.0])[5])
        new_gap = stalled_ticks - int(checked_ticks(self.calib, mirror_reference([0] * 6, 0.0))[5])
        self.assertGreater(old_gap, 350)
        self.assertLess(new_gap, 350)
        self.assertGreater(new_gap, 0)  # still squeezes the cube

    def test_home_return_reaches_initial_pose_and_preserves_torque(self):
        worker = self.make_worker(speed=2000, watchdog=1)
        worker.start()
        moved = list(self.q)
        moved[0] = 0.15
        worker.publish(moved)
        time.sleep(0.1)
        self.assertGreater(self.bus.current[0], worker.home_ticks[0])
        worker.return_home()
        worker.close()
        self.assertTrue(worker.home_ready.is_set())
        self.assertEqual(self.bus.current, worker.home_ticks)
        self.assertTrue(self.bus.torque_enabled)
        self.assertEqual(worker.shutdown_state, "torque_left_enabled")

    def test_each_home_return_waits_for_fresh_encoder_settling(self):
        worker = self.make_worker(speed=2000, watchdog=1)
        worker.start()
        worker.return_home()
        moved = list(self.q)
        moved[0] = 0.15
        worker.publish(moved)
        time.sleep(0.1)
        self.assertGreater(self.bus.current[0], worker.home_ticks[0])
        started = time.monotonic()
        worker.return_home()
        self.assertGreaterEqual(time.monotonic() - started, 0.2)
        self.assertEqual(self.bus.current, worker.home_ticks)
        worker.close()

    def test_explicit_release_disables_torque(self):
        worker = self.make_worker(release_on_exit=True)
        worker.start()
        worker.close()
        self.assertFalse(self.bus.torque_enabled)
        self.assertEqual(worker.shutdown_state, "torque_disabled")

    def test_unverified_parking_is_skipped_and_preserves_torque(self):
        parked = []
        self.bus.park_home_smoothly = lambda: parked.append(True)
        worker = self.make_worker(park_on_exit=True, release_on_exit=True)
        worker.start()
        worker.close()
        self.assertEqual(parked, [])
        self.assertTrue(self.bus.torque_enabled)
        self.assertEqual(worker.shutdown_state, "torque_left_enabled")

    def test_fault_overrides_requested_release_and_parking(self):
        parked = []
        self.bus.park_home_smoothly = lambda: parked.append(True)
        worker = self.make_worker(watchdog=0.03, release_on_exit=True, park_on_exit=True)
        worker.start()
        self.wait_failure(worker)
        self.assertEqual(parked, [])
        self.assertTrue(self.bus.torque_enabled)
        self.assertEqual(worker.shutdown_state, "torque_left_enabled")

    def test_write_fault_overrides_requested_release(self):
        self.bus.fail_write = True
        worker = self.make_worker(release_on_exit=True)
        with self.assertRaises(RuntimeError):
            worker.start()
        self.wait_failure(worker)
        self.assertTrue(self.bus.torque_enabled)

    def test_fault_preserves_power_without_attempting_home(self):
        worker = self.make_worker(watchdog=0.03)
        worker.start()
        self.wait_failure(worker)
        self.assertTrue(self.bus.torque_enabled)
        self.assertFalse(worker.return_requested.is_set())

    def test_interrupted_home_holds_without_releasing(self):
        worker = self.make_worker(speed=100, watchdog=1)
        worker.start()
        moved = list(self.q)
        moved[0] = 0.15
        worker.publish(moved)
        time.sleep(0.1)
        with self.assertRaises(KeyboardInterrupt):
            worker.return_home(cancelled=lambda: True)
        self.assertTrue(self.bus.closed)
        self.assertTrue(self.bus.torque_enabled)

    def test_allow_holding_is_forwarded_to_bus(self):
        worker = self.make_worker(allow_holding=True)
        worker.start()
        worker.close()
        self.assertTrue(self.bus.allow_holding)

    def test_return_home_timeout_still_holds_without_release(self):
        worker = self.make_worker(speed=2000, watchdog=1, alignment_timeout=0.5)
        worker.start()
        moved = list(self.q)
        moved[0] = 0.15
        worker.publish(moved)
        time.sleep(0.1)
        self.bus.stuck = True
        with self.assertRaisesRegex(RuntimeError, "home pose"):
            worker.return_home()
        self.wait_failure(worker)
        self.assertTrue(self.bus.torque_enabled)
        self.assertFalse(worker.home_ready.is_set())

    def test_gripper_has_separate_speed_limit(self):
        worker = self.make_worker(speed=100, gripper_speed=1600, watchdog=1)
        worker.start()
        target = list(self.q)
        target[0] = 0.5
        target[5] = 0.0
        worker.publish(target)
        time.sleep(0.1)
        worker.close()
        arm_delta = abs(self.bus.current[0] - worker.home_ticks[0])
        jaw_delta = abs(self.bus.current[5] - worker.home_ticks[5])
        self.assertGreater(jaw_delta, arm_delta * 4)
        self.assertLess(jaw_delta, 300)

    def test_default_gripper_speed_tracks_a_slower_fake_actuator(self):
        original = self.bus.sync_write_positions
        previous = [time.monotonic()]

        def slow_jaw(ticks):
            now = time.monotonic()
            dt = now - previous[0]
            previous[0] = now
            actual = self.bus.current[5]
            delta = max(-700 * dt, min(700 * dt, ticks[5] - actual))
            result = original(ticks)
            self.bus.current[5] = actual + delta
            return result

        self.bus.sync_write_positions = slow_jaw
        worker = self.make_worker(watchdog=1)
        worker.start()
        opened = list(self.q)
        opened[5] = 1.2
        worker.publish(opened)
        time.sleep(0.3)
        worker.check()
        worker.close()
        self.assertEqual(worker.speeds[5], 600)
        self.assertEqual(worker.tracking_limits[5], worker.tracking_limit)
        self.assertTrue(self.bus.torque_enabled)

    def test_bad_read_stops(self):
        self.bus.read_current_positions = lambda: None
        worker = self.make_worker()
        with self.assertRaises(RuntimeError):
            worker.start()
        self.wait_failure(worker)

    def test_rejected_startup_does_not_disable_existing_arm(self):
        def refuse():
            raise RuntimeError("Motor already powered")
        self.bus.enable_torque = refuse
        worker = self.make_worker()
        with self.assertRaises(RuntimeError):
            worker.start()
        worker.thread.join(1)
    def test_wrist_roll_limits_strictly_enforced(self):
        # [-2.70, +1.48] strictly enforced
        q_valid_max = [0, 0, 0, 0, 1.48, 0.5]
        q_valid_min = [0, 0, 0, 0, -2.70, 0.5]
        self.assertEqual(len(checked_ticks(self.calib, q_valid_max)), 6)
        self.assertEqual(len(checked_ticks(self.calib, q_valid_min)), 6)

        q_invalid_high = [0, 0, 0, 0, 1.49, 0.5]
        with self.assertRaisesRegex(ValueError, "wrist_roll"):
            checked_ticks(self.calib, q_invalid_high)

        q_invalid_low = [0, 0, 0, 0, -2.71, 0.5]
        with self.assertRaisesRegex(ValueError, "wrist_roll"):
            checked_ticks(self.calib, q_invalid_low)

    def test_wrist_roll_uses_direct_difference(self):
        from isaac_vision_rl.utils.evaluation_mirror import _joint_diff
        diff = _joint_diff([0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 4093, 0])
        self.assertEqual(diff[4], -4088.0)
        diff_rev = _joint_diff([0, 0, 0, 0, 4093, 0], [0, 0, 0, 0, 5, 0])
        self.assertEqual(diff_rev[4], 4088.0)


if __name__ == "__main__":
    unittest.main()
