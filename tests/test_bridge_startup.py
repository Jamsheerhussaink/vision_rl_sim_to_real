"""Motor startup ordering and failure behavior using a fake register bus."""
import unittest
from types import SimpleNamespace

from scripts.hardware.hardware_bridge import FeetechServoBus
from isaac_vision_rl.utils.calibration import FollowerArmCalibration


class RegisterBus:
    def __init__(self):
        self.values = {i: {33: 0, 40: 0, 42: 0, 56: p, 65: 0}
                       for i, p in enumerate([2000, 2200, 2050, 1900, 2100, 2040], 1)}
        self.events = []
        self.read_failure = None
        self.bad_goal = False
        self.fail_enable = None
        self.power_during_goals = False

    def read2ByteTxRx(self, port, motor, address):
        self.events.append(("read", motor, address))
        if self.read_failure == (motor, address):
            return 0, -6, 0
        value = self.values[motor][address]
        if self.bad_goal and address == 42:
            value += 1
        return value, 0, 0

    read1ByteTxRx = read2ByteTxRx

    def write2ByteTxRx(self, port, motor, address, value):
        self.events.append(("write", motor, address, value))
        if address == 40 and value == 1 and motor == self.fail_enable:
            return -7, 0
        self.values[motor][address] = value
        if self.power_during_goals and address == 42:
            self.values[motor][40] = 1
        return 0, 0

    write1ByteTxRx = write2ByteTxRx


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.packet = RegisterBus()
        self.bus = FeetechServoBus.__new__(FeetechServoBus)
        self.bus.connected = True
        self.bus.port_handler = object()
        self.bus.packet_handler = self.packet
        self.bus.calib = FollowerArmCalibration()
        self.bus.scs = SimpleNamespace(COMM_SUCCESS=0)

    def test_runtime_position_failure_identifies_motor_and_retries(self):
        self.packet.read_failure = (3, 56)
        with self.assertRaisesRegex(IOError, "Motor 3 register 56: communication=-6"):
            self.bus.read_current_positions()
        self.assertEqual(self.packet.events.count(("read", 3, 56)), 3)
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_runtime_transient_read_recovers_without_writes(self):
        original = self.packet.read2ByteTxRx
        attempts = []

        def transient(port, motor, address):
            if motor == 3 and address == 56:
                attempts.append(1)
                if len(attempts) == 1:
                    return 0, -6, 0
            return original(port, motor, address)

        self.packet.read2ByteTxRx = transient
        self.assertEqual(self.bus.read_current_positions(), [2000, 2200, 2050, 1900, 2100, 2040])
        self.assertEqual(len(attempts), 2)
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_serial_only_close_preserves_torque_and_writes_nothing(self):
        self.packet.values[1][40] = 1
        closed = []
        self.bus.port_handler = SimpleNamespace(closePort=lambda: closed.append(True))
        self.bus.close(disable_torque=False)
        self.assertEqual(closed, [True])
        self.assertFalse(self.bus.connected)
        self.assertEqual(self.packet.values[1][40], 1)
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_default_bus_close_still_disables_for_other_tools(self):
        for values in self.packet.values.values():
            values[40] = 1
        self.bus.port_handler = SimpleNamespace(closePort=lambda: None)
        self.bus.close()
        self.assertTrue(all(v[40] == 0 for v in self.packet.values.values()))
        self.assertFalse(self.bus.connected)

    def test_every_target_verified_before_first_enable(self):
        positions = self.bus.enable_torque()
        events = self.packet.events
        first_enable = next(i for i, e in enumerate(events) if e[0] == "write" and e[2:] == (40, 1))
        for motor, pos in zip(range(1, 7), positions):
            self.assertIn(("write", motor, 42, pos), events[:first_enable])
            self.assertIn(("read", motor, 42), events[:first_enable])
            self.assertEqual(self.packet.values[motor][40], 1)
        self.assertTrue(all(e[2] in (40, 42) for e in events if e[0] == "write"))

    def test_prepare_only_never_writes_torque(self):
        self.bus.prepare_hold_targets()
        self.assertFalse(any(e[0] == "write" and e[2] == 40 for e in self.packet.events))

    def test_failed_position_read_never_enables(self):
        self.packet.read_failure = (3, 56)
        with self.assertRaises(IOError):
            self.bus.enable_torque()
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_goal_mismatch_never_enables(self):
        self.packet.bad_goal = True
        with self.assertRaises(RuntimeError):
            self.bus.enable_torque()
        self.assertFalse(any(e[0] == "write" and e[2] == 40 for e in self.packet.events))

    def test_partial_enable_failure_restores_disabled_state(self):
        self.packet.fail_enable = 3
        with self.assertRaises(IOError):
            self.bus.enable_torque()
        self.assertTrue(all(v[40] == 0 for v in self.packet.values.values()))

    def test_allow_holding_failure_preserves_preexisting_torque(self):
        # RegisterBus deliberately models a simulator-only rule that rejects
        # powered motors during preparation.  Use a hardware-like subclass so
        # this test reaches the later torque-enable failure path.
        self.packet = type("HoldingRegisterBus", (RegisterBus,), {})()
        self.bus.packet_handler = self.packet
        for values in self.packet.values.values():
            values[40] = 1
        self.packet.fail_enable = 3
        with self.assertRaises(IOError):
            self.bus.enable_torque(allow_holding=True)
        self.assertTrue(all(v[40] == 1 for v in self.packet.values.values()))

    def test_allow_holding_rejects_mixed_torque_before_writes(self):
        self.packet.values[1][40] = 1
        with self.assertRaisesRegex(RuntimeError, "Mixed torque state"):
            self.bus.enable_torque(allow_holding=True)
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_existing_powered_motor_refuses_takeover(self):
        self.packet.values[2][40] = 1
        with self.assertRaises(RuntimeError):
            self.bus.enable_torque()
        self.assertEqual(self.packet.values[2][40], 1)
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_fault_refuses_startup(self):
        self.packet.values[6][65] = 1
        with self.assertRaises(RuntimeError):
            self.bus.enable_torque()
        self.assertFalse(any(e[0] == "write" for e in self.packet.events))

    def test_unexpected_power_during_preparation_is_detected(self):
        self.packet.power_during_goals = True
        with self.assertRaisesRegex(RuntimeError, "became powered"):
            self.bus.enable_torque()
        self.assertFalse(any(e[0] == "write" and e[2] == 40 for e in self.packet.events))


if __name__ == "__main__":
    unittest.main()
