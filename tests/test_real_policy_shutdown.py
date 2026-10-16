"""Deployment cleanup regressions: no camera, simulator, serial or motor access."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from isaac_vision_rl.utils.real_policy import RealPolicy

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/hardware/real_camera_policy.py'
spec = importlib.util.spec_from_file_location('real_camera_shutdown_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_preflight_failure_closes_without_movement_or_torque_change():
    bus = Mock(connected=True)
    report = {}
    module.close_policy_session(None, bus, report)
    bus.close.assert_called_once_with(disable_torque=False)
    bus.park_home_smoothly.assert_not_called()
    bus.enable_torque.assert_not_called()
    bus.sync_write_positions.assert_not_called()
    assert report['motor_shutdown_state'] == 'serial_closed_torque_unchanged'


def test_worker_failure_does_not_trigger_fallback_bus_commands():
    bus = Mock(connected=True)
    worker = SimpleNamespace(close=Mock(side_effect=RuntimeError('worker still running')),
                             shutdown_state='powered')
    report = {}
    module.close_policy_session(worker, bus, report)
    assert bus.mock_calls == []
    assert report['shutdown_error'] == 'worker still running'


def test_successful_worker_shutdown_is_reported_without_second_close():
    bus = Mock(connected=True)
    worker = SimpleNamespace(close=Mock(), shutdown_state='torque_left_enabled')
    report = {}
    module.close_policy_session(worker, bus, report)
    worker.close.assert_called_once_with()
    assert bus.mock_calls == []
    assert report['motor_shutdown_state'] == 'torque_left_enabled'


def test_serial_close_failure_is_not_reported_as_confirmed():
    bus = Mock(connected=True)
    bus.close.side_effect = IOError('serial unavailable')
    report = {}
    module.close_policy_session(None, bus, report)
    assert report['motor_shutdown_state'] == 'shutdown_unconfirmed'
    bus.park_home_smoothly.assert_not_called()


def test_executed_target_replaces_unreachable_controller_memory():
    policy = RealPolicy.__new__(RealPolicy)
    policy.filtered = np.full(6, 99.0)
    sent = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 1.2])
    policy.set_executed_target(sent)
    np.testing.assert_allclose(policy.filtered, sent)
    with np.testing.assert_raises_regex(ValueError, "six finite"):
        policy.set_executed_target([0.0] * 5)
