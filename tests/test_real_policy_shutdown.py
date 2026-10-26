"""Deployment cleanup regressions: no camera, simulator, serial or motor access."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from isaac_vision_rl.utils.calibration import FollowerArmCalibration, JOINT_ORDER
from isaac_vision_rl.utils.home_pose import HOME_TICKS
from isaac_vision_rl.utils.real_policy import (
    RealPolicy, camera_profile_dimensions, deployed_policy_home,
    validate_actor_state_dict, validate_real_checkpoint_metadata,
)

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


def test_saved_rgb_profile_selects_matching_native_camera_dimensions():
    assert camera_profile_dimensions("424x240_letterbox128_v1") == (424, 240)
    assert camera_profile_dimensions("1280x720_letterbox128_v1") == (1280, 720)
    with np.testing.assert_raises_regex(ValueError, "Unsupported RGB profile"):
        camera_profile_dimensions("unknown")


def test_real_policy_accepts_rgb_four_stage_schema_with_matching_actor_inputs():
    validate_real_checkpoint_metadata({
        "so101_agent_schema": "four_stage_v1",
        "so101_policy_variant": "rgb",
    })
    validate_real_checkpoint_metadata({
        "so101_agent_schema": "four_stage_v2_bounded",
        "so101_policy_variant": "rgb",
    })


def test_real_policy_rejects_privileged_actor_stage_input_shape():
    expected = {"mlp.0.weight": np.zeros((256, 9241), dtype=np.float32)}
    compatible = {"mlp.0.weight": np.zeros((256, 9241), dtype=np.float32)}
    validate_actor_state_dict(expected, compatible)
    incompatible = {"mlp.0.weight": np.zeros((256, 9245), dtype=np.float32)}
    with np.testing.assert_raises_regex(ValueError, "simulator-only actor stage input"):
        validate_actor_state_dict(expected, incompatible)


def test_deployed_policy_home_matches_runtime_photo_home_and_open_gripper():
    home = deployed_policy_home()
    ticks = FollowerArmCalibration().rad_to_ticks(home).tolist()
    assert ticks[:5] == list(HOME_TICKS[:5])
    assert home[5] == 1.2
    assert len(home) == len(JOINT_ORDER)


def test_closed_photo_home_is_valid_only_for_aligned_repeat_attempts():
    expected = [2012, 904, 3096, 2797, 3089, 3545]
    closed = [2012, 904, 3096, 2797, 3089, 2037]
    aligned = module.start_pose_check(closed, expected, align_home=True, closed_gripper_tick=2037)
    assert aligned["allowed"]
    assert aligned["starting_from_closed_photo_home"]
    unaligned = module.start_pose_check(closed, expected, align_home=False, closed_gripper_tick=2037)
    assert not unaligned["allowed"]
    moved_arm = [2113, 904, 3096, 2797, 3089, 2037]
    assert not module.start_pose_check(moved_arm, expected, align_home=True, closed_gripper_tick=2037)["allowed"]


def test_table_clearance_guard_rejects_unsafe_target_or_measured_pose():
    unsafe = SimpleNamespace(
        check_segment=Mock(side_effect=ValueError("Hand/table clearance 4.0 mm is below 5.0 mm")),
        minimum=Mock(return_value=0.002),
    )
    with np.testing.assert_raises_regex(ValueError, "4.0 mm"):
        module.safe_command_segment(unsafe, np.zeros(6), np.ones(6))
    with np.testing.assert_raises_regex(RuntimeError, "2.00 mm"):
        module.safe_measured_clearance(unsafe, np.zeros(6))


def test_table_clearance_guard_returns_safe_measurements():
    safe = SimpleNamespace(check_segment=Mock(return_value=0.007), minimum=Mock(return_value=0.006))
    assert module.safe_command_segment(safe, np.zeros(6), np.ones(6)) == 0.007
    assert module.safe_measured_clearance(safe, np.zeros(6)) == 0.006
