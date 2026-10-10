"""Bounds must hold through abrupt policy reversals, not just constant commands."""

import torch

from isaac_vision_rl.utils.motion_filter import CapturedJointHold, bounded_reference_step


def test_capture_hold_uses_actual_angle_and_releases_independently():
    hold = CapturedJointHold(2, "cpu")
    desired = torch.tensor([0.8, -0.9])
    actual = torch.tensor([0.2, -0.3])
    lower, upper = torch.full((2,), -1.0), torch.ones(2)
    assert torch.equal(hold.apply(desired, actual, torch.tensor([False, False]), lower, upper), desired)
    assert torch.equal(hold.apply(desired, actual, torch.tensor([True, False]), lower, upper), torch.tensor([0.2, -0.9]))
    actual[:] = torch.tensor([0.5, -0.1])
    assert torch.equal(hold.apply(desired, actual, torch.tensor([True, True]), lower, upper), torch.tensor([0.2, -0.1]))
    hold.reset(torch.tensor([0]))
    assert hold.active.tolist() == [False, True]
    assert torch.equal(hold.apply(desired, actual, torch.tensor([True, False]), lower, upper), torch.tensor([0.5, -0.9]))
    hold.reset()
    assert not hold.active.any()


def test_capture_reference_respects_joint_limits():
    hold = CapturedJointHold(2, "cpu")
    out = hold.apply(torch.zeros(2), torch.tensor([1.2, -1.2]), torch.ones(2, dtype=torch.bool),
                     torch.full((2,), -1.0), torch.ones(2))
    assert torch.equal(out, torch.tensor([1.0, -1.0]))


def test_speed_and_acceleration_bounds_through_reversal():
    position = torch.zeros(2, 5)
    velocity = torch.zeros_like(position)
    speed = torch.full_like(position, 0.75)
    speed[:, -1] = 0.25
    for i in range(200):
        target = torch.full_like(position, 1.0 if i < 100 else -1.0)
        previous_velocity = velocity.clone()
        previous_position = position.clone()
        position, velocity = bounded_reference_step(position, target, velocity, speed, 2.0, 0.04)
        assert torch.all(velocity.abs() <= speed + 1e-6)
        assert torch.all((velocity - previous_velocity).abs() <= 2.0 * 0.04 + 1e-6)
        assert torch.allclose(position - previous_position, velocity * 0.04, atol=1e-7)


def test_filter_settles_without_persistent_oscillation():
    position = torch.zeros(1, 1)
    velocity = torch.zeros_like(position)
    for _ in range(300):
        position, velocity = bounded_reference_step(position, torch.ones_like(position), velocity,
                                                    torch.full_like(position, 0.75), 2.0, 0.04)
    assert torch.allclose(position, torch.ones_like(position), atol=1e-5)
    assert velocity.abs().max() < 1e-5
