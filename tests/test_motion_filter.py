"""Bounds must hold through abrupt policy reversals, not just constant commands."""

import torch

from isaac_vision_rl.utils.motion_filter import bounded_reference_step






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
