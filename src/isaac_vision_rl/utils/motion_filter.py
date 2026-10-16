"""Velocity and acceleration bounds on joint position references, independent of simulation."""

import torch




def bounded_reference_step(position, target, velocity, speed_limit, acceleration, dt):
    """Brake near a target and bound reference acceleration, including direction reversals.

    Returns reference position and velocity. Bounds apply to commanded references;
    motor tracking and contact forces must be measured separately.
    """
    if acceleration <= 0 or dt <= 0:
        raise ValueError("acceleration and dt must be positive")
    error = target - position
    # Reserve one control step for deceleration. Avoid abrupt stopping at a target.
    braking_speed = (torch.sqrt((acceleration * dt) ** 2 + 2 * acceleration * error.abs()) - acceleration * dt)
    desired = error.sign() * torch.minimum(braking_speed, speed_limit)
    next_velocity = velocity + (desired - velocity).clamp(-acceleration * dt, acceleration * dt)
    return position + next_velocity * dt, next_velocity
