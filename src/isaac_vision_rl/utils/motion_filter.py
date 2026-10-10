"""Velocity and acceleration bounds on joint position references, independent of simulation."""

import torch


class CapturedJointHold:
    """Retain a measured joint angle during capture, with independent episode resets."""

    def __init__(self, count, device):
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.reference = torch.zeros(count, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.active[ids] = False
        self.reference[ids] = 0

    def apply(self, desired, measured, captured, lower, upper):
        first = captured & ~self.active
        self.reference[first] = measured[first]
        self.reference[:] = torch.maximum(torch.minimum(self.reference, upper), lower)
        self.active[:] = captured
        return torch.where(captured, self.reference, desired)


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
