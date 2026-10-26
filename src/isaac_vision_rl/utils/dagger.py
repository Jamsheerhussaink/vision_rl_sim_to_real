"""Small, testable helpers for safe teacher correction rollouts."""

import math


def should_use_teacher(
    *,
    policy_steps: int,
    clearance_m: float,
    takeover_after_steps: int,
    minimum_clearance_m: float,
    teacher_already_in_control: bool = False,
) -> bool:
    """Switch from the learner to the teacher after a short safe probe.

    ``policy_steps`` counts learner actions already executed. The clearance
    test is checked before the next action, so the teacher takes over before
    the configured table margin is crossed.
    """
    if takeover_after_steps < 0:
        raise ValueError("takeover_after_steps must be non-negative")
    if not math.isfinite(clearance_m) or not math.isfinite(minimum_clearance_m):
        raise ValueError("clearance values must be finite")
    if minimum_clearance_m < 0:
        raise ValueError("minimum_clearance_m must be non-negative")
    return bool(
        teacher_already_in_control
        or policy_steps >= takeover_after_steps
        or clearance_m < minimum_clearance_m
    )
