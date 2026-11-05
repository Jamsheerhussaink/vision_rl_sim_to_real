"""Joint-target smoothing shared by training and scripted physics checks."""
import jax.numpy as jnp


def smooth_joint_targets(goal, previous, position, low, high, *, max_delta,
                         smooth_alpha, windup_margin, xp=jnp):
    move = xp.clip(smooth_alpha * (goal - previous), -max_delta, max_delta)
    target = xp.clip(previous + move, position - windup_margin, position + windup_margin)
    return xp.clip(target, low, high)
