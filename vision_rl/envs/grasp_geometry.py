"""Finger placement checks in the cube's own coordinate frame.

This checks geometry, not grip force or successful lifting. A blocked jaw
alone cannot tell a side pinch from one finger resting on the cube's top.
"""
import jax.numpy as jnp


def opposing_finger_sides(fixed_tip, moving_tip, cube, cube_rotation,
                         cube_half=0.025, tolerance=0.008, xp=jnp):
    """Both tips near opposite vertical faces, below the cube's top edge.

    ``cube_rotation`` maps cube-local vectors to world vectors. Row vectors
    multiplied by this matrix therefore map world offsets back to cube-local
    coordinates, so the check also works for a yawed cube.
    """
    fixed = (fixed_tip - cube) @ cube_rotation
    moving = (moving_tip - cube) @ cube_rotation
    below_top = (xp.abs(fixed[2]) < cube_half - 0.003) & (
        xp.abs(moving[2]) < cube_half - 0.003)

    def faces(axis, other):
        return ((fixed[axis] * moving[axis] < 0)
                & (xp.abs(xp.abs(fixed[axis]) - cube_half) <= tolerance)
                & (xp.abs(xp.abs(moving[axis]) - cube_half) <= tolerance)
                & (xp.abs(fixed[other]) <= cube_half + tolerance)
                & (xp.abs(moving[other]) <= cube_half + tolerance))

    return below_top & (faces(0, 1) | faces(1, 0))
