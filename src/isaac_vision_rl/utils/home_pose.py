"""Photo-reference home, measured read-only on the real arm on 2026-10-15."""

from isaac_vision_rl.utils.calibration import FollowerArmCalibration

HOME_TICKS = (2012, 904, 3096, 2797, 3089, 2037)


def home_joint_positions():
    """Use the same calibrated coordinates as the hardware bridge."""
    return FollowerArmCalibration().ticks_to_dict(HOME_TICKS)
