"""Measured SO-101 workcell geometry shared by hardware safety tools."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHOTO_HOME_ROBOT_ASSET = ROOT / "usd" / "robot_grasp_surface_v3.usda"
PHOTO_HOME_ROBOT_POSITION = (-0.25, 0.015, 0.405)
PHOTO_HOME_ROBOT_ROTATION = (0.0, 0.0, 0.0, 1.0)


def photo_home_clearance():
    """Create the wrist/finger-to-table checker for the calibrated workcell."""
    from isaac_vision_rl.utils.table_clearance import HandTableClearance

    return HandTableClearance(PHOTO_HOME_ROBOT_ASSET, PHOTO_HOME_ROBOT_POSITION, PHOTO_HOME_ROBOT_ROTATION)
