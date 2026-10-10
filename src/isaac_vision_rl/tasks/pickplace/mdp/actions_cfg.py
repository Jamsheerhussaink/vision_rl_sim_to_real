"""Action configuration only: keep concrete simulation imports deferred until launch."""

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg, JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class SlewLimitedArmActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type | str = "isaac_vision_rl.tasks.pickplace.mdp.actions:SlewLimitedArmAction"
    max_target_speed: float = 1.5
    max_target_acceleration: float = 0.0
    wrist_roll_speed: float = 0.0
    hold_wrist_roll_after_capture: bool = False


@configclass
class CameraGripperActionCfg(JointPositionActionCfg):
    class_type: type | str = "isaac_vision_rl.tasks.pickplace.mdp.actions:CameraGripperAction"
    enabled: bool = True
    xy_tolerance: float = 0.012
    z_tolerance: float = 0.015
    min_confidence: float = 0.75
    confirm_steps: int = 2
    close_target_speed: float = 0.0
    closure_hold_s: float = 0.6
