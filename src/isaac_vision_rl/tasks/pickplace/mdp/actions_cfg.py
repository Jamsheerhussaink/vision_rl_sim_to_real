"""Action configuration only: keep concrete simulation imports deferred until launch."""

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg, JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class SlewLimitedArmActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type | str = "isaac_vision_rl.tasks.pickplace.mdp.actions:SlewLimitedArmAction"
    max_target_speed: float = 1.5
    max_target_acceleration: float = 0.0
    wrist_roll_speed: float = 0.0


@configclass
class PolicyGripperActionCfg(JointPositionActionCfg):
    class_type: type | str = "isaac_vision_rl.tasks.pickplace.mdp.actions:PolicyGripperAction"
