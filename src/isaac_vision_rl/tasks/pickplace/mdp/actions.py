"""Bounded joint references for realistic motor speeds under exploratory actions."""


import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction

from isaac_vision_rl.utils.motion_filter import bounded_reference_step

from .actions_cfg import PolicyGripperActionCfg, SlewLimitedArmActionCfg




class SlewLimitedArmAction(EMAJointPositionToLimitsAction):
    """Retain full joint range and EMA, but limit target velocity per policy step."""

    def __init__(self, cfg: SlewLimitedArmActionCfg, env):
        super().__init__(cfg, env)
        if cfg.max_target_speed <= 0:
            raise ValueError("max_target_speed must be positive")
        self._max_step = cfg.max_target_speed * env.step_dt
        self._assist_env = env
        self._target_velocity = torch.zeros_like(self._processed_actions)
        self._speed_limits = torch.full_like(self._processed_actions, cfg.max_target_speed)
        self._roll_index = self._joint_names.index("wrist_roll")
        if cfg.wrist_roll_speed > 0:
            self._speed_limits[:, self._roll_index] = cfg.wrist_roll_speed

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # The policy observes processed targets as filter memory. Keep that
        # observation consistent with the actual EMA memory after a reset.
        self._processed_actions[ids] = self._prev_applied_actions[ids]
        self._target_velocity[ids] = 0

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite policy actions; refusing to send them into physics")
        previous = self._prev_applied_actions.clone()
        super().process_actions(actions)
        if self.cfg.max_target_acceleration > 0:
            reference, velocity = bounded_reference_step(
                previous, self._processed_actions, self._target_velocity, self._speed_limits,
                self.cfg.max_target_acceleration, self._assist_env.step_dt,
            )
            # Hard joint bounds take priority over smooth braking at an endpoint.
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            clipped = torch.maximum(torch.minimum(reference, limits[..., 1]), limits[..., 0])
            self._target_velocity[:] = torch.where(reference == clipped, velocity, 0.0)
            self._processed_actions[:] = clipped
        else:
            self._processed_actions[:] = previous + torch.clamp(
                self._processed_actions - previous, -self._max_step, self._max_step
            )
        self._prev_applied_actions[:] = self._processed_actions


class PolicyGripperAction(JointPositionAction):
    """Apply the learned jaw target without camera-based overrides."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._control_dt=env.step_dt
        self._reference_velocity=torch.zeros_like(self._processed_actions)
        self._previous_target=self._processed_actions.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        if self.cfg.max_target_speed>0:
            actual=self._asset.data.joint_pos.torch[:,self._joint_ids]
            self._processed_actions[ids]=actual[ids]
            self._previous_target[ids]=actual[ids]
            self._reference_velocity[ids]=0
            return
        # Preserve the RGB controller's observed reset target for checkpoints.
        self._processed_actions[slice(None) if env_ids is None else env_ids] = 1.2

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite gripper action")
        super().process_actions(actions)
        if self.cfg.max_target_speed>0:
            reference,velocity=bounded_reference_step(
                self._previous_target,self._processed_actions,self._reference_velocity,
                torch.full_like(self._processed_actions,self.cfg.max_target_speed),
                self.cfg.max_target_acceleration,self._control_dt)
            limits=self._asset.data.soft_joint_pos_limits.torch[:,self._joint_ids]
            clipped=torch.maximum(torch.minimum(reference,limits[...,1]),limits[...,0])
            self._reference_velocity[:]=torch.where(reference==clipped,velocity,0.)
            self._processed_actions[:]=clipped
            self._previous_target[:]=clipped
