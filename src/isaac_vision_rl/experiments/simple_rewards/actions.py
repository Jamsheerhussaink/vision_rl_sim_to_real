"""Experimental jaw controller; original controller remains untouched."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaac_vision_rl.tasks.pickplace.mdp.actions import PolicyGripperAction
from isaac_vision_rl.utils.motion_filter import bounded_reference_step
from .jaw_guidance import closing_permission


class GuidedGripperAction(PolicyGripperAction):
    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Non-finite gripper action')
        # Obtain the learned target before applying the existing motion filter.
        JointPositionAction.process_actions(self, actions)
        from .stages import machine, measurements
        m = measurements(self._env)
        allowed = closing_permission(machine(self._env).stage,
            m['pinch_center_distance'], m['downward_cosine'], m['cube_between_fingers'])
        self._processed_actions[:] = torch.where(allowed[:, None], self._processed_actions,
                                                torch.full_like(self._processed_actions, 1.2))
        reference, velocity = bounded_reference_step(
            self._previous_target, self._processed_actions, self._reference_velocity,
            torch.full_like(self._processed_actions, self.cfg.max_target_speed),
            self.cfg.max_target_acceleration, self._control_dt)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        clipped = torch.maximum(torch.minimum(reference, limits[..., 1]), limits[..., 0])
        self._reference_velocity[:] = torch.where(reference == clipped, velocity, 0.)
        self._processed_actions[:] = clipped
        self._previous_target[:] = clipped
