"""Bounded joint references for realistic motor speeds under exploratory actions."""

from typing import cast

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.utils.math import matrix_from_quat

from isaac_vision_rl.utils.grasp_assist import CameraGraspLatch
from isaac_vision_rl.utils.motion_filter import CapturedJointHold, bounded_reference_step

from .actions_cfg import CameraGripperActionCfg, SlewLimitedArmActionCfg


class CameraGripperAction(JointPositionAction):
    """Hybrid lift controller: keep open until a camera capture, then hold closed.

    Raw policy gripper actions are overridden while enabled. Placement disables
    this lift-only assist so policy release commands remain available.
    """

    def __init__(self, cfg: CameraGripperActionCfg, env):
        super().__init__(cfg, env)
        self._assist_env = env
        self._assist_cfg = cfg
        self._latch = CameraGraspLatch(
            env.num_envs, env.device, cfg.xy_tolerance, cfg.z_tolerance, cfg.min_confidence, cfg.confirm_steps
        )
        self._tool_body = self._asset.find_bodies("gripper_frame_link")[0][0] if cfg.enabled else None
        self._latch.closure_hold_steps = max(1, round(cfg.closure_hold_s / env.step_dt))
        self._released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._jaw_reference = torch.full((env.num_envs,), 1.2, device=env.device)
        self.capture_diagnostics = {}

    @property
    def grasp_latched(self):
        return self._latch.latched

    @property
    def grasp_released(self):
        return self._released

    @property
    def closure_hold_steps(self):
        return self._latch.hold_steps

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._latch.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._released[ids] = False
        self._processed_actions[ids] = 1.2
        self._jaw_reference[ids] = 1.2

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite gripper action")
        super().process_actions(actions)
        if not self._assist_cfg.enabled:
            return
        assert self._tool_body is not None
        from .observations import camera_cube_observation

        camera = camera_cube_observation(self._assist_env)
        base_rotation = matrix_from_quat(self._asset.data.root_quat_w.torch)
        tool_rotation = matrix_from_quat(self._asset.data.body_quat_w.torch[:, self._tool_body])
        relative_world = torch.bmm(base_rotation, camera[:, :3].unsqueeze(-1))
        relative_tool = torch.bmm(tool_rotation.transpose(1, 2), relative_world).squeeze(-1)
        latched = self._latch.update(relative_tool, camera[:, 5], camera[:, 6], tool_rotation[:, 2, 2] < -0.70)
        self.capture_diagnostics = {
            **self._latch.gates,
            "relative_tool_xyz": relative_tool,
            "ready_steps": self._latch.ready_steps,
        }
        log = self._assist_env.extras.setdefault("log", {})
        for name, gate in self._latch.gates.items():
            log[f"Grasp/capture_gate_{name}"] = gate.float().mean()

        # Placement release: when the cube has been carried over the box and lowered into it
        if "box" in self._assist_env.scene.keys():
            box = self._assist_env.scene["box"]
            cube = self._assist_env.scene["cube"]
            cube_pos = cube.data.root_pos_w.torch
            box_pos = box.data.root_pos_w.torch
            xy_box_dist = torch.norm(box_pos[:, :2] - cube_pos[:, :2], dim=-1)
            rel_box_z = cube_pos[:, 2] - box_pos[:, 2]
            auto_release = (xy_box_dist < 0.055) & (rel_box_z < 0.09) & (rel_box_z > 0.015)
            policy_release = (xy_box_dist < 0.08) & (actions[:, 0] > 0.3)
            should_release = latched & (auto_release | policy_release)
            self._released |= should_release
            latched = latched & ~self._released

        desired = torch.where(latched, 0.0, 1.2)
        if self._assist_cfg.close_target_speed > 0:
            step = self._assist_cfg.close_target_speed * self._assist_env.step_dt
            self._jaw_reference += (desired - self._jaw_reference).clamp(-step, step)
            self._processed_actions[:, 0] = self._jaw_reference
        else:
            self._processed_actions[:, 0] = desired
        self._assist_env.extras.setdefault("log", {})["Grasp/assist_latched_fraction"] = latched.float().mean()


class SlewLimitedArmAction(EMAJointPositionToLimitsAction):
    """Retain full joint range and EMA, but limit target velocity per policy step."""

    def __init__(self, cfg: SlewLimitedArmActionCfg, env):
        super().__init__(cfg, env)
        if cfg.max_target_speed <= 0:
            raise ValueError("max_target_speed must be positive")
        self._max_step = cfg.max_target_speed * env.step_dt
        self._assist_env = env
        self._capture_targets = self._processed_actions.clone()
        self._holding_capture = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._target_velocity = torch.zeros_like(self._processed_actions)
        self._speed_limits = torch.full_like(self._processed_actions, cfg.max_target_speed)
        self._roll_index = self._joint_names.index("wrist_roll")
        if cfg.wrist_roll_speed > 0:
            self._speed_limits[:, self._roll_index] = cfg.wrist_roll_speed
        self._roll_hold = CapturedJointHold(env.num_envs, env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # The policy observes processed targets as filter memory. Keep that
        # observation consistent with the actual EMA memory after a reset.
        self._processed_actions[ids] = self._prev_applied_actions[ids]
        self._holding_capture[ids] = False
        self._target_velocity[ids] = 0
        self._roll_hold.reset(env_ids)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite policy actions; refusing to send them into physics")
        previous = self._prev_applied_actions.clone()
        super().process_actions(actions)
        gripper = None
        if hasattr(self._assist_env, "action_manager"):
            gripper = cast(CameraGripperAction, self._assist_env.action_manager.get_term("gripper_action"))
            if self.cfg.hold_wrist_roll_after_capture:
                latch = gripper.grasp_latched & ~gripper.grasp_released
                joint_id = self._joint_ids[self._roll_index]
                limits = self._asset.data.soft_joint_pos_limits.torch[:, joint_id]
                self._processed_actions[:, self._roll_index] = self._roll_hold.apply(
                    self._processed_actions[:, self._roll_index],
                    self._asset.data.joint_pos.torch[:, joint_id],
                    latch, limits[:, 0], limits[:, 1],
                )
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
        if gripper is not None:
            hold = cast(CameraGripperAction, gripper).closure_hold_steps > 0
            start_hold = hold & ~self._holding_capture
            self._capture_targets[start_hold] = self._asset.data.joint_pos.torch[:, self._joint_ids][start_hold]
            self._processed_actions[hold] = self._capture_targets[hold]
            self._target_velocity[hold] = 0
            self._holding_capture[:] = hold
        self._prev_applied_actions[:] = self._processed_actions
