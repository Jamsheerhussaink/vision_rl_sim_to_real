# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Distance between end-effector tool center point and object."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch

    distance = torch.norm(cube_pos - ee_pos, dim=-1)
    # A small, local shaping reward; distant hovering should have little value.
    return torch.exp(-distance / 0.05)


class ReachCubeProgress(ManagerTermBase):
    """Reward each new closest approach directly toward the cube.

    No overhead waypoint or descent stage is required. Stationary proximity
    and backtracking do not generate additional progress credit.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._best_distance = torch.full((env.num_envs,), float("inf"), device=env.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._best_distance[ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        ee_body_name: str = "gripper_frame_link",
    ) -> torch.Tensor:
        robot = env.scene[ee_cfg.name]
        cube = env.scene[object_cfg.name]
        body_idx = robot.find_bodies(ee_body_name)[0][0]
        distance = torch.norm(cube.data.root_pos_w.torch - robot.data.body_pos_w.torch[:, body_idx], dim=-1)
        previous = torch.where(torch.isfinite(self._best_distance), self._best_distance, distance)
        progress = torch.clamp(previous - distance, min=0.0) / 0.05
        self._best_distance[:] = torch.minimum(previous, distance)
        return progress


def bilateral_cube_contact(env, min_force=0.02):
    """Require cube-specific contact on both gripper bodies, not contact with the table."""
    contacts = []
    for name in ("fixed_finger_contact", "moving_finger_contact"):
        forces = env.scene[name].data.normal_force_matrix_w
        if forces is None:
            raise RuntimeError("Filtered finger-to-cube contact reporting is required")
        forces = forces.torch if hasattr(forces, "torch") else forces
        contacts.append(torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1).amax(dim=1) > min_force)
    bilateral = contacts[0] & contacts[1]
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["Grasp/bilateral_contact_fraction"] = bilateral.float().mean()
    return bilateral


class GraspCubeProgress(ManagerTermBase):
    """Pay only for new best grasp quality; stationary/repeated touching cannot farm reward."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._best_score = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self._best_score[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        score = grasp_cube(env)
        progress = (score - self._best_score).clamp_min(0)
        self._best_score[:] = torch.maximum(score, self._best_score)
        return progress / env.step_dt


class LiftCubeProgress(GraspCubeProgress):
    """Pay only new best contact-gated lift quality, not repeated hovering."""

    def __call__(self, env):
        score = lift_cube_upward(env)
        progress = (score - self._best_score).clamp_min(0)
        self._best_score[:] = torch.maximum(score, self._best_score)
        return progress / env.step_dt


class LiftSuccess(ManagerTermBase):
    """Terminate after two-sided cube contact and a stable elevated hold for 0.5 s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._held_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._held_steps[ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float = 0.47,
        hold_time_s: float = 0.5,
        ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        ee_body_name: str = "gripper_frame_link",
    ) -> torch.Tensor:
        robot = env.scene[ee_cfg.name]
        cube = env.scene[object_cfg.name]
        body_idx = robot.find_bodies(ee_body_name)[0][0]
        gripper_idx = robot.find_joints("gripper")[0][0]
        cube_pos = cube.data.root_pos_w.torch
        offset = cube_pos - robot.data.body_pos_w.torch[:, body_idx]
        xy_dist = torch.norm(offset[:, :2], dim=-1)
        z_dist = torch.abs(offset[:, 2])
        held = (
            bilateral_cube_contact(env)
            & (cube_pos[:, 2] > minimal_height)
            & (xy_dist < 0.035)
            & (z_dist < 0.045)
            # The verified loaded grasp settles near 0.97 rad; empty-jaw
            # closure angles cannot identify a grasp around this cube.
            & (robot.data.joint_pos.torch[:, gripper_idx] < 1.10)
            & (torch.norm(cube.data.root_lin_vel_w.torch, dim=-1) < 0.25)
        )
        self._held_steps[:] = torch.where(held, self._held_steps + 1, 0)
        return self._held_steps >= math.ceil(hold_time_s / env.step_dt)


def _tool_downward_alignment(robot: Articulation, body_idx: int, env) -> torch.Tensor:
    if hasattr(robot.data, "body_quat_w"):
        tool_quat = robot.data.body_quat_w.torch if hasattr(robot.data.body_quat_w, "torch") else robot.data.body_quat_w
        tool_quat = tool_quat[:, body_idx]
        tool_rot = matrix_from_quat(tool_quat)
        return torch.clamp(-tool_rot[:, 2, 2], min=0.0, max=1.0)
    return torch.ones(env.num_envs, device=env.device)


def ee_approach_from_above(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Reward positioning the gripper tool center point above the cube before descending."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch

    z_diff = ee_pos[:, 2] - cube_pos[:, 2]
    xy_dist = torch.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=-1)
    # Smooth target around 6 cm above the cube. Lateral alignment is much tighter
    # than the old 15 cm gate, so side contact does not earn approach reward.
    xy_alignment = torch.exp(-torch.square(xy_dist / 0.035))
    height_alignment = torch.exp(-torch.square((z_diff - 0.06) / 0.045))
    safely_above = (z_diff > 0.015).float()
    on_table = (cube_pos[:, 2] < 0.44).float()

    downward = _tool_downward_alignment(robot, body_idx, env)
    return on_table * safely_above * xy_alignment * height_alignment * downward


def open_gripper_approach(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Reward keeping gripper jaws OPEN while approaching the cube (prevents prematurely closing)."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch
    offset = cube_pos - ee_pos
    distance = torch.norm(offset, dim=-1)

    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos.torch[:, gripper_idx]

    # When approaching (> 5cm away), reward maintaining open jaws (~0.80 rad)
    far_from_cube = (distance > 0.05).float()
    open_score = torch.clamp(gripper_pos / 0.80, min=0.0, max=1.0)

    return far_from_cube * open_score


def premature_gripper_closure_penalty(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Penalize closing until the cube is centered in the grasp cavity."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]
    body_idx = robot.find_bodies(ee_body_name)[0][0]
    gripper_idx = robot.find_joints("gripper")[0][0]

    offset = cube.data.root_pos_w.torch - robot.data.body_pos_w.torch[:, body_idx]
    xy_dist = torch.norm(offset[:, :2], dim=-1)
    z_dist = torch.abs(offset[:, 2])
    grasp_ready = (xy_dist < 0.030) & (z_dist < 0.040)

    gripper_pos = robot.data.joint_pos.torch[:, gripper_idx]
    closure = torch.clamp((1.15 - gripper_pos) / 0.75, min=0.0, max=1.0)
    return (~grasp_ready).float() * closure


def grasp_cube(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Reward closing the gripper fingers when the cube is centered between the fingertips."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch
    offset = cube_pos - ee_pos
    xy_dist = torch.norm(offset[:, :2], dim=-1)
    z_dist = torch.abs(offset[:, 2])

    # Find gripper joint index dynamically
    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos.torch[:, gripper_idx]

    # Closing amount: from open (1.20) to clamped around 5cm cube (~0.96-1.03 rad)
    close_amount = torch.clamp((1.15 - gripper_pos) / 0.20, min=0.0, max=1.0)

    # Require the cube to be centered laterally and vertically between the fingers.
    # A tool touching the side of the cube receives no grasp credit.
    near_cube = torch.clamp(1.0 - xy_dist / 0.030, min=0.0, max=1.0) * torch.clamp(
        1.0 - z_dist / 0.035, min=0.0, max=1.0
    )

    downward = _tool_downward_alignment(robot, body_idx, env)

    return near_cube * close_amount * downward * bilateral_cube_contact(env).float()


def cube_table_drag_penalty(
    env: ManagerBasedRLEnv,
    table_height_threshold: float = 0.435,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Heavy penalty for dragging, shoving, or sliding the cube horizontally across the table (safely bounded)."""
    cube: RigidObject = env.scene[object_cfg.name]
    cube_pos = cube.data.root_pos_w.torch
    cube_lin_vel = cube.data.root_lin_vel_w.torch

    # Check if cube is resting on or close to the tabletop (Z < threshold)
    is_on_table = (cube_pos[:, 2] < table_height_threshold).float()

    # Horizontal speed on table (XY plane) capped at 1.0 m/s for numerical stability
    xy_speed = torch.clamp(torch.norm(cube_lin_vel[:, :2], dim=-1), min=0.0, max=1.0)

    # Bounded penalty: maximum 3.0 per step (prevents physics solver spikes from corrupting gradients)
    return is_on_table * (xy_speed + 2.0 * torch.square(xy_speed))


class CubeTableDisplacementPenalty(ManagerTermBase):
    """Penalize accumulated tabletop displacement from the reset position.

    This catches slow dragging, which can evade a velocity-only penalty.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        cube_xy = env.scene["cube"].data.root_pos_w.torch[:, :2]
        self._start_xy = cube_xy.clone()

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._start_xy[ids] = self._env.scene["cube"].data.root_pos_w.torch[ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        table_height_threshold: float = 0.44,
        free_motion_m: float = 0.005,
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ) -> torch.Tensor:
        cube = env.scene[object_cfg.name]
        cube_pos = cube.data.root_pos_w.torch
        displacement = torch.norm(cube_pos[:, :2] - self._start_xy, dim=-1)
        excess = torch.clamp(displacement - free_motion_m, min=0.0, max=0.10)
        on_table = (cube_pos[:, 2] < table_height_threshold).float()
        return on_table * (excess / 0.025 + torch.square(excess / 0.05))


def gripper_table_collision_penalty(
    env: ManagerBasedRLEnv,
    table_height: float = 0.405,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Penalize the gripper scraping or digging into the tabletop (strictly clamped to max 1.0 per step)."""
    robot: Articulation = env.scene[ee_cfg.name]
    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_z = robot.data.body_pos_w.torch[:, body_idx, 2]

    # Clamp penetration to max 5cm so depenetration solver spikes never blow up PPO returns
    penetration = torch.clamp(table_height - ee_z, min=0.0, max=0.05)
    return penetration * 20.0


def safe_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high joint velocities with robust clamping against physics singularities."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.clamp(robot.data.joint_vel.torch[:, asset_cfg.joint_ids], min=-10.0, max=10.0)
    return torch.sum(torch.square(joint_vel), dim=1)


def lift_cube_upward(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.428,
    target_height: float = 0.52,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Primary reward for picking and lifting the cube vertically off the tabletop.

    Requires proximity, closure and cube-filtered contact on both gripper bodies.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch
    offset = cube_pos - ee_pos
    xy_dist = torch.norm(offset[:, :2], dim=-1)
    z_dist = torch.abs(offset[:, 2])

    # Gripper joint angle (0.80 open, ~0.96 clamping 5cm cube)
    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos.torch[:, gripper_idx]
    is_clamped = torch.clamp((1.15 - gripper_pos) / 0.20, min=0.0, max=1.0)

    # A valid lift begins with the cube centered between the fingers. Side contact
    # and a closed gripper near the cube are insufficient.
    centered = torch.clamp(1.0 - xy_dist / 0.030, min=0.0, max=1.0) * torch.clamp(
        1.0 - z_dist / 0.040, min=0.0, max=1.0
    )
    grasp_envelopment = centered * is_clamped

    cube_z = cube_pos[:, 2]

    # Credit begins immediately as the cube lifts off the tabletop (> 0.428m)
    lift_progress = torch.clamp((cube_z - minimal_height) / (target_height - minimal_height), min=0.0, max=1.0)

    # 2. Lift-off bonus: broke contact with table surface
    is_lifted = (cube_z > minimal_height).float()

    # 3. Target height milestone: held high in the air (> 0.50m)
    is_target_height = (cube_z > (target_height - 0.02)).float()

    # 4. Upward velocity shaping: reward upward pull (vz > 0) when lifting
    cube_vz = cube.data.root_lin_vel_w.torch[:, 2]
    upward_vel = torch.clamp(cube_vz, min=0.0, max=0.5) * (1.0 - is_target_height)

    # 5. Upward gripper pull shaping: when firmly grasping, reward upward movement of the gripper
    if hasattr(robot.data, "body_lin_vel_w"):
        ee_vz = robot.data.body_lin_vel_w.torch[:, body_idx, 2]
    else:
        ee_vz = torch.zeros(env.num_envs, device=env.device)
    ee_upward_vel = torch.clamp(ee_vz, min=0.0, max=0.5) * (1.0 - is_target_height)
    ee_lift_progress = torch.clamp((ee_pos[:, 2] - minimal_height) / (target_height - minimal_height), min=0.0, max=1.0)

    lift_total = (
        lift_progress * 4.0
        + is_lifted * 3.0
        + is_target_height * 5.0
        + upward_vel * 2.0
        + ee_lift_progress * 2.0
        + ee_upward_vel * 2.0
    )

    # Gate elevation by the geometric grasp proxy.
    return grasp_envelopment * lift_total * bilateral_cube_contact(env).float()


def cube_box_ground_proximity_penalty(
    env: ManagerBasedRLEnv,
    table_height_threshold: float = 0.44,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Penalize shoving or pushing the cube into the box while still on or near the tabletop."""
    cube: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[goal_cfg.name]

    cube_pos = cube.data.root_pos_w.torch
    box_pos = box.data.root_pos_w.torch

    # Only penalize if cube is still resting on or sliding along the table surface
    is_on_table = (cube_pos[:, 2] < table_height_threshold).float()

    # Horizontal distance to box center
    xy_dist = torch.norm(box_pos[:, :2] - cube_pos[:, :2], dim=-1)

    # Heavily penalize shoving the cube into the box on the ground (within 14cm of box)
    proximity = torch.clamp((0.14 - xy_dist) / 0.14, min=0.0, max=1.0)
    return is_on_table * proximity


def object_goal_distance(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.440,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Dense transport reward, gated by elevation and geometric grasp proxy."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[goal_cfg.name]

    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch
    ee_cube_dist = torch.norm(cube_pos - ee_pos, dim=-1)

    # Must be held inside gripper (< 5.2cm) and lifted > 0.46m
    gripper_idx = robot.find_joints("gripper")[0][0]
    is_held = ((ee_cube_dist < 0.052) & (robot.data.joint_pos.torch[:, gripper_idx] < 0.50)).float()
    is_lifted = (cube_pos[:, 2] > minimal_height).float()

    xy_dist = torch.norm(box.data.root_pos_w.torch[:, :2] - cube_pos[:, :2], dim=-1)
    # Continuous linear approach gradient from 0.30m + exponential capture bonus
    approach = torch.clamp(1.0 - xy_dist / 0.30, min=0.0, max=1.0)
    capture = torch.exp(-xy_dist / 0.06)
    place_reward = approach * 2.0 + capture * 3.0

    return is_held * is_lifted * place_reward


def cube_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Whole-cube containment and settling; release/history are checked by PlaceSuccess.

    Interior dimensions come from usd/box.usd (walls at x=+/-0.096,
    y=+/-0.071; floor top z=0.004 in the box rigid-body frame).
    """
    cube: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[goal_cfg.name]
    rotation = matrix_from_quat(box.data.root_quat_w.torch)
    rel = torch.bmm(
        rotation.transpose(1, 2), (cube.data.root_pos_w.torch - box.data.root_pos_w.torch).unsqueeze(-1)
    ).squeeze(-1)
    cube_rotation = matrix_from_quat(cube.data.root_quat_w.torch)
    relative_rotation = torch.bmm(rotation.transpose(1, 2), cube_rotation)
    extent = relative_rotation.abs().sum(dim=-1) * 0.025
    inside_xy = (rel[:, 0].abs() + extent[:, 0] < 0.095) & (rel[:, 1].abs() + extent[:, 1] < 0.070)
    inside_z = (rel[:, 2] - extent[:, 2] >= 0.002) & (rel[:, 2] - extent[:, 2] <= 0.006)
    settled = (torch.norm(cube.data.root_lin_vel_w.torch, dim=-1) < 0.05) & (
        torch.norm(cube.data.root_ang_vel_w.torch, dim=-1) < 0.5
    )
    return inside_xy & inside_z & settled


class PlaceSuccess(ManagerTermBase):
    """Require a sustained lift first, then release, withdrawal and a settled cube."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._lift_tracker = LiftSuccess(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._held_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._lift_tracker.reset(env_ids)
        self._was_lifted[ids] = False
        self._held_steps[ids] = 0

    def __call__(self, env, hold_time_s=0.4):
        self._was_lifted |= self._lift_tracker(env)
        robot, cube = env.scene["robot"], env.scene["cube"]
        body_idx = robot.find_bodies("gripper_frame_link")[0][0]
        gripper_idx = robot.find_joints("gripper")[0][0]
        withdrawn = torch.norm(cube.data.root_pos_w.torch - robot.data.body_pos_w.torch[:, body_idx], dim=-1) > 0.065
        released = robot.data.joint_pos.torch[:, gripper_idx] > 1.10
        placed = self._was_lifted & cube_in_box(env) & released & withdrawn
        self._held_steps[:] = torch.where(placed, self._held_steps + 1, 0)
        return self._held_steps >= math.ceil(hold_time_s / env.step_dt)


def release_cube_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "gripper_frame_link",
) -> torch.Tensor:
    """Dense placement shaping: lower a raised cube over the interior, then open."""
    robot: Articulation = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[goal_cfg.name]
    body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, body_idx]
    cube_pos = cube.data.root_pos_w.torch
    box_pos = box.data.root_pos_w.torch
    xy_dist = torch.norm(box_pos[:, :2] - cube_pos[:, :2], dim=-1)
    over_box = torch.exp(-20.0 * xy_dist)
    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos.torch[:, gripper_idx]
    released = torch.clamp((gripper_pos - 1.10) / 0.10, min=0.0, max=1.0)
    height = cube_pos[:, 2] - box_pos[:, 2]
    above_floor = (height > 0.025).float()
    lowering = torch.exp(-torch.abs(height - 0.029) / 0.08)
    close_xy = (xy_dist < 0.04).float()
    near_ee = (torch.norm(cube_pos - ee_pos, dim=-1) < 0.08).float()
    # An open gripper far from the cube must not earn a release reward.
    return above_floor * close_xy * over_box * near_ee * lowering * (0.25 + released)
