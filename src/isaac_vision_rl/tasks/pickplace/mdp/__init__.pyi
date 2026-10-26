# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "assert_finite_physics",
    "SlewLimitedArmActionCfg",
    "PolicyGripperActionCfg",
    "select_task_camera_view",
    "time_remaining",
    "filtered_joint_targets",
    "global_camera_rgb",
    "critic_task_state",
    "PlaceSuccess",
    "ReachCubeProgress",
    "LiftSuccess",
    "GraspCubeProgress",
    "LiftCubeProgress",
    "bilateral_cube_contact",
    "continuous_reach_cube",
    "joint_pos_target_l2",
    "object_ee_distance",
    "ee_approach_from_above",
    "open_gripper_approach",
    "premature_gripper_closure_penalty",
    "grasp_cube",
    "grasp_approach_shaped",
    "grasp_close_shaped",
    "contact_grade_shaped",
    "cube_table_drag_penalty",
    "CubeTableDisplacementPenalty",
    "gripper_table_collision_penalty",
    "safe_joint_vel_penalty",
    "lift_cube_upward",
    "cube_box_ground_proximity_penalty",
    "object_goal_distance",
    "cube_in_box",
    "release_cube_in_box",
    "pinch_force_bonus",
    "gripper_camera_boundary_penalty",
    "gripper_outside_camera_view",
    "object_outside_camera_view",
    "unsafe_hand_table_contact",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions_cfg import PolicyGripperActionCfg, SlewLimitedArmActionCfg
from .events import select_task_camera_view
from .health import assert_finite_physics
from .observations import (
    critic_task_state,
    filtered_joint_targets,
    global_camera_rgb,
    time_remaining,
)
from .terminations import (
    unsafe_hand_table_contact,
    gripper_outside_camera_view,
    object_outside_camera_view,
)
from .rewards import (
    CubeTableDisplacementPenalty,
    GraspCubeProgress,
    LiftCubeProgress,
    LiftSuccess,
    PlaceSuccess,
    ReachCubeProgress,
    bilateral_cube_contact,
    continuous_reach_cube,
    cube_box_ground_proximity_penalty,
    cube_in_box,
    cube_table_drag_penalty,
    cube_tilt_penalty,
    compute_fixed_pad_pos,
    compute_sidewall_depth_gate,
    compute_cube_upright,
    dense_rgb_approach,
    ee_approach_from_above,
    grasp_cube,
    grasp_approach_shaped,
    grasp_close_shaped,
    contact_grade_shaped,
    gripper_camera_boundary_penalty,
    gripper_table_collision_penalty,
    joint_pos_target_l2,
    lift_cube_upward,
    object_ee_distance,
    object_goal_distance,
    open_gripper_approach,
    pinch_force_bonus,
    premature_gripper_closure_penalty,
    release_cube_in_box,
    safe_joint_vel_penalty,
)
