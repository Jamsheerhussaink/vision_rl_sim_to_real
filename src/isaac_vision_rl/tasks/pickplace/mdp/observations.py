"""Deployment-available policy inputs and environment-local critic state."""

import torch
import os


def global_camera_rgb(env):
    """Aspect-preserving RGB actor image, with no depth/geometry estimator."""
    from isaac_vision_rl.utils.rgb_image import actor_rgb
    image = env.scene["camera"].data.output["rgb"]
    return actor_rgb(image.torch if hasattr(image, "torch") else image)


def time_remaining(env):
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0, 1).unsqueeze(-1)


def filtered_joint_targets(env):
    """Expose actuator filter memory so the policy can predict its next command."""
    return torch.cat(
        [env.action_manager.get_term(name).processed_actions for name in ("arm_action", "gripper_action")], dim=-1
    )






def critic_task_state(env):
    """Position, velocity, rotation and reward/termination memory for V(s)."""
    cube, box, robot = env.scene["cube"], env.scene["box"], env.scene["robot"]
    ee_idx = robot.find_bodies("gripper_frame_link")[0][0]
    cube_pos = cube.data.root_pos_w.torch
    box_pos = box.data.root_pos_w.torch
    ee_pos = robot.data.body_pos_w.torch[:, ee_idx]
    origins = env.scene.env_origins
    reach = torch.norm(cube_pos - ee_pos, dim=-1)
    held_time = torch.zeros(env.num_envs, device=env.device)
    lifted = torch.zeros_like(held_time)
    lift_time = torch.zeros_like(held_time)
    grasp_best, lift_best = torch.zeros_like(held_time), torch.zeros_like(held_time)
    # Observation shapes are inferred before reward and termination managers exist.
    if hasattr(env, "reward_manager"):
        best = env.reward_manager.get_term_cfg("reach_progress").func._best_distance
        reach = torch.where(torch.isfinite(best), best, reach)
        grasp_best = env.reward_manager.get_term_cfg("grasp_cube").func._best_score
        lift_best = env.reward_manager.get_term_cfg("lift_cube").func._best_score
    if hasattr(env, "termination_manager"):
        success_name = "lift_success" if "lift_success" in env.termination_manager.active_terms else "placed"
        success = env.termination_manager.get_term_cfg(success_name).func
        held_time = success._held_steps.float() * env.step_dt
        lifted = getattr(success, "_was_lifted", lifted)
        lift_tracker = getattr(success, "_lift_tracker", success)
        lift_time = lift_tracker._held_steps.float() * env.step_dt
    return torch.cat(
        (
            cube_pos - origins,
            box_pos - origins,
            ee_pos - origins,
            cube_pos - ee_pos,
            box_pos - cube_pos,
            cube.data.root_lin_vel_w.torch,
            cube.data.root_quat_w.torch,
            (env.episode_length_buf / env.max_episode_length).unsqueeze(-1),
            reach.unsqueeze(-1),
            held_time.unsqueeze(-1),
            lifted.float().unsqueeze(-1),
            lift_time.unsqueeze(-1),
            grasp_best.unsqueeze(-1),
            lift_best.unsqueeze(-1),
        ),
        dim=-1,
    )
