"""Deployment-available policy inputs and environment-local critic state."""

import torch


def global_camera_rgb(env):
    """Keep full sensor resolution for geometry, downsample only the CNN input."""
    image = env.scene["camera"].data.output["rgb"]
    image = image.torch if hasattr(image, "torch") else image
    rgb = image[..., :3].permute(0, 3, 1, 2).float() / 255.0
    return torch.nn.functional.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)


def camera_cube_observation(env):
    """Global RGB-D estimate relative to TCP in robot-base axes, plus visibility.

    Output: dx, dy, dz, sin(4*yaw), cos(4*yaw), confidence, valid.
    A missing/occluded cube produces zeros and valid=0, never simulator pose.
    Robot proprioception and calibrated camera extrinsics are permitted inputs.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    from isaac_vision_rl.perception.cube import YellowCubeEstimator

    def tensor(value):
        return value.torch if hasattr(value, "torch") else value

    counter = getattr(env, "common_step_counter", -1)
    episodes = env.episode_length_buf
    cache = getattr(env, "_camera_cube_cache", None)
    if cache is not None and cache[0] == counter and torch.equal(cache[1], episodes):
        return cache[2]
    camera, robot = env.scene["camera"], env.scene["robot"]
    camera.update(env.step_dt, force_recompute=True)
    rgb = tensor(camera.data.output["rgb"])[..., :3].detach().cpu().numpy().astype(np.uint8)
    depth = tensor(camera.data.output["distance_to_image_plane"]).detach().cpu().numpy()
    K = tensor(camera.data.intrinsic_matrices).detach().cpu().numpy()
    camera_pos = tensor(camera.data.pos_w).detach().cpu().numpy()
    camera_rot = Rotation.from_quat(tensor(camera.data.quat_w_ros).detach().cpu().numpy()).as_matrix()
    base_pos = tensor(robot.data.root_pos_w).detach().cpu().numpy()
    base_rot = Rotation.from_quat(tensor(robot.data.root_quat_w).detach().cpu().numpy()).as_matrix()
    body = robot.find_bodies("gripper_frame_link")[0][0]
    tcp = tensor(robot.data.body_pos_w)[:, body].detach().cpu().numpy()
    values = np.zeros((env.num_envs, 7), dtype=np.float32)
    for i in range(env.num_envs):
        transform = np.eye(4)
        transform[:3, :3] = base_rot[i].T @ camera_rot[i]
        transform[:3, 3] = base_rot[i].T @ (camera_pos[i] - base_pos[i])
        estimate = YellowCubeEstimator(transform).estimate_depth(rgb[i], depth[i].squeeze(), K[i])
        if estimate.valid and estimate.center_base_m is not None and estimate.yaw_mod_quarter_turn_rad is not None:
            tcp_base = base_rot[i].T @ (tcp[i] - base_pos[i])
            values[i, :3] = np.asarray(estimate.center_base_m) - tcp_base
            yaw = float(estimate.yaw_mod_quarter_turn_rad)
            values[i, 3:] = [np.sin(4 * yaw), np.cos(4 * yaw), estimate.confidence, 1.0]
    result = torch.tensor(values, device=env.device)
    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log["Perception/cube_visible_fraction"] = result[:, 6].mean()
        log["Perception/cube_confidence"] = result[:, 5].mean()
    env._camera_cube_cache = (counter, episodes.clone(), result)
    return result


def time_remaining(env):
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0, 1).unsqueeze(-1)


def filtered_joint_targets(env):
    """Expose actuator filter memory so the policy can predict its next command."""
    return torch.cat(
        [env.action_manager.get_term(name).processed_actions for name in ("arm_action", "gripper_action")], dim=-1
    )


def grasp_assist_state(env):
    term = env.action_manager.get_term("gripper_action")
    return torch.stack((term.grasp_latched.float(), (term.closure_hold_steps.float() / 15).clamp(0, 1)), dim=-1)


def cube_ee_relative(env):
    """Privileged lift-baseline cue: cube position relative to the tool center.

    This is intentionally used only for the first state-assisted lift curriculum.
    A vision-only deployment policy should replace it with an estimator from RGB.
    """
    cube = env.scene["cube"]
    robot = env.scene["robot"]
    ee_idx = robot.find_bodies("gripper_frame_link")[0][0]
    return cube.data.root_pos_w.torch - robot.data.body_pos_w.torch[:, ee_idx]


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
