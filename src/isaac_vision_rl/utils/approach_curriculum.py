"""Reset-only difficulty progression; the policy controls every subsequent action."""
import torch


def sample_start_indices(count, progress, bank_size, device):
    progress = min(1.0, max(0.0, float(progress)))
    kind = torch.rand(count, device=device)
    maximum = max(1, round((bank_size - 1) * max(0.1, progress)))
    indices = torch.randint(0, maximum + 1, (count,), device=device)
    indices[kind < 0.2] = 0  # Retain local grasp practice.
    indices[kind >= 0.9 - 0.4 * progress] = bank_size - 1  # 10% -> 50% home starts.
    return indices


def reset_approach_curriculum(env, env_ids, joint_names, start_poses, ramp_steps):
    robot = env.scene["robot"]
    if not getattr(env, "_approach_starts_verified", False):
        # Load USD geometry only after Kit has launched, avoiding mixed USD libraries.
        from isaac_vision_rl.utils.table_clearance import HandTableClearance
        cfg = robot.cfg
        guard = HandTableClearance(cfg.spawn.usd_path, cfg.init_state.pos, cfg.init_state.rot)
        if min(guard.minimum(p) for p in start_poses) < 0.005:
            raise ValueError("Approach reset poses do not satisfy 5 mm nominal table clearance")
        env._approach_starts_verified = True
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    progress = min(1.0, env.common_step_counter / ramp_steps)
    indices = sample_start_indices(len(env_ids), progress, len(start_poses), env.device)
    values = torch.tensor(start_poses, device=env.device, dtype=torch.float32)[indices]
    joint_ids = [robot.find_joints(name)[0][0] for name in joint_names]
    positions = robot.data.default_joint_pos.torch[env_ids].clone()
    positions[:, joint_ids] = values
    robot.write_joint_position_to_sim_index(position=positions, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions), env_ids=env_ids)
    robot.set_joint_position_target(positions, env_ids=env_ids)
