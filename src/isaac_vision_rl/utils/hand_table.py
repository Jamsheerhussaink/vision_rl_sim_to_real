"""Measured collision-mesh clearance and filtered hand/table normal forces."""
import torch
from isaaclab.utils.math import matrix_from_quat

TABLE_SENSORS = ("fixed_table_contact", "moving_table_contact", "wrist_table_contact")


def hand_table_metrics(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_hand_table_vertices"):
        from scipy.spatial import ConvexHull
        from isaac_vision_rl.utils.table_clearance import HandTableClearance

        cfg = robot.cfg
        geometry = HandTableClearance(cfg.spawn.usd_path, cfg.init_state.pos, cfg.init_state.rot)
        # The minimum of a linear projection occurs at a convex-hull vertex.
        env._hand_table_vertices = [
            (robot.find_bodies(name)[0][0], torch.tensor(
                points[ConvexHull(points[:, :3]).vertices, :3], device=env.device, dtype=torch.float32
            )) for name, points in geometry.points.items()
        ]
    minimum = torch.full((env.num_envs,), float("inf"), device=env.device)
    origins = env.scene.env_origins
    origins = origins.torch if hasattr(origins, "torch") else origins
    for body, points in env._hand_table_vertices:
        rotation = matrix_from_quat(robot.data.body_quat_w.torch[:, body])
        heights = torch.einsum("nj,pj->np", rotation[:, 2, :], points)
        heights += robot.data.body_pos_w.torch[:, body, 2, None]
        minimum = torch.minimum(minimum, heights.min(dim=1).values - origins[:, 2] - 0.40)
    force = torch.zeros_like(minimum)
    for name in TABLE_SENSORS:
        value = env.scene[name].data.normal_force_matrix_w
        if value is None:
            raise RuntimeError("Filtered hand/table forces are required")
        value = value.torch if hasattr(value, "torch") else value
        force += torch.linalg.vector_norm(value.reshape(env.num_envs, -1, 3), dim=-1).sum(dim=1)
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update({
            "Safety/min_hand_table_clearance_m": minimum.min(),
            "Safety/hand_table_contact_fraction": (force > 0.02).float().mean(),
        })
    return minimum, force
