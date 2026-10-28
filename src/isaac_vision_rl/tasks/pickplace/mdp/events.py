"""Workcell presentation events and the held-start reset used to practice carrying."""

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


def select_task_camera_view(env, env_ids=None):
    """Use the existing sensor camera in the GUI instead of an orbit camera.

    Headless runs have no viewport and require no presentation changes.
    """
    import os
    if os.environ.get('SO101_REWARD_PROFILE')=='four-stage':
        from isaac_vision_rl.utils.newton_colors import apply_newton_colors
        for visualizer in env.sim.visualizers:
            viewer=getattr(visualizer,'_viewer',None)
            if viewer is not None and getattr(viewer,'model',None) is not None:
                apply_newton_colors(viewer)
    # Newton's GL camera is world-space. Offset by env_0's origin so a grid
    # centered elsewhere does not leave the camera looking between workcells.
    for visualizer in env.sim.visualizers:
        if visualizer.cfg.visualizer_type in ("newton", "newton_gl") and not getattr(
            visualizer, "_task_camera_selected", False
        ):
            origin = env.scene.env_origins[0]
            if hasattr(origin, "torch"):
                origin = origin.torch
            origin = origin.tolist()
            cfg = env.cfg.sim.default_visualizer_cfg
            visualizer.set_camera_view(
                tuple(a + b for a, b in zip(origin, cfg.eye, strict=True)),
                tuple(a + b for a, b in zip(origin, cfg.lookat, strict=True)),
            )
            visualizer._task_camera_selected = True
    if not env.sim.has_gui:
        return
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        return
    camera_path = env.scene.env_prim_paths[0] + "/Camera"
    if not env.sim.stage.GetPrimAtPath(camera_path).IsValid():
        raise RuntimeError(f"Task camera does not exist: {camera_path}")
    viewport.set_active_camera(camera_path)


def solve_pan_offset(base_xy, cube_xy, box_xy, target_distance, pan_sign, max_turn=1.6, steps=321):
    """Shoulder-pan change (rad) that puts the held cube ``target_distance`` from the box, or as near as it can.

    The held cube turns with the arm about the robot base. ``pan_sign`` is +1 when a positive pan turns the arm
    counter-clockwise seen from above. Of two possible turns the smaller one is chosen.
    """
    turns = torch.linspace(-max_turn, max_turn, steps, device=base_xy.device)
    angle = turns * pan_sign
    cos, sin = torch.cos(angle), torch.sin(angle)
    rel = cube_xy - base_xy
    x = rel[:, None, 0] * cos - rel[:, None, 1] * sin
    y = rel[:, None, 0] * sin + rel[:, None, 1] * cos
    new_xy = base_xy[:, None, :] + torch.stack([x, y], dim=-1)
    distance = torch.norm(new_xy - box_xy[:, None, :], dim=-1)
    cost = (distance - target_distance[:, None]).abs() + 1e-3 * turns.abs()
    return turns[cost.argmin(dim=1)]


class HeldStartReset(ManagerTermBase):
    """Start some tries with the cube already held at lift height, a chosen distance from the box.

    States come from ``scripts/collect_held_states.py``. Each saved hold is turned about the robot base, so the
    grip on the cube is kept exactly, and only the distance to the box changes. This runs after the normal reset
    events, and before Isaac Lab syncs the arm filter memory to the joint positions written here.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        data = np.load(cfg.params["states_path"])
        device = env.device
        self._joint = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self._cube_pos = torch.as_tensor(data["cube_pos_local"], dtype=torch.float32, device=device)
        self._cube_quat = torch.as_tensor(data["cube_quat"], dtype=torch.float32, device=device)
        self._pan_sign = float(data["pan_sign"])
        self._pan_index = int(data["pan_joint_index"])
        if len(self._joint) == 0:
            raise ValueError("The held-state file has no states")

    def __call__(
        self,
        env,
        env_ids,
        states_path: str,
        fraction: float = 0.3,
        min_distance: float = 0.07,
        max_distance: float = 0.17,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ) -> None:
        ids = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
        if len(ids) == 0:
            return
        robot, cube, box = env.scene[robot_cfg.name], env.scene[cube_cfg.name], env.scene[box_cfg.name]
        origins = env.scene.env_origins
        origins = (origins.torch if hasattr(origins, "torch") else origins)[ids]

        pick = torch.randint(0, len(self._joint), (len(ids),), device=env.device)
        joint = self._joint[pick].clone()
        cube_pos = self._cube_pos[pick] + origins
        cube_quat = self._cube_quat[pick]

        base_xy = robot.data.root_pos_w.torch[ids, :2]
        box_xy = box.data.root_pos_w.torch[ids, :2]
        target = min_distance + (max_distance - min_distance) * torch.rand(len(ids), device=env.device)
        turn = solve_pan_offset(base_xy, cube_pos[:, :2], box_xy, target, self._pan_sign)
        angle = turn * self._pan_sign

        rel = cube_pos[:, :2] - base_xy
        cos, sin = torch.cos(angle), torch.sin(angle)
        cube_pos[:, 0] = base_xy[:, 0] + rel[:, 0] * cos - rel[:, 1] * sin
        cube_pos[:, 1] = base_xy[:, 1] + rel[:, 0] * sin + rel[:, 1] * cos
        zeros = torch.zeros_like(angle)
        cube_quat = quat_mul(quat_from_euler_xyz(zeros, zeros, angle), cube_quat)

        joint[:, self._pan_index] += turn
        limits = robot.data.soft_joint_pos_limits.torch[ids]
        joint = joint.clamp(limits[..., 0], limits[..., 1])

        robot.write_joint_position_to_sim_index(position=joint, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint), env_ids=ids)
        robot.set_joint_position_target_index(target=joint, env_ids=ids)
        cube.write_root_pose_to_sim_index(root_pose=torch.cat([cube_pos, cube_quat], dim=-1), env_ids=ids)
        cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
