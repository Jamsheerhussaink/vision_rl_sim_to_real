from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def get_camera_projection_params(env: ManagerBasedRLEnv):
    """Retrieve camera pose, intrinsics, and dimensions with workcell defaults."""
    camera = env.scene["camera"] if hasattr(env.scene, "keys") and "camera" in env.scene.keys() else None
    device = env.device if hasattr(env, "device") else "cpu"
    num_envs = env.num_envs if hasattr(env, "num_envs") else 1

    if camera is not None and hasattr(camera, "data") and hasattr(camera.data, "pos_w"):
        cam_pos = camera.data.pos_w.torch if hasattr(camera.data.pos_w, "torch") else camera.data.pos_w
        cam_quat = camera.data.quat_w_ros.torch if hasattr(camera.data.quat_w_ros, "torch") else camera.data.quat_w_ros
        K = camera.data.intrinsic_matrices.torch if hasattr(camera.data.intrinsic_matrices, "torch") else camera.data.intrinsic_matrices
        if hasattr(camera, "image_shape") and camera.image_shape is not None:
            img_h, img_w = camera.image_shape
        else:
            img_w = getattr(camera.cfg, "width", 1280)
            img_h = getattr(camera.cfg, "height", 720)
    else:
        # Fallback to nominal workcell camera configuration:
        # OpenGL pos (-0.05, -0.22, 0.77), ROS quat (0.9497, 0.0, 0.0, -0.3132)
        cam_pos = torch.tensor([[-0.05, -0.22, 0.77]], device=device, dtype=torch.float32).expand(num_envs, 3)
        cam_quat = torch.tensor([[0.9497, 0.0, 0.0, -0.3132]], device=device, dtype=torch.float32).expand(num_envs, 4)
        K = torch.tensor([
            [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]
        ], device=device, dtype=torch.float32).expand(num_envs, 3, 3)
        img_w, img_h = 1280, 720

    return cam_pos, cam_quat, K, float(img_w), float(img_h)


def points_outside_camera_view(
    points_w: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    K: torch.Tensor,
    img_w: float,
    img_h: float,
    margin: float = 0.02,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
) -> torch.Tensor:
    """Vectorized check whether points in world coordinates lie outside camera frustum."""
    if points_w.ndim == 2:
        points_w = points_w.unsqueeze(1)  # (N, 1, 3)
    N, P, _ = points_w.shape

    delta_w = points_w - cam_pos.unsqueeze(1)  # (N, P, 3)
    delta_w_flat = delta_w.reshape(-1, 3)
    quat_rep = cam_quat.unsqueeze(1).expand(N, P, 4).reshape(-1, 4)
    v_cam = quat_apply_inverse(quat_rep, delta_w_flat).view(N, P, 3)

    x_cam = v_cam[..., 0]
    y_cam = v_cam[..., 1]
    z_cam = v_cam[..., 2]

    valid_depth = (z_cam >= min_depth) & (z_cam <= max_depth)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    z_safe = torch.where(z_cam > 0.01, z_cam, torch.ones_like(z_cam))
    u = fx * (x_cam / z_safe) + cx
    v = fy * (y_cam / z_safe) + cy

    u_norm = u / img_w
    v_norm = v / img_h

    inside = (
        valid_depth
        & (u_norm >= margin)
        & (u_norm <= 1.0 - margin)
        & (v_norm >= margin)
        & (v_norm <= 1.0 - margin)
    )
    return (~inside).any(dim=-1)


def gripper_outside_camera_view(
    env: ManagerBasedRLEnv,
    margin: float = 0.02,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
    body_names: Sequence[str] = ("gripper_frame_link", "gripper_link", "moving_jaw_so101_v1_link"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's gripper (TCP and fingers) leaves the camera's field of view."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_positions = []
    for name in body_names:
        body_id = robot.find_bodies(name)[0][0]
        body_positions.append(robot.data.body_pos_w.torch[:, body_id])
    pts_w = torch.stack(body_positions, dim=1)  # (N, len(body_names), 3)

    cam_pos, cam_quat, K, img_w, img_h = get_camera_projection_params(env)
    return points_outside_camera_view(
        pts_w, cam_pos, cam_quat, K, img_w, img_h, margin=margin, min_depth=min_depth, max_depth=max_depth
    )


def object_outside_camera_view(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    margin: float = 0.01,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
) -> torch.Tensor:
    """Terminate when a tracked object (e.g. cube) leaves the camera's field of view."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pts_w = asset.data.root_pos_w.torch.unsqueeze(1)  # (N, 1, 3)
    cam_pos, cam_quat, K, img_w, img_h = get_camera_projection_params(env)
    return points_outside_camera_view(
        pts_w, cam_pos, cam_quat, K, img_w, img_h, margin=margin, min_depth=min_depth, max_depth=max_depth
    )
