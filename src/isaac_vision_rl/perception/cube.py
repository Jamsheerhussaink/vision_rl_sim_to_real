"""Known-size upright yellow cube localization from RGB and calibrated 3D points."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CubeEstimate:
    valid: bool
    reason: str
    center_base_m: tuple[float, float, float] | None = None
    yaw_mod_quarter_turn_rad: float | None = None
    confidence: float = 0.0
    points: int = 0

    def to_dict(self):
        return asdict(self)


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Camera-to-base transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)
    ):
        raise ValueError("Camera-to-base transform must contain a proper rotation and homogeneous last row")
    return matrix


def load_calibration(path, require_measured=True):
    data = json.loads(Path(path).read_text())
    if (
        data.get("units") != "metres"
        or data.get("camera_frame") != "color_optical"
        or data.get("base_z_up") is not True
    ):
        raise ValueError("Calibration requires metres, color_optical camera frame, and upright robot-base Z")
    if require_measured and data.get("source") != "measured":
        raise ValueError(
            "Live hardware requires measured camera-to-base calibration; simulation transforms are rejected"
        )
    return validate_transform(data["T_base_camera"])


class YellowCubeEstimator:
    """Fit the visible top face; reject missing/ambiguous/tilted or partial geometry.

    Assumes an upright 5 cm yellow cube and a robot base with Z up. Confidence is
    a geometry heuristic, not a calibrated probability. No last-position or GT fallback.
    """

    def __init__(self, T_base_camera, size_m=0.05, min_points=20):
        self.transform = validate_transform(T_base_camera)
        if size_m <= 0:
            raise ValueError("Cube size must be positive")
        self.size = size_m
        self.min_points = min_points

    def estimate_depth(self, rgb, depth_z_m, K):
        depth = np.asarray(depth_z_m, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        if depth.shape != rgb.shape[:2] or K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Aligned RGB/depth and finite positive-focal-length intrinsics are required")
        # This convenience path is for undistorted pinhole images, including simulation.
        mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), np.array([18, 90, 75]), np.array([40, 255, 255]))
        v, u = np.nonzero(mask)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(K).T
        points = rays * depth[v, u, None]
        return self.estimate_cloud(rgb, points, np.stack((u, v), axis=-1))

    def estimate_cloud(self, rgb, points_color_m, pixels_color):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB must be an HxWx3 uint8 image")
        xyz = np.asarray(points_color_m, dtype=np.float64)
        pixels = np.asarray(pixels_color, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or pixels.shape != (len(xyz), 2):
            raise ValueError("Point cloud must be Nx3 with matching Nx2 color pixels")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([18, 90, 75]), np.array([40, 255, 255]))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(pixels).all(axis=1)
        h, w = rgb.shape[:2]
        finite &= (xyz[:, 2] > 0.1) & (xyz[:, 2] < 3)
        finite &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        xyz = xyz[finite]
        uv = pixels[finite].astype(int)
        if not len(xyz):
            return CubeEstimate(False, "no valid depth points")
        component = labels[uv[:, 1], uv[:, 0]]
        # A gripper can split one cube's visible top and front into separate
        # image components. Merge only nearby 3D patches, then require the full
        # known-size footprint; do not infer missing geometry from ground truth.
        patches = []
        for label in range(1, count):
            cloud = xyz[component == label]
            if stats[label, cv2.CC_STAT_AREA] >= self.min_points and len(cloud) >= self.min_points:
                patches.append(cloud @ self.transform[:3, :3].T + self.transform[:3, 3])
        groups = []
        for patch in patches:
            matches = [
                i
                for i, g in enumerate(groups)
                if np.linalg.norm(np.median(g, axis=0) - np.median(patch, axis=0)) < self.size * 1.75
            ]
            if not matches:
                groups.append(patch)
            else:
                merged = np.concatenate([patch] + [groups[i] for i in matches])
                groups = [g for i, g in enumerate(groups) if i not in matches] + [merged]
        candidates = []
        for base in groups:
            top_level = np.percentile(base[:, 2], 85)
            top = base[np.abs(base[:, 2] - top_level) < 0.003]
            if len(top) < self.min_points:
                continue
            # Verify that the top is horizontal; do not extrapolate a tilted held cube.
            A = np.c_[top[:, :2], np.ones(len(top))]
            slope = np.linalg.lstsq(A, top[:, 2], rcond=None)[0]
            residual = np.sqrt(np.mean((A @ slope - top[:, 2]) ** 2))
            if np.linalg.norm(slope[:2]) > 0.18 or residual > 0.003:
                continue
            surface = base[(base[:, 2] > top_level - self.size * 0.95) & (base[:, 2] < top_level + 0.005)]
            rectangle = cv2.minAreaRect(surface[:, :2].astype(np.float32))
            widths = np.array(rectangle[1])
            ratios = widths / self.size
            if np.any(ratios < 0.72) or np.any(ratios > 1.25):
                continue
            z = float(np.median(top[:, 2]) - self.size / 2)
            center = (float(rectangle[0][0]), float(rectangle[0][1]), z)
            confidence = float(np.clip(1 - np.max(np.abs(ratios - 1)) - residual / 0.01, 0, 1))
            candidates.append(
                CubeEstimate(
                    True,
                    "upright visible-surface fit",
                    center,
                    np.deg2rad(rectangle[2]) % (np.pi / 2),
                    confidence,
                    len(top),
                )
            )
        if len(candidates) != 1:
            return CubeEstimate(
                False, "ambiguous yellow cubes" if len(candidates) > 1 else "no complete upright cube top face"
            )
        return candidates[0]


def fit_camera_to_base(camera_points, base_points):
    """Rigid fit from >=6 matched non-collinear measured landmarks, in metres."""
    camera = np.asarray(camera_points, dtype=np.float64)
    base = np.asarray(base_points, dtype=np.float64)
    if camera.shape != base.shape or camera.ndim != 2 or camera.shape[1] != 3 or len(camera) < 6:
        raise ValueError("Use at least six matched Nx3 landmark pairs")
    if not np.isfinite(camera).all() or not np.isfinite(base).all():
        raise ValueError("Landmarks must be finite")
    c, b = camera.mean(0), base.mean(0)
    centered = camera - c
    if np.linalg.matrix_rank(centered, tol=1e-6) < 2 or np.linalg.matrix_rank(base - b, tol=1e-6) < 2:
        raise ValueError("Landmarks must not be collinear")
    U, _, Vt = np.linalg.svd(centered.T @ (base - b))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = b - R @ c
    residual = np.linalg.norm(camera @ R.T + T[:3, 3] - base, axis=1)
    if max(residual) > 0.005:
        raise ValueError(f"Landmark fit error {max(residual):.4f} m exceeds 5 mm; check measurements")
    return validate_transform(T), residual
