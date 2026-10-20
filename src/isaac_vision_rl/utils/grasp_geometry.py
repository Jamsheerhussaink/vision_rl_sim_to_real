"""Offline joint-chain geometry used by the physical grasp preflight."""

from typing import Any, cast

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def transform(pos, quat):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    result[:3, 3] = pos
    return result


def _usd_value(prim, name):
    attribute = prim.GetAttribute(name)
    if attribute is None or not attribute.IsValid():
        raise ValueError(f"Missing USD joint attribute {name} on {prim.GetPath()}")
    value = attribute.Get()
    if value is None:
        raise ValueError(f"Unauthored USD joint attribute {name} on {prim.GetPath()}")
    return value


class GraspGeometry:
    def __init__(self, usd_path, base_pos, base_quat):
        from pxr import Usd, UsdPhysics

        stage = cast(Any, Usd.Stage.Open(str(usd_path)))
        if stage is None:
            raise ValueError(f"Unable to open robot USD: {usd_path}")
        self.base = np.eye(4)
        self.base[:3, :3] = Rotation.from_quat(base_quat).as_matrix()
        self.base[:3, 3] = base_pos
        self.joints = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                self.joints[prim.GetName()] = (
                    transform(_usd_value(prim, "physics:localPos0"), _usd_value(prim, "physics:localRot0")),
                    np.linalg.inv(
                        transform(_usd_value(prim, "physics:localPos1"), _usd_value(prim, "physics:localRot1"))
                    ),
                )
        self.lower = []
        self.upper = []
        for name in ARM_NAMES:
            prim = next(p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint))
            # Isaac articulation soft limits use factor 1 by default in this task.
            self.lower.append(np.deg2rad(_usd_value(prim, "physics:lowerLimit")) + 0.01)
            self.upper.append(np.deg2rad(_usd_value(prim, "physics:upperLimit")) - 0.01)

    def forward(self, angles):
        result = self.base.copy()
        for name, angle in zip(ARM_NAMES, angles, strict=True):
            a, b = self.joints[name]
            spin = np.eye(4)
            spin[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
            result = result @ a @ spin @ b
        a, b = self.joints["gripper_frame_joint"]
        return result @ a @ b

    def solve(self, position, initial=None, downward=True, position_weight=1.0):
        if not np.isfinite(position_weight) or position_weight<=0:
            raise ValueError("position_weight must be positive and finite")
        position = np.asarray(position)

        def residual(q):
            pose = self.forward(q)
            error = position_weight * (pose[:3, 3] - position)
            downward_align = 0.08 * (pose[:3, 2] - [0, 0, -1])
            roll_penalty = 0.1 * q[4]
            return np.r_[error, downward_align, roll_penalty] if downward else error

        rng = np.random.default_rng(42)
        starts = [np.zeros(5)] if initial is None else [np.asarray(initial)]
        starts += [rng.uniform(self.lower, self.upper) for _ in range(12)]
        solutions = [
            least_squares(residual, np.clip(x, self.lower, self.upper), bounds=(self.lower, self.upper), max_nfev=150)
            for x in starts
        ]
        answer = min(solutions, key=lambda x: np.linalg.norm(x.fun))
        pose = self.forward(answer.x)
        error = np.linalg.norm(pose[:3, 3] - position)
        alignment = np.dot(pose[:3, 2], [0, 0, -1])
        if error > 0.005 or (downward and alignment < 0.95):
            raise ValueError(
                f"Grasp target unreachable: position error {error:.4f} m, downward alignment {alignment:.3f}"
            )
        return answer.x
