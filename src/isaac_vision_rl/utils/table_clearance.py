"""Check the actual wrist/finger collision vertices against the flat tabletop."""

import numpy as np
from scipy.spatial.transform import Rotation

from isaac_vision_rl.utils.calibration import JOINT_ORDER
from isaac_vision_rl.utils.grasp_geometry import transform


class HandTableClearance:
    def __init__(self, usd_path, base_pos, base_quat, table_z=0.40):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))
        cache = UsdGeom.XformCache()
        self.table_z = table_z
        self.base = np.eye(4)
        self.base[:3, :3] = Rotation.from_quat(base_quat).as_matrix()
        self.base[:3, 3] = base_pos
        self.joints = []
        children = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                parent = str(prim.GetRelationship("physics:body0").GetTargets()[0])
                child = str(prim.GetRelationship("physics:body1").GetTargets()[0])
                children.add(child)
                a = transform(prim.GetAttribute("physics:localPos0").Get(), prim.GetAttribute("physics:localRot0").Get())
                b = transform(prim.GetAttribute("physics:localPos1").Get(), prim.GetAttribute("physics:localRot1").Get())
                axis = prim.GetAttribute("physics:axis").Get() or "Z"
                self.joints.append((prim.GetName(), parent, child, a, np.linalg.inv(b), axis))
        self.root = next(parent for _, parent, *_ in self.joints if parent not in children)
        self.points = {}
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.GetName() not in ("wrist_link", "gripper_link", "moving_jaw_so101_v1_link"):
                continue
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            relative = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            local = np.c_[points, np.ones(len(points))] @ relative.T
            self.points.setdefault(body.GetName(), []).append(local)
        self.points = {name: np.concatenate(parts) for name, parts in self.points.items()}
        if set(self.points) != {"wrist_link", "gripper_link", "moving_jaw_so101_v1_link"}:
            raise ValueError("Missing wrist/finger collision geometry; refusing unverified clearance")

    def body_poses(self, angles):
        values = dict(zip(JOINT_ORDER, angles, strict=True))
        poses = {self.root: self.base}
        pending = list(self.joints)
        while pending:
            progress = False
            for item in pending[:]:
                name, parent, child, a, b, axis = item
                if parent not in poses:
                    continue
                spin = np.eye(4)
                direction = np.eye(3)["XYZ".index(axis)]
                spin[:3, :3] = Rotation.from_rotvec(direction * values.get(name, 0.0)).as_matrix()
                poses[child] = poses[parent] @ a @ spin @ b
                pending.remove(item)
                progress = True
            if not progress:
                raise ValueError("Disconnected robot joint graph")
        return {path.rsplit("/", 1)[-1]: pose for path, pose in poses.items()}

    def minimum(self, angles):
        poses = self.body_poses(angles)
        return min(float((points @ poses[name].T)[:, 2].min()) for name, points in self.points.items()) - self.table_z

    def measured_minimum(self, positions, quaternions):
        """Use PhysX measured link poses, rather than the commanded trajectory."""
        minimum = float("inf")
        for name, points in self.points.items():
            rotation = Rotation.from_quat(quaternions[name]).as_matrix()
            z = (points[:, :3] @ rotation.T + positions[name])[:, 2].min()
            minimum = min(minimum, float(z) - self.table_z)
        return minimum

    def check_segment(self, start, end, margin=0.01, samples=121):
        minimum = min(self.minimum((1 - t) * np.asarray(start) + t * np.asarray(end)) for t in np.linspace(0, 1, samples))
        if minimum < margin:
            raise ValueError(f"Hand/table clearance {1000*minimum:.1f} mm is below {1000*margin:.1f} mm")
        return minimum
