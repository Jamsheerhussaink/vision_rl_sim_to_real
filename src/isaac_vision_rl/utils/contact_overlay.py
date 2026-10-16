"""Draw historical pad reference boxes and actual contacts on video copies."""

import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def project_points(points, camera_position, camera_quaternion_xyzw, intrinsics):
    """Project world points using ROS camera axes (+Z forward, +Y down)."""
    rotation = Rotation.from_quat(camera_quaternion_xyzw).as_matrix()
    camera_points = (np.asarray(points) - camera_position) @ rotation
    valid = camera_points[:, 2] > 0.001
    pixels = np.full((len(points), 2), np.nan)
    projected = camera_points[valid] @ np.asarray(intrinsics).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


class ContactOverlay:
    """Use the USD pad bounds and live body poses; preserve physics and RGB input."""

    def __init__(self, usd_path, robot):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Cannot open collision asset: {usd_path}")
        cache = UsdGeom.XformCache()
        signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
        self.edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                      if np.count_nonzero(signs[a] != signs[b]) == 1]
        self.pads = []
        for name, body_name, color, sensor in (
            ("static_pad", "gripper_link", (255, 255, 0), "fixed_finger_contact"),
            ("moving_pad", "moving_jaw_so101_v1_link", (255, 0, 255), "moving_finger_contact"),
        ):
            prim = next((p for p in stage.Traverse() if p.GetName() == name), None)
            if prim is None:
                raise ValueError(f"Missing collision pad: {name}")
            body = prim.GetParent()
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
            matrix = np.asarray(relative).T
            corners = signs * float(prim.GetAttribute("size").Get()) / 2
            corners = (np.c_[corners, np.ones(8)] @ matrix.T)[:, :3]
            ids = robot.find_bodies(body_name)[0]
            if len(ids) != 1:
                raise ValueError(f"Expected one body for {body_name}")
            self.pads.append((name, ids[0], corners, color, sensor))

    @staticmethod
    def array(value):
        value = value.torch if hasattr(value, "torch") else value
        return value.detach().cpu().numpy()

    def draw(self, frame, robot, camera, forces, scene):
        import cv2

        positions = self.array(robot.data.body_pos_w)[0]
        quaternions = self.array(robot.data.body_quat_w)[0]
        eye = self.array(camera.data.pos_w)[0]
        orientation = self.array(camera.data.quat_w_ros)[0]
        intrinsics = self.array(camera.data.intrinsic_matrices)[0]
        diagnostics = {}
        for row, (name, body, corners, color, sensor) in enumerate(self.pads):
            rotation = Rotation.from_quat(quaternions[body]).as_matrix()
            world = corners @ rotation.T + positions[body]
            pixels, valid = project_points(world, eye, orientation, intrinsics)
            for a, b in self.edges:
                if valid[a] and valid[b]:
                    pa, pb = tuple(np.rint(pixels[a]).astype(int)), tuple(np.rint(pixels[b]).astype(int))
                    cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{name} ref: {forces.get(sensor, 0):.2f} N (body contact)",
                        (8, 40 + row * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            diagnostics[name] = {"corners_world_m": world.tolist(),
                                 "force_n": forces.get(sensor, 0)}
            contact_positions = scene[sensor].data.contact_pos_w
            if contact_positions is not None:
                points = self.array(contact_positions)[0].reshape(-1, 3)
                points = points[np.isfinite(points).all(axis=1)]
                # PhysX reports mean contact positions for each body/filter pair.
                diagnostics[name]["mean_contact_points_world_m"] = points.tolist()
                projected, visible = project_points(points, eye, orientation, intrinsics)
                for point in projected[visible]:
                    cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, color, -1)
        return diagnostics
