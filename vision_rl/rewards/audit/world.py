"""Toy kinematic world that produces `PickInputs` for scripted strategies.

Positions are world metres, matching scene.xml: table top z=0.40, cube 5 cm.
The gripper is a point that moves at most `max_speed` per step. The cube
follows the gripper while the jaw is closed on it, otherwise it falls to
whatever it rests on. A few `events` model physical tricks the arm could
pull off in the real sim (tipping the cube onto an edge, leaving it leaning
on something).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME
from vision_rl.envs import SCENE_XML

TABLE_Z = 0.40
CUBE_HALF = 0.025
CUBE_REST_Z = TABLE_Z + CUBE_HALF
CUBE_START = np.array([0.274, -0.128, CUBE_REST_Z])
READY_POS = np.array([0.22, -0.08, 0.54])     # gripper "ready" pose, gripper + wrist inside the view
TILT_RISE = CUBE_HALF * (np.sqrt(2.0) - 1.0)  # centre rise when the cube stands on an edge
GRAVITY_STEP = 9.81 * 0.03 ** 2               # dz velocity change per 30 ms control step


class Camera:
    """Pinhole projection of world points through the scene's main camera."""

    def __init__(self, name: str = CAMERA_NAME, margin: float = 0.05):
        m = mujoco.MjModel.from_xml_path(SCENE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert cid >= 0, name
        self.pos = d.cam_xpos[cid].copy()
        self.rot = d.cam_xmat[cid].reshape(3, 3).copy()
        self.tan_y = np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
        self.tan_x = self.tan_y * CAM_WIDTH / CAM_HEIGHT
        self.margin = margin

    def uv(self, p) -> np.ndarray:
        """Normalised image coords, [-1, 1] inside the frame (MuJoCo cameras look along -z).

        Clipped to [-2, 2]; points behind the camera map to (2, 2).
        """
        q = self.rot.T @ (np.asarray(p) - self.pos)
        if -q[2] < 0.01:
            return np.array([2.0, 2.0])
        return np.clip([q[0] / -q[2] / self.tan_x, q[1] / -q[2] / self.tan_y], -2.0, 2.0)

    def visibility(self, points) -> tuple[float, float]:
        """(overshoot past the safe margin, number of points outside the frame).

        Overshoot is capped at 1 per coordinate so one bad point can't swamp the reward.
        """
        excess, out = 0.0, 0.0
        for p in points:
            a = np.abs(self.uv(p))
            excess += float(np.clip(a - (1.0 - self.margin), 0.0, 1.0).sum())
            out += float((a > 1.0).any())
        return excess, out


@dataclass
class ToyWorld:
    camera: Camera
    max_speed: float = 0.01
    grip: np.ndarray = field(default_factory=lambda: READY_POS.copy())
    cube: np.ndarray = field(default_factory=lambda: CUBE_START.copy())
    cube_vz: float = 0.0
    support_z: float = CUBE_REST_Z              # where the cube rests when let go
    attached: bool = False
    enclosed: bool = False                      # open jaw straddles the cube (last step)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def step(self, target, jaw_open: bool, event: str | None = None):
        """Move one control step; return the dict of reward inputs."""
        old = self.grip.copy()
        delta = np.asarray(target, float) - self.grip
        dist = np.linalg.norm(delta)
        if dist > self.max_speed:
            delta *= self.max_speed / dist
        new = self.grip + delta
        new[2] = max(new[2], TABLE_Z + 0.005)

        # Grasp only when an OPEN jaw that straddled the cube last step now closes on it.
        if not jaw_open and self.enclosed and not self.attached:
            self.attached, self.offset = True, self.cube - self.grip

        # A closed, empty jaw can't pass through the cube: from above it rests on the top
        # face, from the side it pushes the cube along the table.
        top = self.cube[2] + CUBE_HALF
        if not self.attached and not jaw_open:
            gap = new[:2] - self.cube[:2]
            if np.linalg.norm(gap) < CUBE_HALF + 0.005:
                if old[2] >= top:
                    new[2] = max(new[2], top + 0.002)
                elif new[2] < top:
                    away = gap / max(np.linalg.norm(gap), 1e-6)
                    self.cube[:2] = new[:2] - away * (CUBE_HALF + 0.005)
        self.grip = new

        if event == "tilt":            # knock the cube onto one edge
            self.support_z = CUBE_REST_Z + TILT_RISE
        elif event == "ledge":         # leave the cube leaning on the box wall / robot base
            self.support_z = CUBE_REST_Z + 0.04
        elif event == "floor":         # cube back flat on the table
            self.support_z = CUBE_REST_Z

        d_xy = np.linalg.norm(self.grip[:2] - self.cube[:2])
        dz = self.grip[2] - self.cube[2]
        self.enclosed = jaw_open and d_xy < 0.015 and abs(dz) < 0.02
        if jaw_open and self.attached:
            self.attached = False
            self.cube_vz = (self.grip[2] - old[2])     # released cube keeps the arm's upward speed

        if self.attached:
            self.cube = self.grip + self.offset
            self.cube_vz = 0.0
        else:                                          # ballistic fall onto the support
            self.cube_vz -= GRAVITY_STEP
            self.cube[2] = self.cube[2] + self.cube_vz
            if self.cube[2] <= self.support_z:
                self.cube[2], self.cube_vz = self.support_z, 0.0

        d_xy = np.linalg.norm(self.grip[:2] - self.cube[:2])
        dz = self.grip[2] - self.cube[2]
        vel = (self.grip - old) / self.max_speed
        action = np.array([*vel, 0.0, 0.0, 1.0 if jaw_open else -1.0])
        # Two robot points: the gripper and a wrist point 8 cm above it.
        excess, out = self.camera.visibility([self.grip, self.grip + [0, 0, 0.08]])
        return dict(
            d_xy=d_xy, dz=dz, cube_lift=self.cube[2] - CUBE_REST_Z,
            gripper_h=self.grip[2] - TABLE_Z, grip_open=float(jaw_open),
            jaw_stalled=float(self.attached and not jaw_open),
            home_dist=np.linalg.norm(self.grip - READY_POS),
            vis_excess=excess, vis_out=out, near_limit=0.0, action=action,
        )
