"""SO-101 top-down PICK task for one world, on MJX-Warp (GPU).

`reset(key)` and `step(state, action)` are pure JAX functions for ONE world;
`VecEnv` (vec_env.py) vmaps them over all worlds, adds camera images, frame
stacking and auto-reset. The reward is `vision_rl.rewards.pick_reward`; this
file only measures the physics numbers it needs (`PickInputs`).

Action, 6 numbers in [-1, 1]:
  [0:5] arm joint goals: ready_pose + a * half_range. The joint target moves
        `smooth_alpha` of the way there each step, at most `max_joint_delta` rad,
        and never more than `windup_margin` from the real joint angle
        (a = 0 is the ready pose over the cube area, so early random actions
        explore the useful space).
  [5]   jaw: > +jaw_hysteresis opens, < -jaw_hysteresis closes, in between keeps the
        current state. Binary on purpose: with a proportional jaw, "slightly closed"
        parks the jaw half-open and still looks "stalled".

Observation (proprio, no cube position: the policy must find it in the image):
  arm+jaw qpos (6), qvel * 0.1 (6), current joint targets (6),
  grasp point xyz (3, from forward kinematics, available on the real robot),
  time left in the episode (1). Time is in the observation so treating the
  time limit as the end of the episode is correct for the critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME
from vision_rl.envs.grasp_geometry import opposing_finger_sides
from vision_rl.envs.control import smooth_joint_targets
from vision_rl.rewards.pick import (PickInputs, PickRewardConfig, PickRewardState,
                                    init_pick_state, pick_reward)

N_ARM = 5
TABLE_Z = 0.40
VIS_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
VIS_MARGIN = 0.05            # same safe margin as rewards/audit Camera
LIMIT_MARGIN = 0.05          # rad, "near a joint limit"
STALL_MARGIN = 0.2           # rad above the fully-closed jaw = blocked by something


@dataclass(frozen=True)
class PickEnvConfig:
    ctrl_dt: float = 0.03                 # ~ the real camera's 30 fps
    episode_steps: int = 250              # 7.5 s
    max_joint_delta: float = 0.035        # rad per control step (1.17 rad/s, ~67 deg/s: smooth and responsive)
    smooth_alpha: float = 0.40            # each step moves this fraction of the way to the goal
                                          # (capped at max_joint_delta): steady speed, soft stop
    jaw_hysteresis: float = 0.3           # jaw opens above +h, closes below -h, else keeps its
                                          # state (a 0 threshold chattered ~56 times per episode)
    windup_margin: float = 0.1            # rad: joint target stays this close to the real joint.
                                          # Without it the target ran 1-1.8 rad into the table and
                                          # ~20 "up" steps in a row were needed before the arm moved.
    start_jaw_open: bool = True           # episodes start with the jaw open (it started closed and the policy
                                          # never opened it, so the top-down latch / held never fired)
    jaw_open: float = 1.2                 # rad, ~11 cm between the fingertips
    cube_range: float = 0.03              # cube xy spawn: keyframe spot +/- this (m)
    cube_yaw_range: float = np.pi / 4     # cube yaw spawn: +/- this
    joint_noise: float = 0.02             # rad, start pose noise
    ready_height: float = 0.12            # ready pose: grasp point this high over the cube spot
    # Solver: robot.xml has 10 / 20 (Madhan's), which Warp reports hitting during grasps.
    # Physics is ~2% of a training step (rendering dominates), so more is nearly free.
    solver_iterations: int = 30
    ls_iterations: int = 30
    reward: PickRewardConfig = field(default_factory=PickRewardConfig)


@struct.dataclass
class PickState:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array            # episode over (time limit, cube fell, or physics blew up)
    step: jax.Array
    cube_z0: jax.Array
    rstate: PickRewardState
    ep_return: jax.Array
    ep_success: jax.Array      # latched: success happened this episode
    ep_held: jax.Array         # latched: held happened this episode
    metrics: dict


def _ik_ready_pose(m: mujoco.MjModel, site: int, target: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Arm joints that put the grasp point at `target` with the fingers pointing down."""
    from scipy.optimize import least_squares

    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:4, 0] + LIMIT_MARGIN, m.jnt_range[:4, 1] - LIMIT_MARGIN

    def resid(x):
        d.qpos[:] = q0
        d.qpos[:4] = x
        mujoco.mj_kinematics(m, d)
        down = d.site_xmat[site].reshape(3, 3)[:, 0]
        return np.r_[(d.site_xpos[site] - target) / 0.001, (down - [0, 0, -1]) / 0.05]

    best = None
    for seed in ([0, 0, 0, 1.2], [0, -0.8, 0.8, 1.4], [0, 0.5, -0.5, 1.5], list(q0[:4])):
        s = np.array(seed, float)
        s[0] = np.arctan2(target[1], target[0] - m.body("base").pos[0])
        r = least_squares(resid, np.clip(s, lo + 1e-3, hi - 1e-3), bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    q = q0[:N_ARM].copy()
    q[:4] = best.x
    return q


class PickEnv:
    def __init__(self, mj_model: mujoco.MjModel, cfg: PickEnvConfig = PickEnvConfig(),
                 num_envs: int = 1, contacts_per_world: int = 64, njmax: int = 256):
        """Warp buffers: `contacts_per_world * num_envs` contacts shared by ALL worlds
        (random actions peak at ~32 per world), `njmax` constraint rows per world
        (random actions peak at ~183). Too small = Warp drops contacts and warns."""
        self.cfg = cfg
        m = self.mj_model = mj_model
        m.opt.iterations, m.opt.ls_iterations = cfg.solver_iterations, cfg.ls_iterations
        self.mjx_model = mjx.put_model(m, impl="warp")
        self.naconmax, self.njmax = contacts_per_world * num_envs, njmax
        self.n_substeps = max(1, round(cfg.ctrl_dt / m.opt.timestep))

        key = m.key("home").id
        self.home_qpos = np.asarray(m.key_qpos[key]).copy()
        self.home_ctrl = np.asarray(m.key_ctrl[key]).copy()
        cube_jnt = m.body("cube").jntadr[0]
        self.cube_q = int(m.jnt_qposadr[cube_jnt])
        self.cube_xy0 = self.home_qpos[self.cube_q:self.cube_q + 2].copy()
        self.cube_z0 = float(self.home_qpos[self.cube_q + 2])
        self.site = m.site("gripperframe").id
        self.cube_body = m.body("cube").id
        self.fixed_tip = m.geom("fixed_jaw_sph_tip1").id
        self.moving_tip = m.geom("moving_jaw_sph_tip1").id
        self.jaw_q = int(m.jnt_qposadr[m.joint("gripper").id])
        self.wrist_q = int(m.jnt_qposadr[m.joint("wrist_roll").id])

        self.jnt_lo = jnp.asarray(m.jnt_range[:N_ARM + 1, 0])
        self.jnt_hi = jnp.asarray(m.jnt_range[:N_ARM + 1, 1])
        ctrl_lo, ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self.ctrl_lo, self.ctrl_hi = jnp.asarray(ctrl_lo), jnp.asarray(ctrl_hi)

        # Ready pose (action 0) and per-joint scale so a = +/-1 reaches both range ends.
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)
        target = np.r_[self.cube_xy0, self.cube_z0 + cfg.ready_height]
        ready = _ik_ready_pose(m, self.site, target, self.home_qpos)
        self.ready = jnp.asarray(ready)
        self.half_range = jnp.asarray(np.maximum(ctrl_hi[:N_ARM] - ready, ready - ctrl_lo[:N_ARM]))
        self.home_grasp = jnp.asarray(d.site_xpos[self.site].copy())

        # Fixed camera for the visibility penalty (pinhole, same math as rewards/audit Camera).
        cam = m.camera(CAMERA_NAME).id
        self.cam_pos = jnp.asarray(d.cam_xpos[cam].copy())
        self.cam_rot = jnp.asarray(d.cam_xmat[cam].reshape(3, 3).copy())
        self.tan_y = float(np.tan(np.deg2rad(m.cam_fovy[cam]) / 2))
        self.tan_x = self.tan_y * CAM_WIDTH / CAM_HEIGHT
        self.vis_bodies = jnp.asarray([m.body(n).id for n in VIS_BODIES])

    # ------------------------------------------------------------------ sizes
    action_size = N_ARM + 1
    proprio_size = 6 + 6 + 6 + 3 + 1

    # ------------------------------------------------------------------ helpers
    def _grasp(self, data):
        return data.site_xpos[self.site]

    def _cube(self, data):
        return data.qpos[self.cube_q:self.cube_q + 3]

    def _obs(self, data, step):
        t_left = 1.0 - step.astype(jnp.float32) / self.cfg.episode_steps
        return jnp.concatenate([data.qpos[:6], 0.1 * data.qvel[:6], data.ctrl[:6],
                                self._grasp(data), t_left[None]]).astype(jnp.float32)

    def _visibility(self, data):
        pts = jnp.concatenate([data.xpos[self.vis_bodies], self._grasp(data)[None]], axis=0)
        q = (pts - self.cam_pos) @ self.cam_rot               # camera frame (looks along -z)
        depth = -q[:, 2]
        safe = jnp.maximum(depth, 0.01)
        uv = jnp.stack([q[:, 0] / safe / self.tan_x, q[:, 1] / safe / self.tan_y], axis=-1)
        uv = jnp.where(depth[:, None] < 0.01, 2.0, jnp.clip(uv, -2.0, 2.0))
        a = jnp.abs(uv)
        excess = jnp.clip(a - (1.0 - VIS_MARGIN), 0.0, 1.0).sum()
        out = jnp.any(a > 1.0, axis=-1).sum().astype(jnp.float32)
        return excess, out

    def _inputs(self, data, action, cube_z0):
        grasp, cube = self._grasp(data), self._cube(data)
        q = data.qpos[:N_ARM]
        grip_open = self._jaw_is_open(data).astype(jnp.float32)   # the command, after hysteresis
        stalled = (1.0 - grip_open) * (data.qpos[self.jaw_q] > self.jnt_lo[N_ARM] + STALL_MARGIN)
        near = ((q - self.jnt_lo[:N_ARM] < LIMIT_MARGIN) | (self.jnt_hi[:N_ARM] - q < LIMIT_MARGIN))
        excess, out = self._visibility(data)
        w, x, y, z = (data.qpos[self.cube_q + 3 + i] for i in range(4))
        up = jnp.abs(jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]))
        tilt = jnp.arccos(jnp.clip(up.max(), 0.0, 1.0))      # 0 = resting flat on any face
        f = jnp.float32
        return PickInputs(
            d_xy=f(jnp.linalg.norm(grasp[:2] - cube[:2])), dz=f(grasp[2] - cube[2]),
            cube_lift=f(cube[2] - cube_z0), gripper_h=f(grasp[2] - TABLE_Z),
            grip_open=grip_open, jaw_stalled=f(stalled),
            home_dist=f(jnp.linalg.norm(grasp - self.home_grasp)),
            vis_excess=f(excess), vis_out=f(out), near_limit=f(near.sum()),
            action=action.astype(jnp.float32), cube_tilt=f(tilt),
            joint_speed=f(jnp.sum(jnp.square(data.qvel[:N_ARM]))),
            opposing_sides=f(opposing_finger_sides(
                data.geom_xpos[self.fixed_tip], data.geom_xpos[self.moving_tip], cube,
                data.xmat[self.cube_body].reshape(3, 3), self.cfg.reward.cube_half)),
            wrist_roll=f(data.qpos[self.wrist_q]))

    def _jaw_is_open(self, data):
        return data.ctrl[N_ARM] > 0.5 * (self.ctrl_lo[N_ARM] + self.cfg.jaw_open)

    # ------------------------------------------------------------------ API
    def make_data(self):
        return mjx.make_data(self.mj_model, impl="warp", naconmax=self.naconmax,
                             njmax=self.njmax)

    def reset(self, key: jax.Array) -> PickState:
        cfg = self.cfg
        k_q, k_xy, k_yaw = jax.random.split(key, 3)
        qpos = jnp.asarray(self.home_qpos)
        arm = qpos[:N_ARM] + cfg.joint_noise * jax.random.normal(k_q, (N_ARM,))
        qpos = qpos.at[:N_ARM].set(jnp.clip(arm, self.jnt_lo[:N_ARM], self.jnt_hi[:N_ARM]))
        xy = jnp.asarray(self.cube_xy0) + jax.random.uniform(
            k_xy, (2,), minval=-cfg.cube_range, maxval=cfg.cube_range)
        yaw = jax.random.uniform(k_yaw, (), minval=-cfg.cube_yaw_range, maxval=cfg.cube_yaw_range)
        quat = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = qpos.at[self.cube_q:self.cube_q + 7].set(jnp.concatenate([xy, jnp.array([self.cube_z0]), quat]))

        data = self.make_data()
        ctrl = jnp.asarray(self.home_ctrl).at[:N_ARM].set(qpos[:N_ARM])
        if cfg.start_jaw_open:
            ctrl = ctrl.at[N_ARM].set(cfg.jaw_open)
            qpos = qpos.at[self.jaw_q].set(cfg.jaw_open)
        data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel), ctrl=ctrl)
        data = mjx.forward(self.mjx_model, data)

        step = jnp.int32(0)
        x0 = self._inputs(data, jnp.array([0, 0, 0, 0, 0, -1.0]), jnp.float32(self.cube_z0))
        zero = jnp.float32(0.0)
        return PickState(
            data=data, obs=self._obs(data, step), reward=zero, done=zero, step=step,
            cube_z0=jnp.float32(self.cube_z0), rstate=init_pick_state(x0),
            ep_return=zero, ep_success=zero, ep_held=zero,
            metrics=jax.tree.map(jnp.zeros_like, self._metrics(x0, init_pick_state(x0), zero, zero)))

    def _metrics(self, x, rstate, fell, unstable):
        _, terms, _ = pick_reward(self.cfg.reward, rstate, x)
        out = {"held": terms["held"], "success": terms["success"], "aligned": terms["aligned"],
               "opposing_sides": x.opposing_sides,
               "wrist_roll_deg": jnp.degrees(x.wrist_roll),
               "wrist_limit_violation": jnp.float32(jnp.abs(x.wrist_roll) > jnp.pi / 2),
               "lifted": terms["lifted"], "d_xy": x.d_xy, "cube_lift": x.cube_lift,
               "fell": jnp.float32(fell), "unstable": jnp.float32(unstable)}
        out.update({k: v for k, v in terms.items() if k.endswith(("_r", "_pr", "_pen", "_cost"))})
        return out

    def step(self, state: PickState, action: jax.Array) -> PickState:
        cfg = self.cfg
        action = jnp.clip(action, -1.0, 1.0)
        data = state.data

        goal = self.ready + action[:N_ARM] * self.half_range
        target = smooth_joint_targets(
            goal, data.ctrl[:N_ARM], data.qpos[:N_ARM], self.ctrl_lo[:N_ARM], self.ctrl_hi[:N_ARM],
            max_delta=cfg.max_joint_delta, smooth_alpha=cfg.smooth_alpha,
            windup_margin=cfg.windup_margin)
        h = cfg.jaw_hysteresis
        jaw_open = jnp.where(action[N_ARM] > h, True,
                             jnp.where(action[N_ARM] < -h, False, self._jaw_is_open(data)))
        jaw = jnp.where(jaw_open, cfg.jaw_open, self.ctrl_lo[N_ARM])
        data = data.replace(ctrl=data.ctrl.at[:N_ARM].set(target).at[N_ARM].set(jaw))

        def sub(d, _):
            return mjx.step(self.mjx_model, d), None
        data, _ = jax.lax.scan(sub, data, None, length=self.n_substeps)

        x = self._inputs(data, action, state.cube_z0)
        reward, terms, rstate = pick_reward(cfg.reward, state.rstate, x)

        step = state.step + 1
        unstable = ~jnp.all(jnp.isfinite(data.qpos)) | ~jnp.all(jnp.isfinite(data.qvel))
        fell = self._cube(data)[2] < TABLE_Z - 0.02
        done = (step >= cfg.episode_steps) | fell | unstable
        reward = jnp.where(unstable, 0.0, jnp.nan_to_num(reward))

        metrics = {"held": terms["held"], "success": terms["success"],
                   "opposing_sides": x.opposing_sides,
                   "wrist_roll_deg": jnp.degrees(x.wrist_roll),
                   "wrist_limit_violation": jnp.float32(jnp.abs(x.wrist_roll) > jnp.pi / 2),
                   "aligned": terms["aligned"], "lifted": terms["lifted"],
                   "d_xy": x.d_xy, "cube_lift": x.cube_lift,
                   "fell": fell.astype(jnp.float32), "unstable": unstable.astype(jnp.float32)}
        metrics.update({k: v for k, v in terms.items() if k.endswith(("_r", "_pr", "_pen", "_cost"))})
        metrics = jax.tree.map(jnp.nan_to_num, metrics)
        return state.replace(
            data=data, obs=jnp.nan_to_num(self._obs(data, step)), reward=reward,
            done=done.astype(jnp.float32), step=step, rstate=rstate,
            ep_return=state.ep_return + reward,
            ep_success=jnp.maximum(state.ep_success, terms["success"]),
            ep_held=jnp.maximum(state.ep_held, terms["held"]),
            metrics=metrics)
