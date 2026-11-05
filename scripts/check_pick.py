"""Is the pick task actually achievable with the current reward, robot, scene and camera?

    python scripts/check_pick.py            # full report
    python scripts/check_pick.py --no-physics

Uses the real MuJoCo model (not the toy audit world):
  1 gripper geometry   jaw opening vs the 5 cm cube; where the grasp point sits
  2 workspace sweep    for every cube spot on a grid: top-down IK for hover,
                       straight descent and 10 cm lift; joint limits; arm vs
                       table/box collisions; every moving joint in camera view
  3 reward ladder      pick_reward fed with the real FK numbers along that path:
                       does every stage (aligned, top-down, held, success) fire,
                       and do the penalties stay small on the honest path?
  4 home / start pose  is the start pose itself penalised?
  5 physics pick       drive the real sim along the IK path: does the jaw stall on
                       the cube, does the cube rise 10 cm, does `held` hold?
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # tiny reward math; leave the GPU to training

import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.optimize import least_squares

from vision_rl.envs import SCENE_XML
from vision_rl.envs.grasp_geometry import opposing_finger_sides
from vision_rl.envs.control import smooth_joint_targets
from vision_rl.envs.pick_env import PickEnvConfig
from vision_rl.rewards.audit.world import Camera
from vision_rl.rewards.pick import PickInputs, PickRewardConfig, init_pick_state, pick_reward

TABLE_Z, CUBE_HALF = 0.40, 0.025
CUBE_Z = TABLE_Z + CUBE_HALF
DOWN = np.array([0.0, 0.0, -1.0])
LIMIT_MARGIN = 0.05                  # rad; matches the reward's "near limit"
# Robot points that must stay in view. Base and shoulder_pan are left out: the fixed
# camera cuts them off in the real image too.
VIS_BODIES = ["upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
VIS_SITES = ["gripperframe"]
OBSTACLES = ["table", "target_box"]  # bodies the arm must not touch (floor is far below)

CFG = PickRewardConfig()


class Robot:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.home_q = self.d.qpos[:6].copy()
        self.lo, self.hi = self.m.jnt_range[:6, 0].copy(), self.m.jnt_range[:6, 1].copy()
        self.site = self.m.site("gripperframe").id
        self.robot_bodies = set(range(self.m.body("base").id, self.m.body("moving_jaw_so101_v1").id + 1))
        self.obstacles = {self.m.body(n).id for n in OBSTACLES}
        self.cube_body = self.m.body("cube").id

    def fk(self, q, cube_xyz=None):
        d = self.d
        d.qpos[:6] = q
        if cube_xyz is not None:
            d.qpos[6:9] = cube_xyz
        mujoco.mj_forward(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3)[:, 0].copy()

    def ik(self, target, roll=0.0, seeds=None, margin=LIMIT_MARGIN):
        """IK: grasp point at `target` (hard), approach axis as close to straight down as
        the joints allow (soft). Joints are kept `margin` rad inside their limits so the
        honest path does not pay limit_pen.

        Solves the 4 positioning joints; wrist roll (which only spins the jaw about the
        approach axis) is held at `roll`. Returns (q, pos_err, tilt_deg).
        """
        lo, hi = self.lo[:4] + margin, self.hi[:4] - margin

        def resid(x):
            q = np.r_[x, roll, self.home_q[5]]
            p, a = self.fk(q)
            return np.r_[(p - target) / 0.0002, (a - DOWN) / 0.05]

        best = None
        seeds = seeds or [self.home_q[:4], np.array([0.0, 0.0, 0.0, 1.2]),
                          np.array([0.0, -0.8, 0.8, 1.4]), np.array([0.0, 0.5, -0.5, 1.5])]
        for s in seeds:
            s = s.copy()
            s[0] = np.arctan2(target[1], target[0] - 0.045)          # aim the pan joint first
            r = least_squares(resid, np.clip(s, lo + 1e-3, hi - 1e-3),
                              bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
            if best is None or r.cost < best.cost:
                best = r
        q = np.r_[best.x, roll, self.home_q[5]]
        p, a = self.fk(q)
        ang = np.degrees(np.arccos(np.clip(a @ DOWN, -1, 1)))
        return q, float(np.linalg.norm(p - target)), float(ang)

    def collides(self, q, cube_xyz) -> float:
        """Deepest penetration (m, >0 = touching) between the arm and table/box."""
        self.fk(q, cube_xyz)
        worst = 0.0
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if pair & self.robot_bodies and pair & self.obstacles:
                worst = max(worst, -c.dist)
        return worst

    def in_box_area(self, x, y, clearance=0.04) -> bool:
        """Cube centre within the box footprint grown by the cube half-width + clearance."""
        b = self.m.body("target_box").id
        local = self.d.xmat[b].reshape(3, 3).T @ (np.array([x, y, 0.0]) - self.d.xpos[b])
        return abs(local[0]) < 0.10 + clearance and abs(local[1]) < 0.08 + clearance

    def vis_points(self):
        return ([self.d.xpos[self.m.body(n).id].copy() for n in VIS_BODIES]
                + [self.d.site_xpos[self.m.site(n).id].copy() for n in VIS_SITES])

    def near_limit(self, q) -> int:
        q5 = np.asarray(q[:5])
        return int(np.sum((q5 - self.lo[:5] < LIMIT_MARGIN) | (self.hi[:5] - q5 < LIMIT_MARGIN)))


# ------------------------------------------------------------------ 1 geometry
def check_geometry(rob: Robot):
    m, d = rob.m, rob.d
    print("\n1) GRIPPER GEOMETRY")
    tips = lambda: [np.mean([d.geom_xpos[m.geom(f"{j}_jaw_sph_tip{i}").id] for i in (1, 2, 3)], 0)
                    for j in ("fixed", "moving")]
    for g in (rob.lo[5], 0.45, rob.hi[5]):
        q = rob.home_q.copy(); q[5] = g; rob.fk(q)
        f, mv = tips()
        print(f"   jaw {g:+.2f} rad -> tip gap {np.linalg.norm(f - mv) * 100:5.1f} cm")
    # Jaw opened to the cube's width: how far past the grasp point do the fingertips reach?
    q = rob.home_q.copy(); q[5] = 0.45
    p, a = rob.fk(q)
    f, mv = tips()
    depth = max((t - p) @ a for t in (f, mv))
    print(f"   at a 5 cm opening the fingertips reach {depth * 100:+.1f} cm past the grasp point")
    clear = CUBE_HALF - depth
    print(f"   -> grasp point at the cube centre (dz=0) puts the fingertips {clear * 100:.1f} cm above "
          f"the table: {'OK' if clear > 0.003 else 'too low, shift the grasp band up'}")


# ------------------------------------------------------------------ 2 workspace sweep
def best_roll(rob: Robot, target, cube_yaw):
    """Wrist roll that lines the jaw's closing direction up with a cube face."""
    best = (1e9, 0.0)
    limit = min(abs(rob.m.actuator("wrist_roll").ctrlrange))
    for roll in np.linspace(-limit, limit, 13):
        q, pe, ae = rob.ik(target, roll)
        rob.fk(q)
        m, d = rob.m, rob.d
        f = d.geom_xpos[m.geom("fixed_jaw_sph_tip1").id]
        mv = d.geom_xpos[m.geom("moving_jaw_sph_tip1").id]
        ang = np.arctan2(*(mv - f)[1::-1])
        err = abs(((ang - cube_yaw) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        best = min(best, (err + pe * 10, roll))
    return best[1]


GRASP_OFFSET = 0.0  # robot.xml now includes the 8 mm grasp-marker correction.


def jaw_axis(rob: Robot, q):
    """Horizontal unit vector from the fixed fingertip towards the moving one."""
    q = q.copy(); q[5] = 0.45
    rob.fk(q)
    m, d = rob.m, rob.d
    v = d.geom_xpos[m.geom("moving_jaw_sph_tip1").id] - d.geom_xpos[m.geom("fixed_jaw_sph_tip1").id]
    v[2] = 0.0
    return v / np.linalg.norm(v)


def plan_pick(rob: Robot, cube_xy, cube_yaw=0.0, grasp_dz=0.005, offset=GRASP_OFFSET, reward_cfg=None):
    """IK waypoints for a top-down pick at `cube_xy`. Returns list of (label, q, target, pos_err, tilt)."""
    cfg = reward_cfg or CFG
    c = np.array([cube_xy[0], cube_xy[1], CUBE_Z])
    roll = best_roll(rob, c + [0, 0, 0.08], cube_yaw)
    c = c + offset * jaw_axis(rob, rob.ik(c + [0, 0, grasp_dz], roll)[0])
    path = []
    # Hover only as high as the reward's top-down latch needs (topdown_h + 1 cm).
    for label, h in [("hover", cfg.topdown_h + 0.01), ("mid", 0.035), ("grasp", grasp_dz),
                     ("lift5", grasp_dz + 0.05), ("lift", grasp_dz + cfg.pick_height + 0.015)]:
        q, pe, ae = rob.ik(c + [0, 0, h], roll)
        path.append((label, q, c + [0, 0, h], pe, ae))
    return path, roll


def sweep(rob: Robot, cam: Camera, xs, ys):
    print(f"\n2) WORKSPACE SWEEP  (hover {(CFG.topdown_h + 0.01) * 100:.0f} cm -> descend -> grasp -> "
          f"lift {CFG.pick_height * 100:.0f} cm, gripper as vertical as the joints allow)")
    print("   legend: # all OK | v a joint leaves the camera view | t gripper tilted >15 deg at grasp |\n"
          "           L only reachable with a joint on its limit | c arm hits table/box | x out of reach |\n"
          "           B cube would overlap the cardboard box | . cube itself out of view")
    rows, stats = [], {k: 0 for k in "#vtLcxB."}
    for y in ys[::-1]:
        row = ""
        for x in xs:
            cube_uv = cam.uv([x, y, CUBE_Z])
            if (np.abs(cube_uv) > 0.95).any():
                row += "."; stats["."] += 1; continue
            if rob.in_box_area(x, y):
                row += "B"; stats["B"] += 1; continue
            path, _ = plan_pick(rob, (x, y))
            if any(pe > 0.002 for _, _, _, pe, _ in path):
                full = [rob.ik(t, q[4], margin=0.0)[1] for _, q, t, *_ in path]
                ch = "L" if max(full) < 0.002 else "x"
            elif path[2][4] > 15:
                ch = "t"
            elif any(rob.collides(q, [x, y, CUBE_Z] if lab == "hover" else [x + 1, y, CUBE_Z]) > 0.002
                     for lab, q, *_ in path):
                ch = "c"
            else:
                excess = 0.0
                for _, q, *_ in path:
                    rob.fk(q)
                    excess += cam.visibility(rob.vis_points())[0]
                ch = "#" if excess < 1e-6 else "v"
            row += ch; stats[ch] += 1
        rows.append(f"   y={y:+.2f} |{row}|")
    for r in rows:
        print(r)
    print(f"           x: {xs[0]:.2f} ... {xs[-1]:.2f} (step {xs[1] - xs[0]:.2f} m)")
    total = sum(stats.values())
    print("   " + "  ".join(f"{k}:{v}" for k, v in stats.items()) + f"  (of {total})")
    return stats


# ------------------------------------------------------------------ 3/4 reward along the path
def reward_inputs(rob: Robot, cam: Camera, q, cube, jaw_open, stalled, home_pos, prev_q):
    p, _ = rob.fk(q, cube)
    excess, out = cam.visibility(rob.vis_points())
    dq = np.clip((np.asarray(q[:5]) - prev_q[:5]) / 0.08, -1, 1)
    return dict(d_xy=np.linalg.norm(p[:2] - cube[:2]), dz=p[2] - cube[2], cube_lift=cube[2] - CUBE_Z,
                gripper_h=p[2] - TABLE_Z, grip_open=float(jaw_open), jaw_stalled=float(stalled),
                home_dist=np.linalg.norm(p - home_pos), vis_excess=excess, vis_out=out,
                near_limit=float(rob.near_limit(q)), action=np.r_[dq, 1.0 if jaw_open else -1.0])


def reward_ladder(rob: Robot, cam: Camera, cube_xy):
    print(f"\n3) REWARD LADDER on the real robot path, cube at {tuple(np.round(cube_xy, 3))}")
    path, roll = plan_pick(rob, cube_xy)
    home_pos, _ = rob.fk(rob.home_q)
    c0 = np.array([cube_xy[0], cube_xy[1], CUBE_Z])
    # (label, q, cube, jaw_open, stalled) with 4 interpolated steps between waypoints
    seq, prev = [], rob.home_q
    plan = [("start", rob.home_q, c0, False, False)]
    for lab, q, tgt, *_ in path:
        lifted = lab.startswith("lift")
        grasp_off = path[2][2][2] - CUBE_Z                     # grasp point above cube centre
        cube = (tgt - [0, 0, grasp_off]) if lifted else c0
        plan.append((lab, q, cube, lab in ("hover", "mid", "grasp"), lifted))
        if lab == "grasp":                                     # close the jaw on the cube
            plan.append(("close", q, c0, False, True))
    for (la, qa, ca, oa, sa), (lb, qb, cb, ob, sb) in zip(plan, plan[1:]):
        for t in np.linspace(0, 1, 5)[1:]:
            seq.append((lb, qa + t * (qb - qa), ca + t * (cb - ca), ob, sb))
    seq += [seq[-1]] * 5                                       # hold at the top
    state, rows = None, []
    for lab, q, cube, jaw_open, stalled in seq:
        x = reward_inputs(rob, cam, q, cube, jaw_open, stalled, home_pos, prev)
        prev = q
        xi = PickInputs(**{k: jnp.asarray(v, jnp.float32) for k, v in x.items()})
        state = state or init_pick_state(xi)
        r, terms, state = pick_reward(CFG, state, xi)
        rows.append((lab, float(r), {k: float(v) for k, v in terms.items()}, x))
    print("   stage   reward  aligned topdown held success  vis_pen limit_pen low_pen")
    shown = set()
    for lab, r, t, x in rows:
        if lab in shown and lab not in ("lift",):
            continue
        shown.add(lab)
        print(f"   {lab:6s} {r:+7.2f}    {t['aligned']:.0f}      {float(state.topdown):.0f}      "
              f"{t['held']:.0f}     {t['success']:.0f}    {t['vis_pen']:+6.2f}   {t['limit_pen']:+6.2f}  {t['low_pen']:+6.2f}")
    ok = rows[-1][2]["success"] > 0.5
    print(f"   -> success reachable in reward terms: {'YES' if ok else 'NO'}")
    return ok, rows


def home_check(rob: Robot, cam: Camera, cube_xy):
    print("\n4) START / HOME POSE")
    p, a = rob.fk(rob.home_q, [cube_xy[0], cube_xy[1], CUBE_Z])
    excess, out = cam.visibility(rob.vis_points())
    lim = rob.near_limit(rob.home_q)
    d_xy = np.linalg.norm(p[:2] - np.asarray(cube_xy))
    low = (p[2] - TABLE_Z < CFG.low_zone_h) and d_xy >= CFG.grasp_dist
    names = [rob.m.joint(i).name for i in range(5)]
    at_lim = [n for n, q, lo, hi in zip(names, rob.home_q, rob.lo, rob.hi) if q - lo < LIMIT_MARGIN or hi - q < LIMIT_MARGIN]
    print(f"   grasp point {np.round(p, 3)}  ({(p[2] - TABLE_Z) * 100:.1f} cm above table, {d_xy * 100:.1f} cm from cube)")
    print(f"   joints on a limit: {at_lim or 'none'}   -> limit_pen {-CFG.limit_penalty * lim:+.2f}/step")
    print(f"   in the low zone away from the cube: {low}   -> low_pen {-CFG.low_zone_penalty * low:+.2f}/step")
    print(f"   visibility overshoot {excess:.2f}, points out of view {out:.0f}   -> vis_pen "
          f"{-(CFG.vis_scale * excess + CFG.vis_out_penalty * out):+.2f}/step")
    return lim, low, excess


# ------------------------------------------------------------------ 5 physics
def physics_pick(rob: Robot, cube_xy, cube_yaw=None, *, approach="safe", recording=None,
                 control=None, reward_cfg=None, task_height=None, return_report=False):
    print(f"\n5) PHYSICS PICK (real MuJoCo, position servos) at cube {tuple(np.round(cube_xy, 3))}")
    m = rob.m
    control = control or PickEnvConfig()
    cfg = reward_cfg or CFG
    height = cfg.pick_height if task_height is None else task_height
    m.opt.iterations, m.opt.ls_iterations = control.solver_iterations, control.ls_iterations
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if control.start_jaw_open:
        d.qpos[5] = d.ctrl[5] = control.jaw_open
    d.qpos[6:9] = [cube_xy[0], cube_xy[1], CUBE_Z]
    if cube_yaw is not None:
        d.qpos[9:13] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
    mujoco.mj_forward(m, d)
    yaw = 2 * np.arctan2(d.qpos[12], d.qpos[9])
    initial_penetration = 0.0
    for c in d.contact[:d.ncon]:
        bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
        if rob.cube_body in bodies and bodies & rob.robot_bodies:
            initial_penetration = max(initial_penetration, -float(c.dist))
    path, roll = plan_pick(rob, cube_xy, yaw, reward_cfg=replace(cfg, pick_height=height))
    wp = {lab: q for lab, q, *_ in path}
    OPEN, CLOSED = 1.2, rob.lo[5]
    jaw_hist = []
    substeps = max(1, round(control.ctrl_dt / m.opt.timestep))
    peak_wrist = 0.0
    stable_steps = 0
    longest_stable = 0
    wrist_q = int(m.jnt_qposadr[m.joint("wrist_roll").id])
    fixed_tip = m.geom("fixed_jaw_sph_tip1").id
    moving_tip = m.geom("moving_jaw_sph_tip1").id
    first_contact = None
    elapsed_steps = 0
    grasp_achieved = False
    peak_lift = 0.0
    if recording is not None:
        recording.append(dict(time=0.0, phase="reset", qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                              ctrl=d.ctrl.copy(), goal=d.ctrl[:5].copy(), contacts=""))

    def sides_ok():
        return bool(opposing_finger_sides(
            d.geom_xpos[fixed_tip], d.geom_xpos[moving_tip], d.qpos[6:9],
            d.xmat[rob.cube_body].reshape(3, 3), xp=np))

    def go(q_to, jaw, secs, phase):
        nonlocal peak_wrist, stable_steps, longest_stable, first_contact, elapsed_steps, grasp_achieved, peak_lift
        # Change goals only on camera/control ticks, just like the training env.
        n = max(0, round(secs / control.ctrl_dt)) * substeps
        for i in range(n):
            if i % substeps == 0:
                d.ctrl[:5] = smooth_joint_targets(
                    np.asarray(q_to[:5]), d.ctrl[:5], d.qpos[:5],
                    m.actuator_ctrlrange[:5, 0], m.actuator_ctrlrange[:5, 1],
                    max_delta=control.max_joint_delta, smooth_alpha=control.smooth_alpha,
                    windup_margin=control.windup_margin, xp=np)
            d.ctrl[5] = jaw
            mujoco.mj_step(m, d)
            elapsed_steps += 1
            contact_names = []
            contact_bodies = set()
            for c in d.contact[:d.ncon]:
                bodies = [int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
                if rob.cube_body not in bodies or c.dist > 0:
                    continue
                other = int(c.geom2 if bodies[0] == rob.cube_body else c.geom1)
                if int(m.geom_bodyid[other]) not in rob.robot_bodies:
                    continue
                body_name = m.body(m.geom_bodyid[other]).name
                mesh = int(m.geom_dataid[other])
                label = m.geom(other).name or (f"{body_name}/{m.mesh(mesh).name}"
                        if m.geom_type[other] == mujoco.mjtGeom.mjGEOM_MESH else body_name)
                contact_names.append(label)
                contact_bodies.add(body_name)
                if first_contact is None:
                    first_contact = (phase, elapsed_steps * m.opt.timestep, label)
            if recording is not None and i % substeps == substeps - 1:
                recording.append(dict(time=elapsed_steps * m.opt.timestep, phase=phase,
                                      qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(),
                                      goal=np.asarray(q_to[:5]).copy(),
                                      contacts=", ".join(sorted(set(contact_names)))))
            peak_wrist = max(peak_wrist, abs(float(d.qpos[wrist_q])))
            both = "gripper" in contact_bodies and "moving_jaw_so101_v1" in contact_bodies
            grasp = jaw == CLOSED and sides_ok() and both
            grasp_achieved |= bool(grasp)
            peak_lift = max(peak_lift, float(d.qpos[8] - CUBE_Z))
            stable = grasp and d.qpos[8] - CUBE_Z >= height
            stable_steps = stable_steps + 1 if stable else 0
            longest_stable = max(longest_stable, stable_steps)
        jaw_hist.append(d.qpos[5])

    if approach == "safe":
        home_point = d.site_xpos[rob.site].copy()
        clear, _, _ = rob.ik(home_point + [0, 0, 0.08], roll=float(d.qpos[wrist_q]))
        go(clear, CLOSED, 1.5, "raise_clear")
        go(wp["hover"], CLOSED, 1.5, "move_above")
    else:
        go(wp["hover"], OPEN if control.start_jaw_open else CLOSED, 1.5, "direct_approach")
    go(wp["hover"], OPEN, 0.4, "open_above")
    go(wp["grasp"], OPEN, 1.0, "descend")
    approach_shift = float(np.linalg.norm(d.qpos[6:8] - cube_xy))
    go(wp["grasp"], CLOSED, 0.6, "close")
    jaw_closed_on_cube = float(d.qpos[5])
    go(wp["lift"], CLOSED, 1.2, "lift")
    remaining = control.episode_steps * control.ctrl_dt - elapsed_steps * m.opt.timestep
    if remaining > 0:
        go(wp["lift"], CLOSED, remaining, "hold")
    cube = d.qpos[6:9]
    p = d.site_xpos[rob.site]
    lift = cube[2] - CUBE_Z
    stalled = jaw_closed_on_cube > rob.lo[5] + 0.2       # empty jaw closes to the lower limit
    print(f"   jaw angle after closing on the cube: {jaw_closed_on_cube:.3f} rad "
          f"(empty jaw closes to {rob.lo[5]:.3f}) -> stalled by cube: {stalled}")
    print(f"   cube lifted {lift * 100:.1f} cm (success needs {height * 100:.1f} cm); "
          f"grasp point to cube: d_xy {np.linalg.norm(p[:2] - cube[:2]) * 100:.1f} cm, dz {(p[2] - cube[2]) * 100:+.1f} cm")
    print(f"   opposite-side finger placement: {sides_ok()}; peak wrist {np.degrees(peak_wrist):.1f} deg; "
          f"longest held lift {longest_stable * m.opt.timestep:.2f} s")
    print(f"   cube horizontal displacement from spawn: {np.linalg.norm(cube[:2] - cube_xy) * 100:.1f} cm")
    print(f"   first robot/cube contact: {first_contact}; displacement before closing: {approach_shift * 100:.1f} cm")
    print(f"   initial robot/cube overlap: {initial_penetration * 1000:.2f} mm")
    ok = stalled and lift >= height and np.linalg.norm(p[:2] - cube[:2]) < cfg.align_tol \
        and abs(p[2] - cube[2]) < cfg.grasp_z_tol and sides_ok() \
        and peak_wrist <= np.pi / 2 and longest_stable * m.opt.timestep >= 0.3 and approach_shift < 0.01 \
        and elapsed_steps * m.opt.timestep <= control.episode_steps * control.ctrl_dt + 1e-6
    print(f"   -> real physics reaches success: {'YES' if ok else 'NO'}")
    if return_report:
        return dict(success=bool(ok), grasp_achieved=bool(grasp_achieved),
                    cube_xy=list(map(float, cube_xy)), cube_yaw_deg=float(np.degrees(yaw)),
                    lift_m=float(lift), peak_lift_m=peak_lift, required_height_m=float(height),
                    peak_wrist_deg=float(np.degrees(peak_wrist)),
                    stable_lift_s=longest_stable * m.opt.timestep,
                    elapsed_s=elapsed_steps * m.opt.timestep,
                    approach_shift_m=approach_shift, initial_overlap_m=initial_penetration,
                    first_contact=first_contact)
    return ok


# ------------------------------------------------------------------ 6 visibility vs lift
def visibility_vs_lift(rob: Robot, cam: Camera, spots, lifts=(0.10, 0.08, 0.06, 0.05), tilts=(5, 30, 45)):
    """At how many cube spots can the arm hold the cube `lift` up with every joint in view?

    The arm has one spare degree of freedom (gripper tilt), so this searches for the
    pose with the least view overshoot, allowing the gripper to tilt up to `tilt` deg.
    """
    print("\n6) VISIBILITY vs LIFT HEIGHT  (spots where the lifted pose keeps every joint in view)")
    lo, hi = rob.lo[:4] + LIMIT_MARGIN, rob.hi[:4] - LIMIT_MARGIN

    def overshoot():
        return sum(np.clip(np.abs(cam.uv(pt)) - (1 - cam.margin), 0, None).sum() for pt in rob.vis_points())

    def solve(target, max_tilt):
        def res(x):
            p, a = rob.fk(np.r_[x, 0.4, rob.home_q[5]])
            tilt = np.degrees(np.arccos(np.clip(a @ DOWN, -1, 1)))
            return np.r_[(p - target) / 0.0002, overshoot() / 0.002, max(0, tilt - max_tilt) / 0.5]
        best = None
        for s in ([0, -0.3, 0.3, 1.4], [0, 0.3, -0.3, 1.2], [0, -0.8, 1.2, 0.6], [0, 0.6, 0.4, 0.2]):
            s = np.array(s, float); s[0] = np.arctan2(target[1], target[0] - 0.045)
            r = least_squares(res, np.clip(s, lo + 1e-3, hi - 1e-3), bounds=(lo, hi))
            best = r if best is None or r.cost < best.cost else best
        p, a = rob.fk(np.r_[best.x, 0.4, rob.home_q[5]])
        tilt = np.degrees(np.arccos(np.clip(a @ DOWN, -1, 1)))
        return np.linalg.norm(p - target) < 0.002 and overshoot() < 1e-4 and tilt <= max_tilt + 1

    print("   lift   " + "".join(f"  tilt<={t:2d}deg" for t in tilts))
    for lift in lifts:
        cells = [sum(solve(np.array([x, y, CUBE_Z + 0.005 + lift]), t) for x, y in spots) for t in tilts]
        print(f"   {lift * 100:3.0f} cm " + "".join(f"  {c:4d}/{len(spots):<6d}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--physics-only", action="store_true", help="only run the scripted grip/lift check")
    ap.add_argument("--spawn-grid", action="store_true",
                    help="with --physics-only: test centre and four +/-3 cm corners at -45, 0, +45 deg")
    ap.add_argument("--approach", choices=("safe", "direct"), default="safe",
                    help="safe raises clear before moving above the cube; direct retains the old test route")
    ap.add_argument("--step", type=float, default=0.03, help="sweep grid step (m)")
    ap.add_argument("--pick-height", type=float, help="override the success lift height (m)")
    args = ap.parse_args()
    global CFG
    if args.pick_height:
        CFG = replace(CFG, pick_height=args.pick_height)

    rob, cam = Robot(), Camera()
    cube_xy = rob.d.qpos[6:8].copy()                       # keyframe cube spot
    if args.spawn_grid and not args.physics_only:
        ap.error("--spawn-grid requires --physics-only")
    if args.physics_only:
        if args.no_physics:
            ap.error("--physics-only conflicts with --no-physics")
        if args.spawn_grid:
            checks = [physics_pick(rob, cube_xy + [dx, dy], yaw, approach=args.approach)
                      for dx, dy in [(0, 0), (-.03, -.03), (-.03, .03), (.03, -.03), (.03, .03)]
                      for yaw in [-np.pi / 4, 0, np.pi / 4]]
            print(f"\nScripted physics picks: {sum(checks)}/{len(checks)} pass (not learned-policy results)")
            raise SystemExit(0 if all(checks) else 1)
        raise SystemExit(0 if physics_pick(rob, cube_xy, approach=args.approach) else 1)
    check_geometry(rob)
    xs = np.round(np.arange(0.10, 0.43, args.step), 3)
    ys = np.round(np.arange(-0.30, 0.24, args.step), 3)
    sweep(rob, cam, xs, ys)
    reward_ladder(rob, cam, cube_xy)
    home_check(rob, cam, cube_xy)
    if not args.no_physics:
        physics_pick(rob, cube_xy, approach=args.approach)
    spots = [(x, y) for x in xs for y in ys
             if not (np.abs(cam.uv([x, y, CUBE_Z])) > 0.95).any() and not rob.in_box_area(x, y)]
    visibility_vs_lift(rob, cam, spots)


if __name__ == "__main__":
    main()
