"""Staged top-down PICK reward for the SO-101 (pure function, no physics).

Ported from Madhan's `compute_pick_reward` (madhan/vision_rl-main,
envs/so101_pick.py) and hardened against the hacks found by
`scripts/check_reward.py`. The env computes the numbers in
`PickInputs` from physics; this file only turns them into a reward, so it can
be unit tested and audited without a simulator.

All heights are measured ABOVE THE TABLE TOP (our table is at z=0.40,
Madhan's at z=0), so the same numbers work in both scenes.

Stages (each one unlocks the next):
  1 align    get the gripper horizontally over the cube        -d_xy, +progress
  2 open     open the jaw while above the cube                 +once per episode
  3 descend  come straight down, only while aligned            -height, +new record low
  4 grasp    jaw closed, blocked by the cube, after a top-down  +per step
             open-jaw approach
  5 lift     raise the held cube, success at pick_height       +height, +success
  6 home     carry the held, lifted cube to the ready pose     -dist, +progress
Penalties: low zone away from the cube (table / side swipe), arm leaving the
camera view, jerky actions, joints at their limits, action size.

`PickRewardConfig.madhan()` switches every fix off so the audit can show
which hacks the original logic allows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax.numpy as jnp


@dataclass(frozen=True)
class PickRewardConfig:
    # --- geometry (metres, above the table top) --------------------------
    cube_half: float = 0.025          # 5 cm cube
    align_tol: float = 0.02           # d_xy below this = "over the cube"
    grasp_z_tol: float = 0.03         # |gripper z - cube z| below this = "at grasp height"
    grasp_dist: float = 0.045         # d_xy inside which being low is allowed
    low_zone_h: float = 0.055         # gripper below this while away from the cube = penalty
    lift_thresh: float = 0.02         # cube raised this much = lifted
    pick_height: float = 0.085        # cube raised this much while held = success: its bottom clears
                                      # the 7.5 cm box walls by 1 cm. Straight-down tops out at ~6.5-8 cm
                                      # (wrist_flex limit), so this needs a slightly tilted carry.
    max_held_tilt: float = 0.26       # rad (15 deg): a more tilted cube NEAR THE TABLE is pinched on an
                                      # edge, not held (tipping a 5 cm cube on its edge raises it ~1 cm)
    topdown_h: float = 0.06           # gripper this high over the cube, jaw open = top-down approach
    height_cap: float = 0.25          # height cost saturates here

    # --- reward scales ----------------------------------------------------
    align_scale: float = 1.0
    align_progress_scale: float = 3.0
    open_bonus: float = 0.5
    descend_scale: float = 1.0
    descend_progress_scale: float = 3.0
    grasp_bonus: float = 0.1          # was 0.5: holding still out-paid lifting (policy grabbed and sat)
    lift_scale: float = 40.0          # was 4: per metre of lift, so 5 cm now pays 2.0 per step
    success_bonus: float = 5.0
    home_scale: float = 1.0
    home_progress_scale: float = 3.0
    low_zone_penalty: float = 0.5
    ctrl_cost_scale: float = 0.01
    vis_scale: float = 2.0            # per unit of image-edge overshoot (normalised coords)
    vis_out_penalty: float = 0.2      # per point fully outside the image
    rate_scale: float = 0.02          # action change between steps (0.1 froze exploration: std 0.6->0.38, no grasp)
    joint_speed_scale: float = 0.01   # per (rad/s)^2 of arm joint speed: favours slow, smooth moves
    limit_penalty: float = 0.1        # per joint near its limit

    # --- hack fixes (all True = ours, all False = Madhan's logic) ---------
    held_requires_stall: bool = True  # "held" needs the jaw blocked by the cube, not just closed nearby
    gate_on_held: bool = True         # approach shaping switches off only while really holding
    lift_requires_hold: bool = True   # lift reward only for a held cube (no tilting / ledges / tosses)
    height_cost_everywhere: bool = True  # not over the cube = pay the max height cost (no edge hovering)
    held_requires_topdown: bool = True   # a grasp counts only after an open-jaw approach from above
    held_requires_opposing_sides: bool = True  # reject a finger on top / both fingers on one side
    staged_open_approach: bool = True  # bonus, descent tracking and height cost share one valid approach

    @classmethod
    def madhan(cls) -> "PickRewardConfig":
        """Madhan's original reward LOGIC with our thresholds (his lift line and pick height
        were absolute heights for a 3.5 cm cube; we keep our 2 cm / 8.5 cm)."""
        return replace(cls(), held_requires_stall=False, gate_on_held=False,
                       lift_requires_hold=False, height_cost_everywhere=False,
                       held_requires_topdown=False, low_zone_h=0.02,
                       vis_scale=0.0, vis_out_penalty=0.0, rate_scale=0.0, limit_penalty=0.0,
                       joint_speed_scale=0.0, max_held_tilt=4.0,
                       held_requires_opposing_sides=False, staged_open_approach=False)

    @classmethod
    def first_run(cls) -> "PickRewardConfig":
        """The exact reward of the first 30M-step run (later reward changes switched off)."""
        return replace(cls(), pick_height=0.10, grasp_bonus=0.5, lift_scale=4.0, rate_scale=0.01,
                       joint_speed_scale=0.0, max_held_tilt=4.0,
                       held_requires_opposing_sides=False, staged_open_approach=False)


class PickInputs(NamedTuple):
    """Per-step numbers the env reads from physics."""
    d_xy: jnp.ndarray          # horizontal gripper -> cube distance
    dz: jnp.ndarray            # gripper height minus cube-centre height
    cube_lift: jnp.ndarray     # cube-centre height minus its resting height
    gripper_h: jnp.ndarray     # gripper height above the table
    grip_open: jnp.ndarray     # 1 if the jaw is commanded open, else 0
    jaw_stalled: jnp.ndarray   # 1 if the jaw is commanded closed but stopped by the cube
    home_dist: jnp.ndarray     # gripper distance to the ready pose
    vis_excess: jnp.ndarray    # summed overshoot of robot points past the image safe margin
    vis_out: jnp.ndarray       # number of robot points outside the image
    near_limit: jnp.ndarray    # number of joints within a small margin of their limit
    action: jnp.ndarray        # [6] action in [-1, 1]
    cube_tilt: jnp.ndarray = 0.0    # rad, cube tilt from resting flat on any face
    joint_speed: jnp.ndarray = 0.0  # sum of squared arm joint velocities (rad/s)^2
    opposing_sides: jnp.ndarray = 1.0  # legacy callers; real env always measures finger placement
    wrist_roll: jnp.ndarray = 0.0    # measured joint angle (rad), for evaluation metrics


class PickRewardState(NamedTuple):
    """What the reward remembers between steps."""
    prev_d_xy: jnp.ndarray
    min_dz: jnp.ndarray        # descend ratchet: lowest dz this approach
    was_open: jnp.ndarray      # open bonus latch
    topdown: jnp.ndarray       # latch: came down over the cube with the jaw open
    prev_home_dist: jnp.ndarray
    prev_action: jnp.ndarray


def init_pick_state(x: PickInputs) -> PickRewardState:
    f = jnp.float32
    return PickRewardState(prev_d_xy=f(x.d_xy), min_dz=f(x.dz), was_open=f(0.0), topdown=f(0.0),
                           prev_home_dist=f(x.home_dist),
                           prev_action=jnp.zeros_like(x.action, dtype=jnp.float32))


def pick_reward(cfg: PickRewardConfig, s: PickRewardState, x: PickInputs):
    """One step of reward. Returns (reward, terms dict, next state)."""
    f = lambda b: jnp.asarray(b, jnp.float32)
    grip_closed = 1.0 - x.grip_open

    aligned = f(x.d_xy < cfg.align_tol)
    at_height = f(jnp.abs(x.dz) < cfg.grasp_z_tol)
    low_zone = f((x.gripper_h < cfg.low_zone_h) & (x.d_xy >= cfg.grasp_dist))
    valid_open_approach = aligned * f(x.dz > cfg.topdown_h) * x.grip_open

    # Top-down latch: set by hovering over the cube with the jaw open, cleared by going low
    # anywhere else. Without it a fast side swipe out-scored the top-down pick.
    topdown = jnp.maximum(s.topdown, valid_open_approach) * (1.0 - low_zone)

    held = aligned * at_height * grip_closed
    if cfg.held_requires_stall:
        # A closed jaw near the cube is not a grasp; the jaw must be stopped BY the cube.
        held = held * x.jaw_stalled
    if cfg.held_requires_topdown:
        held = held * topdown
    if cfg.held_requires_opposing_sides:
        held = held * x.opposing_sides
    # A cube tipped onto its edge and pinched across the diagonal looked "stalled" and sat
    # ~1 cm up; the first policy farmed that for the whole episode. Off the table (> lift_thresh)
    # a tilted cube is really in the hand, and a tilted carry is needed to clear the box walls.
    held = held * f((x.cube_tilt < cfg.max_held_tilt) | (x.cube_lift > cfg.lift_thresh))

    cube_up = f(x.cube_lift > cfg.lift_thresh)
    lifted = held * cube_up if cfg.gate_on_held else cube_up
    success = held * f(x.cube_lift >= cfg.pick_height)

    # Approach shaping runs until the cube is in hand. Madhan switched it off whenever the
    # cube was off the table, so knocking it onto a ledge silenced every approach cost.
    approach = 1.0 - (held if cfg.gate_on_held else jnp.maximum(cube_up, success))

    # 1 align
    align_r = -cfg.align_scale * x.d_xy * approach
    align_pr = cfg.align_progress_scale * (s.prev_d_xy - x.d_xy) * approach

    # 2 open: paid once per episode (a per-step bonus was campable)
    above = f(x.dz > cfg.grasp_z_tol)
    open_event = valid_open_approach if cfg.staged_open_approach else aligned * above * x.grip_open
    was_open = jnp.maximum(s.was_open, open_event)
    open_r = cfg.open_bonus * (was_open - s.was_open) * approach

    # 3 descend: steady cost for hovering high + bonus only for a NEW record low
    height = jnp.clip(x.dz - cfg.grasp_z_tol, 0.0, cfg.height_cap)
    if cfg.height_cost_everywhere:
        # Not over the cube costs as much as the highest hover. Madhan charged height only
        # while aligned, so hovering just outside align_tol dodged the cost.
        height = jnp.where(aligned > 0.5, height, cfg.height_cap)
    else:
        height = jnp.clip(x.dz - cfg.grasp_z_tol, 0.0, None) * aligned
    descent_ready = aligned * x.grip_open * topdown
    if cfg.staged_open_approach:
        # Being low with a closed or unprepared jaw must not silence the height
        # cost. A real grip still switches approach costs off through `approach`.
        height = jnp.where(descent_ready > 0.5, height, cfg.height_cap)
    descend_r = -cfg.descend_scale * height * approach
    if cfg.staged_open_approach:
        # Arm once per episode, when the open approach is actually established.
        # This discards the low reset height without letting repeated hovering,
        # jaw toggling or re-approaches refill the progress reward.
        first_approach = (open_event > 0.5) & (s.was_open < 0.5)
        previous_min = jnp.where(first_approach, jnp.maximum(x.dz, 0.0), s.min_dz)
        min_dz = jnp.where((descent_ready * approach) > 0.5,
                           jnp.minimum(previous_min, jnp.maximum(x.dz, 0.0)), previous_min)
        descend_pr = cfg.descend_progress_scale * jnp.maximum(previous_min - min_dz, 0.0) * descent_ready * approach
    else:
        min_dz = jnp.where(approach > 0.5, jnp.minimum(s.min_dz, x.dz), x.dz)
        descend_pr = (cfg.descend_progress_scale * jnp.clip(s.min_dz - min_dz, 0.0, None)
                      * aligned * x.grip_open * approach)

    # 4 grasp, 5 lift + success
    grasp_r = cfg.grasp_bonus * held
    lift_r = cfg.lift_scale * jnp.clip(x.cube_lift, 0.0, cfg.pick_height)
    if cfg.lift_requires_hold:
        lift_r = lift_r * held
    succ_r = cfg.success_bonus * success

    # 6 carry home while holding the lifted cube
    carrying = held * lifted
    home_r = -cfg.home_scale * x.home_dist * carrying
    home_pr = cfg.home_progress_scale * (s.prev_home_dist - x.home_dist) * carrying

    # penalties
    low_pen = cfg.low_zone_penalty * low_zone
    vis_pen = cfg.vis_scale * x.vis_excess + cfg.vis_out_penalty * x.vis_out
    rate_pen = (cfg.rate_scale * jnp.sum(jnp.square(x.action - s.prev_action))
                + cfg.joint_speed_scale * x.joint_speed)
    limit_pen = cfg.limit_penalty * x.near_limit
    ctrl_cost = cfg.ctrl_cost_scale * jnp.sum(jnp.square(x.action))

    terms = {
        "align_r": align_r, "align_pr": align_pr, "open_r": open_r,
        "descend_r": descend_r, "descend_pr": descend_pr, "grasp_r": grasp_r,
        "lift_r": lift_r, "succ_r": succ_r, "home_r": home_r, "home_pr": home_pr,
        "low_pen": -low_pen, "vis_pen": -vis_pen, "rate_pen": -rate_pen,
        "limit_pen": -limit_pen, "ctrl_cost": -ctrl_cost,
    }
    reward = sum(terms.values())
    terms.update(held=held, success=success, lifted=lifted, aligned=aligned)
    nxt = PickRewardState(prev_d_xy=f(x.d_xy), min_dz=f(min_dz), was_open=f(was_open),
                          topdown=f(topdown),
                          prev_home_dist=f(x.home_dist), prev_action=f(x.action))
    return reward, terms, nxt
