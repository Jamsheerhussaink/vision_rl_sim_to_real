"""Scripted strategies for the reward audit: honest ones and known hacks.

A strategy is a list of phases `(target, jaw_open, steps, event)`. `target`
is a function of the world (so it can track the cube) returning the gripper
goal. The last phase repeats until the episode ends.

Every exploit names the honest strategy it must LOSE to (`must_lose_to`):
the honest step a learning policy would otherwise have to take next. If the
exploit scores higher, PPO will prefer the hack and stop there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vision_rl.rewards.audit.world import READY_POS, ToyWorld

Phase = tuple  # (target_fn, jaw_open, steps, event)


def at(p) -> Callable:
    return lambda w: np.asarray(p, float)


def over_cube(h: float, dx: float = 0.0, dy: float = 0.0) -> Callable:
    """Point `h` above the cube centre, optionally offset sideways."""
    return lambda w: w.cube + [dx, dy, h]


def ph(target, jaw_open=False, steps=1, event=None) -> Phase:
    return (target, jaw_open, steps, event)


# Reusable honest building blocks (step counts leave time to arrive).
REACH = [ph(over_cube(0.10), False, 45)]
OPEN = [ph(over_cube(0.10), True, 3)]
DESCEND = [ph(over_cube(0.0), True, 14)]
CLOSE = [ph(over_cube(0.0), False, 3)]
GRASP = REACH + OPEN + DESCEND + CLOSE


def hold_here() -> Phase:
    return ph(lambda w: w.grip.copy(), False, 1)


@dataclass
class Strategy:
    name: str
    phases: list
    honest: bool = True
    must_lose_to: str | None = None
    hack: str = ""                    # what the exploit tries to farm


def _cycle(phases: list, times: int) -> list:
    return [p for _ in range(times) for p in phases]


def build_strategies() -> list[Strategy]:
    lift_to = lambda w: np.array([w.grip[0], w.grip[1], 0.425 + 0.115])   # cube 11.5 cm up
    S = [
        # ---------------- honest ladder: each stage must beat the one before
        Strategy("idle", [ph(at(READY_POS), False, 1)]),
        Strategy("reach_only", REACH),
        Strategy("descend_open_wait", REACH + OPEN + DESCEND),
        Strategy("grasp_no_lift", GRASP),
        Strategy("honest_pick", GRASP + [ph(lift_to, False, 1)]),
        Strategy("honest_pick_home", GRASP + [ph(lift_to, False, 14), ph(at(READY_POS), False, 1)]),

        # ---------------- exploits
        Strategy("hover_park_high", REACH + [ph(over_cube(0.25), True, 1)], False,
                 "descend_open_wait", "open bonus, then park high (Madhan BUG E)"),
        Strategy("jaw_toggle", REACH + _cycle([ph(over_cube(0.10), True), ph(over_cube(0.10), False)], 100),
                 False, "descend_open_wait", "re-earn the open bonus by toggling (BUG C)"),
        Strategy("bob_up_down", REACH + _cycle([ph(over_cube(0.04), True, 6), ph(over_cube(0.16), True, 6)], 20),
                 False, "descend_open_wait", "farm descend progress by bobbing (BUG D)"),
        Strategy("align_dither", _cycle([ph(over_cube(0.10, 0.05), False, 6), ph(over_cube(0.10), False, 6)], 21),
                 False, "reach_only", "farm align progress by moving back and forth"),
        Strategy("fake_grasp", REACH + [ph(over_cube(0.0), False, 1)],
                 False, "grasp_no_lift", "rest the CLOSED jaw on top of the cube: 'held' with nothing inside"),
        Strategy("tilt_cube", REACH + [ph(over_cube(0.06), False, 6, "tilt"), ph(over_cube(0.10), False, 1)],
                 False, "descend_open_wait", "tip the cube onto an edge so its centre 'lifts'"),
        Strategy("ledge_park", REACH + [ph(over_cube(0.06), False, 6, "ledge"), ph(at(READY_POS), False, 1)],
                 False, "descend_open_wait", "leave the cube leaning on the box wall: 'lifted' without a grasp"),
        Strategy("side_swipe", [ph(lambda w: w.cube + [-0.10, 0, 0.0], True, 45),
                                ph(over_cube(0.0), True, 12), ph(over_cube(0.0), False, 3),
                                ph(lift_to, False, 1)],
                 False, "honest_pick", "approach low from the side instead of top-down"),
        Strategy("grasp_sit", GRASP + [hold_here()], False, "honest_pick",
                 "grasp and never lift (safe grasp bonus)"),
        Strategy("lift_drop_cycle", GRASP + _cycle(
            [ph(lift_to, False, 14), ph(lambda w: w.grip.copy(), True, 8)]
            + [ph(over_cube(0.10), True, 6)] + DESCEND + CLOSE, 5) + [ph(lift_to, False, 1)],
                 False, "honest_pick", "lift, drop and re-pick to re-earn the stage bonuses"),
        Strategy("toss", GRASP + [ph(lift_to, False, 12), ph(lambda w: w.grip + [0, 0, 0.01], True, 1),
                                  ph(over_cube(0.10), False, 1)],
                 False, "honest_pick", "fling the cube up and let go: lift paid while it flies"),
        Strategy("drag_home", GRASP + [ph(lambda w: np.array([READY_POS[0], READY_POS[1], w.grip[2]]), False, 1)],
                 False, "honest_pick", "drag the held cube along the table towards home"),
        Strategy("hide_out_of_frame", [ph(at([0.10, -0.45, 0.75]), False, 1)], False, "idle",
                 "leave the camera view (must be worse than doing nothing)"),
    ]
    return S


def rollout(strategy: Strategy, world: ToyWorld, episode_length: int):
    """Yield reward inputs for every step of one episode."""
    steps = []
    for target, jaw_open, n, event in strategy.phases:
        for i in range(n):
            steps.append((target, jaw_open, event if i == 0 else None))
    last = strategy.phases[-1]
    for t in range(episode_length):
        target, jaw_open, event = steps[t] if t < len(steps) else (last[0], last[1], None)
        yield world.step(target(world), jaw_open, event)
