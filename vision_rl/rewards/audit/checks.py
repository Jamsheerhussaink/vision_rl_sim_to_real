"""Run strategies through a reward config and judge them.

Three kinds of check:
  ladder   every honest stage scores more than the stage before it
           (otherwise the policy has no reason to take the next step)
  exploit  every hack scores less than the honest step it replaces
  loop     a random trip that ends where it started earns no net reward
           (progress terms must telescope; ratchets and latches must not refill)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from vision_rl.rewards.audit.strategies import (Strategy, build_strategies, hold_here, over_cube,
                                                ph, rollout, REACH, OPEN)
from vision_rl.rewards.audit.world import CUBE_START, Camera, ToyWorld
from vision_rl.rewards.pick import PickInputs, PickRewardConfig, init_pick_state, pick_reward

EPISODE_LENGTH = 250
MARGIN = 0.5            # an exploit must lose by at least this much return ...
REL_MARGIN = 0.10       # ... and by 10% of the honest return (96% of a grasp is still a hack)
LOOP_TOL = 0.5          # a closed trip may earn one new descend record (ratchet), nothing more


@dataclass
class EpisodeResult:
    name: str
    total: float
    terms: dict             # summed per term
    successes: int          # steps with success
    held: int               # steps with held
    cube_end: np.ndarray    # cube position after the episode


@partial(jax.jit, static_argnums=0)
def _score(cfg: PickRewardConfig, xs: PickInputs):
    """Reward for a whole episode of stacked inputs in one scan."""
    first = jax.tree_util.tree_map(lambda a: a[0], xs)

    def body(state, x):
        r, terms, state = pick_reward(cfg, state, x)
        return state, (r, terms)

    _, (r, terms) = jax.lax.scan(body, init_pick_state(first), xs)
    return r, terms


def run_episode(cfg: PickRewardConfig, strategy: Strategy, camera: Camera,
                episode_length: int = EPISODE_LENGTH) -> EpisodeResult:
    world = ToyWorld(camera)
    steps = list(rollout(strategy, world, episode_length))
    # The toy world has no cube tilt or joint speed: those fields are zero.
    xs = PickInputs(**{k: jnp.asarray(np.stack([d.get(k, 0.0) for d in steps]), jnp.float32)
                       for k in PickInputs._fields})
    r, terms = _score(cfg, xs)
    flags = ("held", "success", "lifted", "aligned")
    totals = {k: float(jnp.sum(v)) for k, v in terms.items() if k not in flags}
    return EpisodeResult(strategy.name, float(jnp.sum(r)), totals,
                         int(jnp.sum(terms["success"] > 0.5)), int(jnp.sum(terms["held"] > 0.5)),
                         world.cube.copy())


@dataclass
class Verdict:
    kind: str        # ladder | exploit | loop
    name: str
    ok: bool
    detail: str


def check_all(cfg: PickRewardConfig, camera: Camera | None = None, loop_trials: int = 20,
              seed: int = 0) -> tuple[dict[str, EpisodeResult], list[Verdict]]:
    camera = camera or Camera()
    strategies = build_strategies()
    res = {s.name: run_episode(cfg, s, camera) for s in strategies}
    out: list[Verdict] = []

    ladder = [s.name for s in strategies if s.honest]
    for lo, hi in zip(ladder, ladder[1:]):
        a, b = res[lo].total, res[hi].total
        out.append(Verdict("ladder", f"{hi} > {lo}", b > a + MARGIN, f"{b:8.2f} vs {a:8.2f}"))

    for s in strategies:
        if s.honest:
            continue
        a, ref = res[s.name].total, res[s.must_lose_to].total
        need = max(MARGIN, REL_MARGIN * abs(ref))
        out.append(Verdict("exploit", s.name, a < ref - need,
                           f"{a:8.2f} vs {s.must_lose_to} {ref:8.2f}  ({s.hack})"))

    out.extend(check_loops(cfg, camera, loop_trials, seed))
    return res, out


def check_loops(cfg: PickRewardConfig, camera: Camera, trials: int, seed: int) -> list[Verdict]:
    """Random closed trips around the cube vs. hovering still for the same time.

    Starts hovering 10 cm over the cube with the jaw open (open bonus already paid).
    Trips that end with the cube moved or grabbed are not closed and are skipped.
    """
    rng = np.random.default_rng(seed)
    prefix = REACH + OPEN
    worst, worst_info, closed = -np.inf, "", 0
    for _ in range(trials):
        jaw = bool(rng.integers(2))
        pts = rng.uniform([-0.06, -0.06, 0.02], [0.06, 0.06, 0.20], size=(int(rng.integers(2, 6)), 3))
        legs = [ph(over_cube(p[2], p[0], p[1]), jaw, 12) for p in pts]
        back = [ph(over_cube(0.10), True, 30)]
        n = sum(p[2] for p in legs + back)
        stay = Strategy("stay", prefix + [ph(over_cube(0.10), True, n), hold_here()])
        loop = Strategy("loop", prefix + legs + back + [hold_here()])
        lr = run_episode(cfg, loop, camera)
        if np.linalg.norm(lr.cube_end - CUBE_START) > 0.005 or lr.held:
            continue
        closed += 1
        gain = lr.total - run_episode(cfg, stay, camera).total
        if gain > worst:
            worst, worst_info = gain, f"jaw_open={jaw} via {np.round(pts, 2).tolist()}"
    return [Verdict("loop", f"{closed}/{trials} random closed trips", closed > 0 and worst <= LOOP_TOL,
                    f"best net gain {worst:+.3f}  {worst_info}")]
