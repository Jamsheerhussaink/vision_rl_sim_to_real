"""`PickRewardConfig.first_run()` must score like the first 30M-step run's reward."""
import jax.numpy as jnp
import numpy as np

from vision_rl.rewards.pick import PickInputs, PickRewardConfig, init_pick_state, pick_reward


def first_run_reward(s, x):
    """The first run's formula, written out with its literal numbers."""
    closed = 1.0 - x.grip_open
    aligned = float(x.d_xy < 0.02)
    at_height = float(abs(x.dz) < 0.03)
    low_zone = float(x.gripper_h < 0.055 and x.d_xy >= 0.045)
    topdown = max(float(s.topdown), aligned * float(x.dz > 0.06) * x.grip_open) * (1 - low_zone)
    held = aligned * at_height * closed * x.jaw_stalled * topdown
    lifted = held * float(x.cube_lift > 0.02)
    success = held * float(x.cube_lift >= 0.10)
    approach = 1.0 - held
    was_open = max(float(s.was_open), aligned * float(x.dz > 0.03) * x.grip_open)
    height = min(max(x.dz - 0.03, 0.0), 0.25) if aligned else 0.25
    min_dz = min(float(s.min_dz), x.dz) if approach > 0.5 else x.dz
    action = np.asarray(x.action)
    return (-x.d_xy * approach + 3.0 * (float(s.prev_d_xy) - x.d_xy) * approach
            + 0.5 * (was_open - float(s.was_open)) * approach
            - height * approach
            + 3.0 * max(float(s.min_dz) - min_dz, 0.0) * aligned * x.grip_open * approach
            + 0.5 * held + 4.0 * min(max(x.cube_lift, 0.0), 0.10) * held + 5.0 * success
            - x.home_dist * held * lifted
            + 3.0 * (float(s.prev_home_dist) - x.home_dist) * held * lifted
            - 0.5 * low_zone - 2.0 * x.vis_excess - 0.2 * x.vis_out
            - 0.01 * np.sum((action - np.asarray(s.prev_action)) ** 2)
            - 0.1 * x.near_limit - 0.01 * np.sum(action ** 2))


def random_inputs(rng):
    grip_open = float(rng.random() < 0.5)
    return PickInputs(
        d_xy=rng.uniform(0, 0.04), dz=rng.uniform(-0.04, 0.1), cube_lift=rng.uniform(-0.01, 0.13),
        gripper_h=rng.uniform(0.02, 0.2), grip_open=grip_open,
        jaw_stalled=(1 - grip_open) * float(rng.random() < 0.8), home_dist=rng.uniform(0, 0.4),
        vis_excess=rng.uniform(0, 0.2), vis_out=float(rng.integers(0, 3)),
        near_limit=float(rng.integers(0, 3)), action=rng.uniform(-1, 1, 6).astype(np.float32),
        # The later gates must have no effect, so feed values they would reject.
        cube_tilt=1.0, joint_speed=5.0, opposing_sides=0.0)


def test_first_run_preset_matches_first_run_formula():
    cfg = PickRewardConfig.first_run()
    rng = np.random.default_rng(0)
    n_held = 0
    for i in range(400):
        x = random_inputs(rng)
        s = init_pick_state(x)._replace(
            prev_d_xy=jnp.float32(rng.uniform(0, 0.04)), min_dz=jnp.float32(rng.uniform(0, 0.1)),
            was_open=jnp.float32(rng.random() < 0.5), topdown=jnp.float32(rng.random() < 0.7),
            prev_home_dist=jnp.float32(rng.uniform(0, 0.4)),
            prev_action=jnp.asarray(rng.uniform(-1, 1, 6), jnp.float32))
        reward, terms, _ = pick_reward(cfg, s, x)
        n_held += int(terms["held"] > 0)
        assert np.isclose(float(reward), first_run_reward(s, x), atol=1e-4), i
    assert n_held > 5    # the grasp, lift and success branches really ran
