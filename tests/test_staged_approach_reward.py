"""The real low reset pose must not disable correct open-jaw descent shaping."""
import jax.numpy as jnp
import numpy as np

from vision_rl.rewards.pick import PickInputs, PickRewardConfig, init_pick_state, pick_reward
from vision_rl.train import TrainConfig


def inputs(dz=.01, opened=False, opposing=False, distance=.01):
    f = jnp.float32
    return PickInputs(d_xy=f(distance), dz=f(dz), cube_lift=f(0), gripper_h=f(.025 + dz),
                      grip_open=f(opened), jaw_stalled=f(not opened), home_dist=f(0),
                      vis_excess=f(0), vis_out=f(0), near_limit=f(0),
                      action=jnp.array([0., 0, 0, 0, 0, 1. if opened else -1.]),
                      opposing_sides=f(opposing))


def test_open_bonus_uses_same_height_as_grasp_qualification():
    cfg = PickRewardConfig()
    for dz, bonus, eligible in [(.04, 0., 0.), (.08, .5, 1.)]:
        x = inputs(dz=dz, opened=True)
        _, terms, state = pick_reward(cfg, init_pick_state(x), x)
        assert float(terms["open_r"]) == bonus
        assert float(state.topdown) == eligible
        _, repeated, _ = pick_reward(cfg, state, x)
        assert float(repeated["open_r"]) == 0


def test_low_reset_gets_descent_progress_only_after_valid_open_approach():
    cfg = PickRewardConfig()
    state = init_pick_state(inputs(dz=-.0113, opened=True, distance=.08))
    total = 0.
    for dz in [.08, .06, .04, .02, .01]:
        _, terms, state = pick_reward(cfg, state, inputs(dz=dz, opened=True))
        total += float(terms["descend_pr"])
    np.testing.assert_allclose(total, .21, atol=1e-6)
    # Revisiting hover and the same descent must not refill the reward budget.
    repeated = 0.
    for dz in [.08, .06, .04, .02, .01]:
        _, terms, state = pick_reward(cfg, state, inputs(dz=dz, opened=True))
        repeated += float(terms["descend_pr"]) + float(terms["open_r"])
    assert repeated == 0


def test_early_descent_without_open_approach_cannot_earn_progress():
    cfg = PickRewardConfig()
    state = init_pick_state(inputs(dz=.1))
    for dz in [.08, .04, .01]:
        _, terms, state = pick_reward(cfg, state, inputs(dz=dz))
        assert float(terms["descend_pr"]) == 0
        assert float(terms["held"]) == 0


def test_correct_open_hover_beats_low_closed_press_and_real_grip_is_allowed():
    cfg = PickRewardConfig()
    closed = inputs()
    cs = init_pick_state(closed)._replace(prev_action=closed.action)
    bad, bad_terms, _ = pick_reward(cfg, cs, closed)
    opened = inputs(dz=.08, opened=True)
    _, _, state = pick_reward(cfg, init_pick_state(opened), opened)
    good, _, _ = pick_reward(cfg, state, opened)  # no one-off bonus on this step
    assert float(good) > float(bad)
    assert float(bad_terms["held"]) == 0
    _, grasp, _ = pick_reward(cfg, state, inputs(opposing=True))
    assert float(grasp["held"]) == 1
    assert float(grasp["descend_r"]) == 0


def test_old_configs_keep_legacy_reward_and_new_configs_record_new_rule():
    cfg = TrainConfig()
    assert TrainConfig.from_dict(cfg.to_dict()).env.reward.staged_open_approach
    old = cfg.to_dict()
    old["env"]["reward"].pop("staged_open_approach")
    assert not TrainConfig.from_dict(old).env.reward.staged_open_approach
