"""CPU checks for carrying progress; these do not evaluate a learned policy."""

from types import SimpleNamespace

import pytest
import torch

from test_picklift_logic import make_env
from isaac_vision_rl.tasks.pickplace.mdp.rewards import PlaceCubeProgress


def setup_carry():
    env = make_env()
    env.extras = {}
    env.scene["box"] = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.20, 0.401]] * 2)))
    )
    env.scene["robot"].data.joint_pos.torch[:] = 0.8
    move(env, y=0.0)
    return env, PlaceCubeProgress(SimpleNamespace(), env)


def move(env, y=0.0, z=0.54, ids=slice(None)):
    env.scene["cube"].data.root_pos_w.torch[ids] = torch.tensor([0.0, y, z])
    env.scene["robot"].data.body_pos_w.torch[ids, 0] = torch.tensor([0.0, y, z])


def pay(term, env):
    # Undo RewardManager time scaling, and include the configured weight.
    return term(env, minimal_height=0.49, full_height=0.526) * env.step_dt * 12.0


def test_initialization_lifting_and_stationary_samples_pay_nothing():
    env, term = setup_carry()
    for z in (0.426, 0.49, 0.51, 0.526, 0.55, 0.55, 0.60, 0.54):
        move(env, z=z)
        assert torch.equal(pay(term, env), torch.zeros(2))
    assert env.extras["log"]["Place/best_carry_progress_percent"] == 0


def test_supported_horizontal_progress_pays_only_new_best():
    env, term = setup_carry()
    assert not pay(term, env).any()
    move(env, y=0.04)
    assert (pay(term, env) > 0).all()
    assert not pay(term, env).any()  # stay
    move(env, y=-0.02)
    assert not pay(term, env).any()  # retreat
    move(env, y=0.04)
    assert not pay(term, env).any()  # repeat
    move(env, y=0.08)
    assert (pay(term, env) > 0).all()


@pytest.mark.parametrize("failure", ["low", "open", "far", "one_finger", "same_force_direction", "outside_pad"])
def test_invalid_grasps_cannot_earn_horizontal_credit(failure):
    env, term = setup_carry()
    pay(term, env)
    move(env, y=0.05)
    if failure == "low":
        move(env, y=0.05, z=0.51)
    elif failure == "open":
        env.scene["robot"].data.joint_pos.torch[:] = 1.2
    elif failure == "far":
        env.scene["robot"].data.body_pos_w.torch[:, 0, 0] = 0.08
    elif failure == "one_finger":
        env.scene["moving_finger_contact"].data.normal_force_matrix_w[:] = 0
    elif failure == "same_force_direction":
        env.scene["moving_finger_contact"].data.normal_force_matrix_w *= -1
    else:
        env.scene["moving_finger_contact"].data.contact_pos_w = torch.ones(2, 1, 1, 3) * 10
    assert not pay(term, env).any()


def test_contact_loss_regrasp_and_height_reentry_do_not_bank_credit():
    env, term = setup_carry()
    pay(term, env)
    force = env.scene["moving_finger_contact"].data.normal_force_matrix_w
    force[:] = 0
    move(env, y=0.08)
    assert not pay(term, env).any()
    force[:, 0, 0, 0] = -1.0
    assert not pay(term, env).any()  # regrasp closer to box
    move(env, y=0.10)
    assert (pay(term, env) > 0).all()
    move(env, y=0.13, z=0.50)
    assert not pay(term, env).any()
    move(env, y=0.13)
    assert not pay(term, env).any()  # lift at unchanged XY
    move(env, y=0.14)
    assert (pay(term, env) > 0).all()


def test_partial_reset_does_not_reset_other_robot_or_pay_free_reward():
    env, term = setup_carry()
    pay(term, env)
    move(env, y=0.05)
    pay(term, env)
    term.reset(torch.tensor([0]))
    move(env, y=0.07)
    reward = pay(term, env)
    assert reward[0] == 0 and reward[1] > 0
    term.reset()
    assert not pay(term, env).any()


def test_episode_carry_budget_and_diagnostics():
    env, term = setup_carry()
    total = pay(term, env)
    for y in (0.04, 0.10, 0.16, 0.20, 0.20, 0.0, 0.20):
        move(env, y=y)
        total += pay(term, env)
    assert ((total > 0) & (total <= 60.0)).all()
    assert torch.isclose(env.extras["log"]["Place/best_carry_progress_percent"], torch.tensor(100.0))
    assert env.extras["log"]["Place/cube_box_xy_distance_cm"] == 0
    assert env.extras["log"]["Place/carry_eligible_fraction"] == 1


def test_horizontal_credit_does_not_depend_on_height_or_policy_interval():
    totals = []
    for z, dt in ((0.526, 0.04), (0.60, 0.02)):
        env, term = setup_carry()
        env.step_dt = dt
        move(env, z=z)
        pay(term, env)
        move(env, y=0.10, z=z)
        totals.append(pay(term, env))
    assert torch.allclose(*totals)
