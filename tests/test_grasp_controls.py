"""Regression checks for top-of-cube pinches, wrist overshoot, and old runs."""
from flax import serialization
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

from vision_rl.envs import SCENE_XML
from vision_rl.envs.grasp_geometry import opposing_finger_sides
from vision_rl.ppo import PPOConfig, make_optimizer
from vision_rl.rewards.pick import PickInputs, PickRewardConfig, init_pick_state, pick_reward
from vision_rl.train import TrainConfig


def test_opposite_faces_accept_yawed_cube_but_reject_top_and_same_side():
    yaw = np.pi / 4
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    cube = np.array([0.239, -0.175, 0.425])
    fixed = cube + rot @ [-0.026, 0, 0]
    moving = cube + rot @ [0.026, 0, 0]
    assert opposing_finger_sides(fixed, moving, cube, rot, xp=np)
    assert not opposing_finger_sides(cube + rot @ [0, 0, 0.027], moving, cube, rot, xp=np)
    assert not opposing_finger_sides(moving, moving, cube, rot, xp=np)
    assert not opposing_finger_sides(cube + rot @ [-0.06, 0, 0], moving, cube, rot, xp=np)


def test_blocked_jaw_on_top_cannot_receive_grasp_or_lift_reward():
    x = PickInputs(d_xy=jnp.float32(0), dz=jnp.float32(0), cube_lift=jnp.float32(.01),
                   gripper_h=jnp.float32(.035), grip_open=jnp.float32(0),
                   jaw_stalled=jnp.float32(1), home_dist=jnp.float32(0),
                   vis_excess=jnp.float32(0), vis_out=jnp.float32(0),
                   near_limit=jnp.float32(0), action=jnp.zeros(6), opposing_sides=jnp.float32(0))
    s = init_pick_state(x)._replace(topdown=jnp.float32(1))
    _, terms, _ = pick_reward(PickRewardConfig(), s, x)
    assert float(terms["held"]) == float(terms["grasp_r"]) == float(terms["lift_r"]) == 0
    _, valid, _ = pick_reward(PickRewardConfig(), s, x._replace(opposing_sides=jnp.float32(1)))
    assert float(valid["held"]) == 1


def test_wrist_stays_below_90_during_repeated_full_range_reversals():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    q = m.jnt_qposadr[m.joint("wrist_roll").id]
    a = m.actuator("wrist_roll").id
    peak = 0.0
    for requested in (10.0, -10.0, 10.0, -10.0):
        for _ in range(500):
            prev = d.ctrl[a]
            goal = np.clip(requested, *m.actuator_ctrlrange[a])
            d.ctrl[a] = prev + np.clip(.4 * (goal - prev), -.035, .035)
            mujoco.mj_step(m, d)
            peak = max(peak, abs(float(d.qpos[q])))
    assert peak < np.pi / 2, f"wrist crossed 90 degrees: {np.degrees(peak):.3f}"


def test_adamw_defaults_and_legacy_optimizer_state():
    cfg = PPOConfig()
    assert cfg.optimizer == "adamw" and cfg.weight_decay == 1e-4 and cfg.total_steps == 50_000_000
    params = {"w": jnp.ones(2)}
    make_optimizer(cfg).init(params)
    old = TrainConfig().to_dict()
    old["ppo"].pop("optimizer")
    old["ppo"].update(weight_decay=1e-4, total_steps=50_000_000)
    legacy = TrainConfig.from_dict(old).ppo
    assert legacy.optimizer == "adamw" and legacy.total_steps == 50_000_000
    n = legacy.num_iterations * legacy.update_epochs * legacy.num_minibatches
    lr = optax.linear_schedule(legacy.learning_rate, legacy.learning_rate * legacy.lr_final_frac, n)
    original = optax.chain(optax.clip_by_global_norm(legacy.max_grad_norm),
                           optax.adamw(lr, eps=1e-5, weight_decay=1e-4)).init(params)
    # Loading the old optimizer state must still work with explicit optimizer selection.
    serialization.from_bytes(make_optimizer(legacy).init(params), serialization.to_bytes(original))
    old["ppo"].pop("weight_decay")
    adam = TrainConfig.from_dict(old).ppo
    assert adam.optimizer == "adam" and adam.weight_decay == 0
    make_optimizer(adam).init(params)


def test_scripted_safe_approach_keeps_wrist_limit_and_episode_deadline():
    from scripts.check_pick import Robot, physics_pick
    rob = Robot()
    result = physics_pick(rob, rob.d.qpos[6:8].copy(), -np.pi / 4,
                          approach="safe", task_height=.10, return_report=True)
    assert result["success"]
    assert result["first_contact"][0] in ("descend", "close")
    assert result["approach_shift_m"] < .01
    assert result["peak_wrist_deg"] < 90
    assert result["elapsed_s"] <= 7.5
    assert result["stable_lift_s"] >= .3
