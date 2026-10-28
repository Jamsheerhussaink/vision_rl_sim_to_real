"""CPU regressions for approach reward and sustained lift success."""

import math
import unittest
from types import SimpleNamespace

import torch

from isaaclab.utils.math import matrix_from_quat

from isaac_vision_rl.tasks.pickplace.mdp.rewards import (
    CubeTableDisplacementPenalty,
    GraspCubeProgress,
    LiftCubeProgress,
    LiftHoldBonus,
    LiftSuccess,
    ReachCubeProgress,
    bilateral_cube_contact,
    camera_margin_cost,
    compute_cube_upright,
    compute_cube_upright_cosine,
    compute_fixed_pad_pos,
    compute_grasp_height_gate,
    compute_pinch_midpoint,
    compute_reward_upright,
    compute_sidewall_depth_gate,
    compute_table_camping_discount,
    compute_wrist_vertical_gate,
    contact_grade_shaped,
    cube_tilt_penalty,
    dense_rgb_approach,
    grasp_approach_shaped,
    grasp_close_shaped,
    grasp_cube,
    gripper_camera_boundary_penalty,
    gripper_camera_margin_cost,
    lift_cube_upward,
    object_ee_distance,
    object_goal_distance,
    pinch_force_bonus,
    premature_gripper_closure_penalty,
)
from isaac_vision_rl.tasks.pickplace.mdp.terminations import (
    gripper_outside_camera_view,
    object_outside_camera_view,
)


class JointIds:
    def __init__(self, values):
        self.torch = torch.tensor(values)

    def __len__(self):
        return len(self.torch)


class MockContactData:
    """Mock contact reporting that places fingertip contact points within CAD bounds."""

    def __init__(self, name, robot, normal_force):
        self.name = name
        self.robot = robot
        self.normal_force_matrix_w = normal_force
        self._contact_pos_w = None

    @property
    def contact_pos_w(self):
        if self._contact_pos_w is not None:
            return self._contact_pos_w
        body_id = 1 if self.name == "fixed_finger_contact" else 2
        pos = self.robot.data.body_pos_w.torch[:, body_id]
        quat = self.robot.data.body_quat_w.torch[:, body_id]
        rot = matrix_from_quat(quat)
        local_offset = torch.tensor(
            [[-0.015, 0.0, -0.075]] if self.name == "fixed_finger_contact" else [[0.0, -0.065, 0.02]],
            dtype=pos.dtype,
            device=pos.device,
        )
        return (pos[:, None, :] + torch.bmm(local_offset.expand(pos.shape[0], 1, 3), rot.transpose(1, 2))).unsqueeze(1)

    @contact_pos_w.setter
    def contact_pos_w(self, val):
        self._contact_pos_w = val


def make_env():
    cube = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.425], [0.0, 0.0, 0.425]])),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])),
            root_lin_vel_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.625], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]] * 2)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]] * 2)),
            joint_pos=SimpleNamespace(torch=torch.tensor([[0.3], [1.2]])),
            joint_vel=SimpleNamespace(torch=torch.zeros(2, 1)),
        ),
        find_bodies=lambda name: ([{"gripper_frame_link": 0, "gripper_link": 1, "moving_jaw_so101_v1_link": 2}[name]], [name]),
        find_joints=lambda _: ([0], ["gripper"]),
    )
    class Scene(dict):
        env_origins = torch.zeros(2, 3)
    scene = Scene(robot=robot, cube=cube)
    from isaac_vision_rl.utils.hand_table import TABLE_SENSORS
    for name in TABLE_SENSORS:
        scene[name] = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=torch.zeros(2, 1, 1, 3)))
    for name in ("fixed_finger_contact", "moving_finger_contact"):
        scene[name] = SimpleNamespace(
            data=MockContactData(
                name,
                robot,
                torch.tensor([[[[1.0, 0.0, 0.0]]]] * 2) * (1 if name == "fixed_finger_contact" else -1),
            )
        )
    return SimpleNamespace(num_envs=2, device="cpu", step_dt=0.04, scene=scene,
                           _hand_table_vertices=[(0, torch.zeros(1, 3))])


class PickLiftLogicTests(unittest.TestCase):
    def test_dense_reach_improves_with_distance_and_is_bounded(self):
        env = make_env()
        far = dense_rgb_approach(env)
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] -= 0.15
        near = dense_rgb_approach(env)
        self.assertTrue(torch.all(near > far))
        self.assertTrue(torch.all((near >= 0) & (near <= 1)))

    def test_dense_support_requires_both_fingers(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.425
        self.assertTrue(torch.all(dense_rgb_approach(env, "support") > 0))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        self.assertTrue(torch.equal(dense_rgb_approach(env, "support"), torch.zeros(2)))

    def test_rgb_distance_credit_is_independent_of_control_frequency(self):
        credits = []
        for fps in (25, 60):
            env = make_env()
            env.step_dt = 1 / fps
            reward = ReachCubeProgress(SimpleNamespace(), env)
            reward(env, time_normalized=True)
            env.scene["robot"].data.body_pos_w.torch[:, :, 2] -= 0.01
            credits.append(3 * env.step_dt * reward(env, time_normalized=True))
            self.assertTrue(torch.equal(reward(env, time_normalized=True), torch.zeros(2)))
            env.scene["robot"].data.body_pos_w.torch[:, :, 2] += 0.01
            self.assertTrue(torch.equal(reward(env, time_normalized=True), torch.zeros(2)))
            env.scene["robot"].data.body_pos_w.torch[:, :, 2] -= 0.01
            self.assertTrue(torch.equal(reward(env, time_normalized=True), torch.zeros(2)))
        torch.testing.assert_close(credits[0], credits[1])
        torch.testing.assert_close(credits[0], torch.full((2,), 0.6))

    def test_one_finger_touch_and_stationary_grasp_cannot_farm_reward(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.425])
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        reward = GraspCubeProgress(SimpleNamespace(), env)
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w[:] = torch.tensor([-1.,0.,0.])
        self.assertTrue((reward(env) > 0).all())
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        reward(env)
        env.scene["moving_finger_contact"].data.normal_force_matrix_w[:] = torch.tensor([-1.,0.,0.])
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        reward.reset(torch.tensor([0]))
        self.assertEqual((reward(env) > 0).tolist(), [True, False])

    def test_contact_outside_fingertip_bounds_is_rejected(self):
        env = make_env()
        self.assertTrue(torch.all(bilateral_cube_contact(env)))
        # Move fixed finger contact point outside CAD fingertip bounds (e.g. up near motor/wrist)
        env.scene["fixed_finger_contact"].data.contact_pos_w = torch.tensor([[[[0.0, 0.0, 0.50]]]] * 2)
        self.assertFalse(torch.any(bilateral_cube_contact(env)))

    def test_non_opposing_contact_forces_are_rejected(self):
        env = make_env()
        self.assertTrue(torch.all(bilateral_cube_contact(env)))
        # Both forces pointing in the same direction (e.g. pushing cube rather than clamping)
        env.scene["moving_finger_contact"].data.normal_force_matrix_w[:] = torch.tensor([1.0, 0.0, 0.0])
        self.assertFalse(torch.any(bilateral_cube_contact(env)))

    def test_hovering_and_one_finger_lift_do_not_earn_lift_progress(self):
        env = make_env()
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.49
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.49
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        reward = LiftCubeProgress(SimpleNamespace(), env)
        self.assertTrue((reward(env) > 0).all())
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        self.assertFalse(LiftSuccess(SimpleNamespace(), env)(env, hold_time_s=0.04).any())

    def test_physics_health_rejects_nan_instead_of_hiding_it(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from isaac_vision_rl.tasks.pickplace.mdp.health import assert_finite_physics

        env = make_env()
        self.assertEqual(assert_finite_physics(env).tolist(), [False, False])
        env.scene["robot"].data.joint_pos.torch[0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as folder, patch("isaac_vision_rl.tasks.pickplace.mdp.health.Path") as path:
            path.return_value.resolve.return_value.parents.__getitem__.return_value = Path(folder)
            with self.assertRaisesRegex(FloatingPointError, "robot/joint_pos"):
                assert_finite_physics(env)
            self.assertTrue((Path(folder) / "outputs/pipeline/invalid_physics_state.json").exists())

    def test_direct_approach_needs_no_overhead_waypoint(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.10, 0.0, 0.425])
        reward = ReachCubeProgress(SimpleNamespace(), env)
        reward(env)
        env.scene["robot"].data.body_pos_w.torch[:, 0, 0] = 0.05
        torch.testing.assert_close(reward(env), torch.ones(2))
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))

    def test_hover_and_backtracking_do_not_earn_progress(self):
        env = make_env()
        reward = ReachCubeProgress(SimpleNamespace(), env)
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] -= 0.05
        torch.testing.assert_close(reward(env), torch.ones(2))
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] += 0.05
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] -= 0.05
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))

    def test_partial_reset_does_not_award_initial_distance(self):
        env = make_env()
        reward = ReachCubeProgress(SimpleNamespace(), env)
        reward(env)
        reward.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))

    def test_reach_reward_prefers_grasp_distance_to_high_hover(self):
        env = make_env()
        hover = object_ee_distance(env)
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.445
        close = object_ee_distance(env)
        self.assertTrue(torch.all(close > 20 * hover))

    def test_side_contact_gets_no_grasp_or_lift_credit(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.045, 0.0, 0.425])
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        self.assertTrue(torch.equal(grasp_cube(env), torch.zeros(2)))
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.46
        self.assertTrue(torch.equal(lift_cube_upward(env), torch.zeros(2)))

    def test_gripper_must_stay_open_until_centered(self):
        env = make_env()
        # Far from the cube: a closed gripper is penalized, an open one is not.
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        self.assertTrue(torch.all(premature_gripper_closure_penalty(env) > 0))
        env.scene["robot"].data.joint_pos.torch[:] = 1.2
        self.assertTrue(torch.equal(premature_gripper_closure_penalty(env), torch.zeros(2)))
        # Centered around the cube: closure is now allowed and rewarded.
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.425])
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        self.assertTrue(torch.equal(premature_gripper_closure_penalty(env), torch.zeros(2)))
        self.assertTrue(torch.all(grasp_cube(env) > 0))

    def test_slow_table_drag_accumulates_penalty(self):
        env = make_env()
        penalty = CubeTableDisplacementPenalty(SimpleNamespace(), env)
        penalty.reset()
        env.scene["cube"].data.root_pos_w.torch[:, 0] += 0.004
        self.assertTrue(torch.equal(penalty(env), torch.zeros(2)))
        env.scene["cube"].data.root_pos_w.torch[:, 0] += 0.021
        self.assertTrue(torch.all(penalty(env) > 0))
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.47
        self.assertTrue(torch.equal(penalty(env), torch.zeros(2)))

    def test_lift_requires_continuous_hold_and_closed_gripper(self):
        env = make_env()
        # Calibrated cube-loaded jaws remain near 0.97 rad, while 1.2 is open.
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.97
        success = LiftSuccess(SimpleNamespace(), env)
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.53
        for _ in range(4):
            result = success(env, hold_time_s=0.2)
            self.assertEqual(result.dtype, torch.bool)
            self.assertFalse(result.any())
        self.assertEqual(success(env, hold_time_s=0.2).tolist(), [True, False])
        # Excessive velocity (> 0.45 m/s) breaks the hold
        env.scene["cube"].data.root_lin_vel_w.torch[0, 0] = 0.50
        self.assertFalse(success(env, hold_time_s=0.2)[0])
        # Stabilizing again allows leaky accumulator to quickly recover
        env.scene["cube"].data.root_lin_vel_w.torch.zero_()
        self.assertTrue(success(env, hold_time_s=0.2)[0])
        success.reset(torch.tensor([0]))
        self.assertFalse(success(env).any())

    def test_hold_bonus_pays_only_for_a_new_longest_hold_in_both_stages(self):
        from isaac_vision_rl.tasks.pickplace.config.so101 import env_cfg

        env = make_env()
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.97
        bonus = LiftHoldBonus(SimpleNamespace(), env)
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.53
        kw = dict(minimal_height=0.50, hold_time_s=10.0)
        # A growing hold pays one step each time it reaches a new longest count
        self.assertEqual(bonus(env, **kw).tolist(), [1.0, 0.0])
        self.assertEqual(bonus(env, **kw).tolist(), [1.0, 0.0])
        # Break the hold and restore it: the count climbs back but never beats the best, so nothing is paid
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.43
        self.assertEqual(bonus(env, **kw).tolist(), [0.0, 0.0])
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        self.assertEqual(bonus(env, **kw).tolist(), [0.0, 0.0])
        # A new try starts again from zero
        bonus.reset(torch.tensor([0]))
        self.assertEqual(bonus(env, **kw).tolist(), [1.0, 0.0])
        # The place stage must not carry the bonus (it does not end at the lift)
        self.assertIsNotNone(env_cfg.PickLiftRewardsCfg().lift_hold)
        self.assertIsNotNone(env_cfg.PickplaceRewardsCfg().lift_hold)

    def test_lift_success_logs_which_hold_condition_fails(self):
        env = make_env()
        env.extras = {}
        # Env 0 has the jaws closed on the cube; env 1 has them open
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.97
        env.scene["robot"].data.joint_pos.torch[1, 0] = 1.3
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.53
        LiftSuccess(SimpleNamespace(), env)(env, minimal_height=0.50)
        log = env.extras["log"]
        self.assertEqual(float(log["Hold/above_height_fraction"]), 1.0)
        self.assertEqual(float(log["Hold/jaw_closed"]), 0.5)
        self.assertEqual(float(log["Hold/cube_upright"]), 1.0)
        # Cube below the success height: no env is counted, and the log stays finite
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.43
        LiftSuccess(SimpleNamespace(), env)(env, minimal_height=0.50)
        self.assertEqual(float(env.extras["log"]["Hold/above_height_fraction"]), 0.0)
        self.assertEqual(float(env.extras["log"]["Hold/all"]), 0.0)

    def test_resting_cube_on_table_earns_zero_lift_reward(self):
        env = make_env()
        # Gripper centered and clamped around cube resting at table level (0.425m)
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.425])
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        # Even with wrist elevated or stationary clamp, resting table cube earns zero lift reward
        self.assertTrue(torch.equal(lift_cube_upward(env), torch.zeros(2)))

    def test_tilted_cube_corner_rocking_earns_only_a_small_lift_reward(self):
        env = make_env()
        # Gripper clamped around cube tilted 35 deg on table (z_cube = 0.434m)
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.434])
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.434
        # 35 deg rotation around Y in xyzw: sin(17.5deg)=0.3007058, cos(17.5deg)=0.953717 -> rot[2, 2]=0.8192
        env.scene["cube"].data.root_quat_w.torch[:] = torch.tensor([0.0, 0.3007058, 0.0, 0.953717])
        self.assertAlmostEqual(float(compute_cube_upright_cosine(env)[0]), 0.8192, places=3)
        self.assertEqual(float(compute_cube_upright(env)[0]), 0.0)
        rocking = lift_cube_upward(env)

        # The same grip really lifted 7 cm pays far more than rocking the tipped cube on the table
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.50])
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.50
        env.scene["cube"].data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        lifted = lift_cube_upward(env)
        self.assertTrue(torch.all(lifted > 0.0))
        self.assertTrue(torch.all(lifted > 5.0 * rocking))

    def test_wrist_vertical_gate_and_table_camping_discount(self):
        env = make_env()
        # 1. Vertical wrist pointing straight down earns full wrist gate
        self.assertEqual(float(compute_wrist_vertical_gate(env)[0]), 1.0)
        # 2. Stationary table camping (z < 0.438m, vz = 0) is discounted by 80% (scale 0.20)
        self.assertAlmostEqual(float(compute_table_camping_discount(env)[0]), 0.20, places=2)
        # 3. Active upward pulling (vz > 0.02 m/s) earns 100% full scale
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.03
        self.assertEqual(float(compute_table_camping_discount(env)[0]), 1.0)
        # 4. Elevated cube (z >= 0.440m) earns 100% full scale
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.0
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.460
        self.assertEqual(float(compute_table_camping_discount(env)[0]), 1.0)

    def test_rebalanced_grasp_height_gate_and_upward_pull_shaping(self):
        env = make_env()
        # Configure bilateral contact and centered gripper around cube at sidewall depth
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.420
        env.scene["robot"].data.joint_pos.torch[:] = 0.95  # closed gripper
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.426  # table rest height
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.05  # active upward pull

        # Grasp reward at table rest (z=0.426) has height_gate = 0.60 (grasp_cube pays only new bests, so this is safe)
        table_grasp = float(grasp_cube(env)[0])
        self.assertGreater(table_grasp, 0.0)

        # Grasp reward elevated at target (z=0.465) has height_gate = 1.00 (1 / 0.60 times higher)
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.465 - 0.006
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.465
        elevated_grasp = float(grasp_cube(env)[0])
        self.assertAlmostEqual(elevated_grasp / table_grasp, 1.0 / 0.60, places=3)

        # Lift reward with upward pull vs downward pull
        if not hasattr(env.scene["robot"].data, "body_lin_vel_w"):
            env.scene["robot"].data.body_lin_vel_w = SimpleNamespace(torch=torch.zeros(2, 3, 3))
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.440
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.434  # centered at sidewall depth
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.10
        env.scene["robot"].data.body_lin_vel_w.torch[:, 0, 2] = 0.10
        pulling_up_reward = float(lift_cube_upward(env)[0])

        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = -0.05
        env.scene["robot"].data.body_lin_vel_w.torch[:, 0, 2] = -0.05
        pulling_down_reward = float(lift_cube_upward(env)[0])
        self.assertGreater(pulling_up_reward, pulling_down_reward)

    def test_grasp_height_gate_floor_is_configurable_and_defaults_to_old_value(self):
        env = make_env()
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.426
        self.assertAlmostEqual(float(compute_grasp_height_gate(env)[0]), 0.20, places=5)
        self.assertAlmostEqual(float(compute_grasp_height_gate(env, floor=0.60)[0]), 0.60, places=5)
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.465
        self.assertAlmostEqual(float(compute_grasp_height_gate(env, floor=0.60)[0]), 1.0, places=5)

    def test_grasp_cube_judges_hand_angle_once_through_wrist_gate(self):
        from unittest import mock

        from isaac_vision_rl.tasks.pickplace.mdp import rewards

        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.420
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.426
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.05
        baseline = grasp_cube(env)
        self.assertTrue(torch.all(baseline > 0.0))
        # The strict exp() alignment check must no longer multiply into grasp_cube
        with mock.patch.object(rewards, "_tool_downward_alignment", return_value=torch.zeros(2)):
            self.assertTrue(torch.equal(grasp_cube(env), baseline))

    def test_lift_goal_is_one_setting_shared_by_reward_and_termination(self):
        from isaac_vision_rl.tasks.pickplace.config.so101 import env_cfg

        rewards = env_cfg.PickLiftRewardsCfg()
        done = env_cfg.PickLiftTerminationsCfg()
        # The success bonus and the "try ends on success" check must never disagree
        for params in (rewards.lift_completed.params, done.lift_success.params):
            self.assertEqual(params["minimal_height"], env_cfg.LIFT_SUCCESS_HEIGHT_M)
            self.assertEqual(params["hold_time_s"], env_cfg.LIFT_HOLD_TIME_S)
        self.assertEqual(rewards.lift_cube.params["target_height"], env_cfg.LIFT_TARGET_HEIGHT_M)
        # Goal: cube 10 cm above its 0.426 m resting height held 0.5 s; dense reward aims at 12 cm
        self.assertAlmostEqual(env_cfg.LIFT_SUCCESS_HEIGHT_M - 0.426, 0.10, places=6)
        self.assertAlmostEqual(env_cfg.LIFT_TARGET_HEIGHT_M - 0.426, 0.12, places=6)
        self.assertEqual(env_cfg.LIFT_HOLD_TIME_S, 0.5)

    def test_lift_reward_keeps_growing_up_to_the_configured_target_height(self):
        env = make_env()
        env.scene["robot"].data.joint_pos.torch[:] = 0.95

        def reward_at(z, **kwargs):
            env.scene["cube"].data.root_pos_w.torch[:, 2] = z
            env.scene["robot"].data.body_pos_w.torch[:, :, 2] = z - 0.006
            return float(lift_cube_upward(env, **kwargs)[0])

        # Default target (0.48 m): the reward has already stopped growing by 0.48 m
        self.assertAlmostEqual(reward_at(0.505), reward_at(0.480), places=5)
        # 12 cm target (0.546 m): the policy is still paid more for every extra centimetre
        self.assertGreater(reward_at(0.505, target_height=0.546), reward_at(0.480, target_height=0.546))
        self.assertGreater(reward_at(0.526, target_height=0.546), reward_at(0.505, target_height=0.546))

    def test_rewards_pay_less_for_a_tipped_cube_but_still_pay_for_a_tilted_lift(self):
        def tilt(env, degrees):
            half = math.radians(degrees) / 2
            env.scene["cube"].data.root_quat_w.torch[:] = torch.tensor([math.sin(half), 0.0, 0.0, math.cos(half)])

        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.420
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.426
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.05

        # Reward ramp: 1 at 18 deg or less, about 0.66 at 30 deg, almost 0 at 45 deg, 0 at 60 deg
        for degrees, expected in ((0, 1.0), (18, 1.0), (30, 0.6641), (45, 0.0284), (60, 0.0)):
            tilt(env, degrees)
            self.assertAlmostEqual(float(compute_reward_upright(env)[0]), expected, places=3)

        # A tilted cube still pays (a corner lift is accepted), but less than an upright one
        tilt(env, 0)
        upright_grasp = grasp_cube(env)
        self.assertTrue(torch.all(upright_grasp > 0.0))
        self.assertTrue(torch.all(pinch_force_bonus(env) > 0.0))
        tilt(env, 30)
        self.assertTrue(torch.all(grasp_cube(env) > 0.0))
        self.assertTrue(torch.all(grasp_cube(env) < upright_grasp))
        # A cube lying nearly on its side (60 deg) earns no grab, squeeze or contact points
        tilt(env, 60)
        self.assertTrue(torch.equal(grasp_cube(env), torch.zeros(2)))
        self.assertTrue(torch.equal(pinch_force_bonus(env), torch.zeros(2)))
        self.assertTrue(torch.equal(contact_grade_shaped(env), torch.zeros(2)))

    def test_touch_score_grows_with_finger_force(self):
        from isaac_vision_rl.tasks.pickplace.mdp.rewards import _touch_ramp

        forces = torch.tensor([0.0, 0.005, 0.025, 0.05, 1.0])
        expected = torch.tensor([0.0, 0.1, 0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(_touch_ramp(forces), expected, atol=1e-6))

    def test_tipping_penalty_weight_is_one(self):
        from isaac_vision_rl.tasks.pickplace.config.so101 import env_cfg

        self.assertEqual(env_cfg.PickLiftRewardsCfg().cube_tilt.weight, -1.0)

    def test_wrist_yank_without_cube_motion_earns_no_extra_lift_reward(self):
        env = make_env()
        env.scene["robot"].data.body_lin_vel_w = SimpleNamespace(torch=torch.zeros(2, 3, 3))
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.440
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.434
        still_reward = lift_cube_upward(env)
        self.assertTrue(torch.all(still_reward > 0.0))

        # Only the wrist shoots upward; the cube does not move -> no move-up credit
        env.scene["robot"].data.body_lin_vel_w.torch[:, 0, 2] = 0.30
        self.assertTrue(torch.allclose(lift_cube_upward(env), still_reward))

    def test_table_camping_discount_is_applied_once_to_pinch_force(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.420
        env.scene["robot"].data.joint_pos.torch[:] = 0.95
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.426

        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.05  # rising: no discount
        rising = float(pinch_force_bonus(env)[0])
        env.scene["cube"].data.root_lin_vel_w.torch[:, 2] = 0.0  # stationary on table: 0.20 discount
        camping = float(pinch_force_bonus(env)[0])
        self.assertGreater(rising, 0.0)
        # A single 0.20 cut, not 0.20 * 0.20 = 0.04
        self.assertAlmostEqual(camping / rising, 0.20, places=3)

    def test_table_contact_disqualifies_an_otherwise_valid_lift(self):
        env = make_env()
        env.scene["robot"].data.joint_pos.torch[:] = 0.97
        # At border height before elevation bypass (z_cube < 0.455), excessive table force (>0.5N) disqualifies
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.450
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.46
        env.scene["fixed_table_contact"].data.normal_force_matrix_w[:, 0, 0, 2] = 1.0
        success = LiftSuccess(SimpleNamespace(), env)
        for _ in range(20):
            self.assertFalse(success(env, minimal_height=0.44).any())
        # Once elevated (>= 0.455m), table contact sensor noise is safely bypassed
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.53
        for _ in range(20):
            res = success(env, hold_time_s=0.2)
        self.assertTrue(res[0])

    def test_object_goal_distance_rewards_transport_when_holding_cube(self):
        env = make_env()
        box = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[0.05, 0.26, 0.401], [0.05, 0.26, 0.401]])))
        )
        env.scene["box"] = box
        # Loaded grasp: jaw at 0.97 rad, bilateral contact active
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.97
        # Position cube lifted at 0.48m near TCP at 0.48m
        env.scene["cube"].data.root_pos_w.torch[0] = torch.tensor([0.05, 0.10, 0.48])
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.05, 0.10, 0.48])
        reward = object_goal_distance(env)
        self.assertTrue(reward[0] > 0.0)
        # When cube is closer to box, reward should increase
        env.scene["cube"].data.root_pos_w.torch[0] = torch.tensor([0.05, 0.22, 0.48])
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.05, 0.22, 0.48])
        closer_reward = object_goal_distance(env)
        self.assertTrue(closer_reward[0] > reward[0])
        # When cube drops below minimal height, reward is 0
        env.scene["cube"].data.root_pos_w.torch[0, 2] = 0.42
        self.assertEqual(float(object_goal_distance(env)[0]), 0.0)

    def test_carry_pay_ramps_from_the_box_wall_to_ten_centimetres(self):
        env = make_env()
        box = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[0.05, 0.26, 0.401], [0.05, 0.26, 0.401]])))
        )
        env.scene["box"] = box
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.97

        def pay(z):
            env.scene["cube"].data.root_pos_w.torch[0] = torch.tensor([0.05, 0.22, z])
            env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.05, 0.22, z])
            return float(object_goal_distance(env, minimal_height=0.49, full_height=0.526)[0])

        self.assertEqual(pay(0.45), 0.0)  # held low, pushing along the table
        self.assertEqual(pay(0.49), 0.0)  # level with the wall top
        full = pay(0.526)
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(pay(0.508), 0.5 * full, places=4)
        self.assertAlmostEqual(pay(0.60), full, places=4)  # no extra pay above 10 cm

    def test_place_stage_lift_pull_reaches_over_the_wall(self):
        from isaac_vision_rl.tasks.pickplace.config.so101 import env_cfg

        rewards = env_cfg.PickplaceRewardsCfg()
        self.assertEqual(rewards.lift_cube.params["target_height"], env_cfg.LIFT_TARGET_HEIGHT_M)
        place = rewards.place_cube.params
        self.assertEqual(place["minimal_height"], env_cfg.BOX_WALL_TOP_M)
        self.assertEqual(place["full_height"], env_cfg.LIFT_SUCCESS_HEIGHT_M)

    def test_premature_gripper_closure_penalty_penalizes_high_or_misaligned_closing(self):
        env = make_env()
        # TCP is placed 5 cm above cube center (too high)
        env.scene["cube"].data.root_pos_w.torch[0] = torch.tensor([0.0, 0.0, 0.425])
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.0, 0.0, 0.475])
        # Closing gripper: 0.3 rad (closed)
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.30
        pen_high = premature_gripper_closure_penalty(env)
        self.assertGreater(float(pen_high[0]), 0.0)

        # Now descend deep: TCP is 0.5 cm above cube center (deep enough, <= 1 cm)
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.0, 0.0, 0.430])
        pen_deep = premature_gripper_closure_penalty(env)
        self.assertEqual(float(pen_deep[0]), 0.0)

        # Gripper open: 1.2 rad -> penalty is always 0
        env.scene["robot"].data.joint_pos.torch[0, 0] = 1.20
        pen_open = premature_gripper_closure_penalty(env)
        self.assertEqual(float(pen_open[0]), 0.0)

    def test_pinch_force_bonus_requires_bilateral_contact_and_scales_with_force(self):
        env = make_env()
        # Position gripper at sidewall grasp depth (0.418m)
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.418])
        # Gripper at 0.98 rad (closed on cube)
        env.scene["robot"].data.joint_pos.torch[0, 0] = 0.98
        bonus = pinch_force_bonus(env)
        self.assertGreater(float(bonus[0]), 0.0)

        # Break bilateral contact: zero moving finger force
        env.scene["moving_finger_contact"].data.normal_force_matrix_w = torch.zeros(2, 1, 1, 3)
        bonus_no_contact = pinch_force_bonus(env)
        self.assertEqual(float(bonus_no_contact[0]), 0.0)

    def test_gripper_outside_camera_view_detects_out_of_frustum(self):
        env = make_env()
        # Env 0: EE inside normal camera view
        env.scene["robot"].data.body_pos_w.torch[0, :] = torch.tensor([-0.063, 0.021, 0.455])
        # Env 1: EE way out of view (e.g. far to the right X=1.5m)
        env.scene["robot"].data.body_pos_w.torch[1, :] = torch.tensor([1.5, 0.0, 0.455])

        out = gripper_outside_camera_view(env, margin=0.02)
        self.assertFalse(bool(out[0]))
        self.assertTrue(bool(out[1]))

    def test_object_outside_camera_view_detects_lost_cube(self):
        env = make_env()
        # Env 0: Cube in nominal position
        env.scene["cube"].data.root_pos_w.torch[0] = torch.tensor([-0.002, -0.052, 0.426])
        # Env 1: Cube knocked off camera (X=2.0m)
        env.scene["cube"].data.root_pos_w.torch[1] = torch.tensor([2.0, -0.052, 0.426])

        out = object_outside_camera_view(env)
        self.assertFalse(bool(out[0]))
        self.assertTrue(bool(out[1]))

    def test_gripper_camera_boundary_penalty_repels_edges(self):
        env = make_env()
        # Center of camera view: zero boundary penalty
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([-0.063, 0.021, 0.455])
        pen_center = gripper_camera_boundary_penalty(env, safe_margin=0.08)
        self.assertEqual(float(pen_center[0]), 0.0)

        # Encroaching near camera boundary edge: positive penalty
        env.scene["robot"].data.body_pos_w.torch[0, 0] = torch.tensor([0.45, 0.021, 0.455])
        pen_edge = gripper_camera_boundary_penalty(env, safe_margin=0.08)
        self.assertGreater(float(pen_edge[0]), 0.0)

    def test_new_camera_cost_tracks_wrist_and_both_finger_pads(self):
        env=make_env()
        env.scene['robot'].data.body_pos_w.torch[0,:,:]=torch.tensor([-.063,.021,.455])
        safe=gripper_camera_margin_cost(env)
        env.scene['robot'].data.body_pos_w.torch[0,:,:]=torch.tensor([.45,.021,.455])
        edge=gripper_camera_margin_cost(env)
        self.assertEqual(float(safe[0]),0.)
        self.assertGreater(float(edge[0]),float(safe[0]))

    def test_camera_margin_cost_is_zero_centered_and_grows_toward_edges(self):
        u=torch.tensor([[.50,.50,.50],[.04,.50,.50],[0.0,.50,.50]])
        v=torch.full_like(u,.50)
        z=torch.full_like(u,.50)
        cost=camera_margin_cost(u,v,z,safe_margin=.08)
        self.assertEqual(float(cost[0]),0.0)
        self.assertGreater(float(cost[1]),0.0)
        self.assertGreater(float(cost[2]),float(cost[1]))
        invisible=camera_margin_cost(torch.full((1,1),.5),torch.full((1,1),.5),torch.tensor([[.01]]))
        self.assertEqual(float(invisible[0]),1.0)

    def test_arm_targets_span_limits_and_gripper_can_open_and_close(self):
        from isaaclab.utils.string import string_to_callable

        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import ActionsCfg

        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        limits = torch.tensor([[[-1.8, 1.8]] * 5 + [[-0.17, 1.74]]] * 2)
        robot = SimpleNamespace(
            num_joints=6,
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros(2, 6)),
                soft_joint_pos_limits=SimpleNamespace(torch=limits),
            ),
            find_joints=lambda requested, **_: (JointIds([names.index(n) for n in requested]), requested),
        )
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.04, scene={"robot": robot})
        cfg = ActionsCfg()
        arm = string_to_callable(cfg.arm_action.class_type)(cfg.arm_action, env)
        arm.reset()
        commands = torch.tensor([[100.0] * 5, [-100.0] * 5])
        for _ in range(80):
            previous = arm.processed_actions.clone()
            arm.process_actions(commands)
            self.assertTrue(torch.all((arm.processed_actions - previous).abs() <= 0.060001))
            self.assertTrue(torch.all(arm.processed_actions.abs() <= 1.8))
        torch.testing.assert_close(arm.processed_actions[0], torch.full((5,), 1.8), atol=0.001, rtol=0)
        torch.testing.assert_close(arm.processed_actions[1], torch.full((5,), -1.8), atol=0.001, rtol=0)
        robot.data.joint_pos.torch[0, :5] = 0.25
        before_other = arm.processed_actions[1].clone()
        arm.reset(torch.tensor([0]))
        torch.testing.assert_close(arm.processed_actions[0], torch.full((5,), 0.25))
        torch.testing.assert_close(arm.processed_actions[1], before_other)
        arm.process_actions(torch.zeros(2, 5))
        torch.testing.assert_close(arm.processed_actions[0], torch.full((5,), 0.20))
        gripper = string_to_callable(cfg.gripper_action.class_type)(cfg.gripper_action, env)
        gripper.process_actions(torch.zeros(2, 1))
        torch.testing.assert_close(gripper.processed_actions, torch.full((2, 1), 0.775))
        gripper.process_actions(torch.tensor([[-100.0], [100.0]]))
        # Closed command stops at the configured jaw floor (0.0 = old behaviour; 0.7 = real cube-stall floor).
        floor = cfg.gripper_action.min_close_target
        torch.testing.assert_close(gripper.processed_actions, torch.tensor([[floor], [1.55]]))

    def test_fixed_finger_pad_z_height_and_sidewall_depth(self):
        env = make_env()
        cube_pos = env.scene["cube"].data.root_pos_w.torch
        cube_pos[:] = torch.tensor([0.05, 0.02, 0.426])

        # 1. Hover height when xy > 0.018m: target height is z_cube + 0.055m = 0.481m
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.15, 0.02, 0.481])
        xy_dist = torch.norm((compute_pinch_midpoint(env) - cube_pos)[:, :2], dim=-1)
        z_target = cube_pos[:, 2] - 0.008 + 0.063 * torch.clamp((xy_dist - 0.006) / 0.012, min=0.0, max=1.0)
        torch.testing.assert_close(z_target, torch.full((2,), 0.481), atol=1e-4, rtol=1e-4)

        # 2. Descended target height when xy <= 0.006m: target height is z_cube - 0.008m = 0.418m
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.05, 0.02, 0.418])
        xy_dist_aligned = torch.norm((compute_pinch_midpoint(env) - cube_pos)[:, :2], dim=-1)
        z_target_aligned = cube_pos[:, 2] - 0.008 + 0.063 * torch.clamp((xy_dist_aligned - 0.006) / 0.012, min=0.0, max=1.0)
        torch.testing.assert_close(z_target_aligned, torch.full((2,), 0.418), atol=1e-4, rtol=1e-4)

        # 3. Fixed finger on roof (Z = 0.451m): gate is 0.0, zeroing contact/grasp rewards
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.05, 0.02, 0.451])
        env.scene["robot"].data.joint_pos.torch[:] = 0.90
        depth_gate_roof = compute_sidewall_depth_gate(env)
        torch.testing.assert_close(depth_gate_roof, torch.zeros(2))
        self.assertEqual(float(contact_grade_shaped(env)[0]), 0.0)
        self.assertEqual(float(grasp_cube(env)[0]), 0.0)
        self.assertEqual(float(pinch_force_bonus(env)[0]), 0.0)
        # Premature close penalty is positive when fixed finger is on roof and closing
        self.assertGreater(float(premature_gripper_closure_penalty(env)[0]), 0.0)

        # 4. Fixed finger at grasp depth (Z = 0.418m): gate is active (> 0.85)
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.05, 0.02, 0.418])
        depth_gate_deep = compute_sidewall_depth_gate(env)
        self.assertTrue(torch.all(depth_gate_deep > 0.85))
        self.assertEqual(float(premature_gripper_closure_penalty(env)[0]), 0.0)

        # 5. Cube tilt check: a cube on its side (60 degrees) drops upright score to 0.0 and triggers penalty
        # 60 deg rotation around X in xyzw: sin(30deg)=0.5, cos(30deg)=0.8660 -> cos(60deg)=0.5 < 0.70
        env.scene["cube"].data.root_quat_w.torch[:] = torch.tensor([0.5, 0.0, 0.0, 0.8660])
        self.assertEqual(float(compute_cube_upright(env)[0]), 0.0)
        self.assertGreater(float(cube_tilt_penalty(env)[0]), 0.0)
        self.assertEqual(float(grasp_cube(env)[0]), 0.0)

    def test_16_env_vectorized_tensor_shapes(self):
        """Verify that all reward terms support 16-environment vectorization without shape mismatches."""
        n = 16
        cube = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=torch.full((n, 3), 0.426)),
                root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)),
                root_lin_vel_w=SimpleNamespace(torch=torch.zeros(n, 3)),
            )
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.418], [0.0, 0.0, 0.418], [0.0, 0.0, 0.418]]] * n)),
                body_quat_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]] * n)),
                joint_pos=SimpleNamespace(torch=torch.full((n, 1), 0.95)),
                joint_vel=SimpleNamespace(torch=torch.zeros(n, 1)),
            ),
            find_bodies=lambda name: ([{"gripper_frame_link": 0, "gripper_link": 1, "moving_jaw_so101_v1_link": 2}[name]], [name]),
            find_joints=lambda _: ([0], ["gripper"]),
        )
        class Scene(dict):
            env_origins = torch.zeros(n, 3)
        scene = Scene(robot=robot, cube=cube)
        from isaac_vision_rl.utils.hand_table import TABLE_SENSORS
        for name in TABLE_SENSORS:
            scene[name] = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=torch.zeros(n, 1, 1, 3)))
        for name in ("fixed_finger_contact", "moving_finger_contact"):
            scene[name] = SimpleNamespace(
                data=MockContactData(
                    name,
                    robot,
                    torch.tensor([[[[1.0, 0.0, 0.0]]]] * n) * (1 if name == "fixed_finger_contact" else -1),
                )
            )
        env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.04, scene=scene,
                               _hand_table_vertices=[(0, torch.zeros(1, 3))])

        terms = [
            ("grasp_approach_shaped", grasp_approach_shaped(env)),
            ("grasp_close_shaped", grasp_close_shaped(env)),
            ("contact_grade_shaped", contact_grade_shaped(env)),
            ("grasp_cube", grasp_cube(env)),
            ("pinch_force_bonus", pinch_force_bonus(env)),
            ("premature_gripper_closure_penalty", premature_gripper_closure_penalty(env)),
            ("lift_cube_upward", lift_cube_upward(env)),
            ("cube_tilt_penalty", cube_tilt_penalty(env)),
            ("dense_reach", dense_rgb_approach(env, "reach")),
            ("dense_align", dense_rgb_approach(env, "align")),
            ("dense_open", dense_rgb_approach(env, "open")),
            ("dense_support", dense_rgb_approach(env, "support")),
        ]
        for name, val in terms:
            self.assertEqual(val.shape, torch.Size([n]), f"Mismatched shape for {name}: {val.shape}")
            self.assertEqual(val.dtype, torch.float32, f"Mismatched dtype for {name}: {val.dtype}")


if __name__ == "__main__":
    unittest.main()
