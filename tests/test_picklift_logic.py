"""CPU regressions for approach reward and sustained lift success."""

import unittest
from types import SimpleNamespace

import torch

from isaaclab.utils.math import matrix_from_quat

from isaac_vision_rl.tasks.pickplace.mdp.rewards import (
    CubeTableDisplacementPenalty,
    GraspCubeProgress,
    LiftCubeProgress,
    LiftSuccess,
    ReachCubeProgress,
    bilateral_cube_contact,
    dense_rgb_approach,
    grasp_cube,
    gripper_camera_boundary_penalty,
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
            root_lin_vel_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.625], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]] * 2)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]] * 2)),
            joint_pos=SimpleNamespace(torch=torch.tensor([[0.3], [1.2]])),
        ),
        find_bodies=lambda name: ([{"gripper_frame_link": 0, "gripper_link": 1, "moving_jaw_so101_v1_link": 2}[name]], [name]),
        find_joints=lambda _: ([0], ["gripper"]),
    )
    scene = {"robot": robot, "cube": cube}
    for name in ("fixed_finger_contact", "moving_finger_contact"):
        scene[name] = SimpleNamespace(
            data=MockContactData(
                name,
                robot,
                torch.tensor([[[[1.0, 0.0, 0.0]]]] * 2) * (1 if name == "fixed_finger_contact" else -1),
            )
        )
    return SimpleNamespace(num_envs=2, device="cpu", step_dt=0.04, scene=scene)


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
        env.scene["cube"].data.root_lin_vel_w.torch[0, 0] = 0.3
        self.assertFalse(success(env, hold_time_s=0.2)[0])
        env.scene["cube"].data.root_lin_vel_w.torch.zero_()
        for _ in range(4):
            self.assertFalse(success(env, hold_time_s=0.2).any())
        self.assertEqual(success(env, hold_time_s=0.2).tolist(), [True, False])
        success.reset(torch.tensor([0]))
        self.assertFalse(success(env).any())

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
        torch.testing.assert_close(gripper.processed_actions, torch.full((2, 1), 0.6))
        gripper.process_actions(torch.tensor([[-100.0], [100.0]]))
        torch.testing.assert_close(gripper.processed_actions, torch.tensor([[0.0], [1.2]]))


if __name__ == "__main__":
    unittest.main()
