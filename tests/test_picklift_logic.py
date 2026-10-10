"""CPU regressions for approach reward and sustained lift success."""

import unittest
from types import SimpleNamespace

import torch

from isaac_vision_rl.tasks.pickplace.mdp.rewards import (
    CubeTableDisplacementPenalty,
    GraspCubeProgress,
    LiftCubeProgress,
    LiftSuccess,
    ReachCubeProgress,
    grasp_cube,
    lift_cube_upward,
    object_ee_distance,
    premature_gripper_closure_penalty,
)


class JointIds:
    def __init__(self, values):
        self.torch = torch.tensor(values)

    def __len__(self):
        return len(self.torch)


def make_env():
    cube = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.425], [0.0, 0.0, 0.425]])),
            root_lin_vel_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.625]], [[0.0, 0.0, 0.625]]])),
            joint_pos=SimpleNamespace(torch=torch.tensor([[0.3], [1.2]])),
        ),
        find_bodies=lambda _: ([0], ["gripper_frame_link"]),
        find_joints=lambda _: ([0], ["gripper"]),
    )
    scene = {"robot": robot, "cube": cube}
    for name in ("fixed_finger_contact", "moving_finger_contact"):
        scene[name] = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=torch.ones(2, 1, 1, 3)))
    return SimpleNamespace(num_envs=2, device="cpu", step_dt=0.04, scene=scene)


class PickLiftLogicTests(unittest.TestCase):
    def test_one_finger_touch_and_stationary_grasp_cannot_farm_reward(self):
        env = make_env()
        env.scene["robot"].data.body_pos_w.torch[:, 0] = torch.tensor([0.0, 0.0, 0.425])
        env.scene["robot"].data.joint_pos.torch[:] = 0.3
        reward = GraspCubeProgress(SimpleNamespace(), env)
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.fill_(1)
        self.assertTrue((reward(env) > 0).all())
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.zero_()
        reward(env)
        env.scene["moving_finger_contact"].data.normal_force_matrix_w.fill_(1)
        self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
        reward.reset(torch.tensor([0]))
        self.assertEqual((reward(env) > 0).tolist(), [True, False])

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
        hold = torch.tensor([15, 0])
        env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(closure_hold_steps=hold))
        arm.process_actions(torch.ones(2, 5))
        torch.testing.assert_close(arm.processed_actions[0], torch.full((5,), 0.25))
        arm.process_actions(-torch.ones(2, 5))
        torch.testing.assert_close(arm.processed_actions[0], torch.full((5,), 0.25))
        hold.zero_()
        arm.process_actions(-torch.ones(2, 5))
        self.assertTrue((arm.processed_actions[0] < 0.25).all())
        cfg.gripper_action.enabled = False  # Verify the unassisted mapping separately.
        gripper = string_to_callable(cfg.gripper_action.class_type)(cfg.gripper_action, env)
        gripper.process_actions(torch.zeros(2, 1))
        torch.testing.assert_close(gripper.processed_actions, torch.full((2, 1), 1.2))
        gripper.process_actions(torch.tensor([[-100.0], [100.0]]))
        torch.testing.assert_close(gripper.processed_actions, torch.tensor([[0.0], [1.2]]))


if __name__ == "__main__":
    unittest.main()
