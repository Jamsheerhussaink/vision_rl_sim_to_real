"""CPU checks of actual CNN PPO updates, TD/GAE and placement phase logic."""

import copy
import importlib.metadata
import math
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from tensordict import TensorDict
from test_picklift_logic import make_env

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from isaac_vision_rl.tasks.pickplace.mdp.observations import critic_task_state
from isaac_vision_rl.tasks.pickplace.mdp.rewards import PlaceSuccess, cube_in_box
from isaac_vision_rl.utils.rl_agent import ExplorationGaussian, ExplorationPPO


def placement_env():
    env = make_env()
    env.scene["cube"].data.root_quat_w = NS(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2))
    env.scene["cube"].data.root_ang_vel_w = NS(torch=torch.zeros(2, 3))
    env.scene["box"] = NS(
        data=NS(
            root_pos_w=NS(torch=torch.tensor([[0.0, 0.0, 0.401]] * 2)),
            root_quat_w=NS(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)),
        )
    )
    return env


class AgentTests(unittest.TestCase):
    def test_noise_bounds_and_finite_log_probability(self):
        dist = ExplorationGaussian(6)
        with torch.no_grad():
            dist.log_std_param[:] = torch.tensor([-100.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        dist.project_noise()
        dist.update(torch.zeros(4, 6))
        self.assertTrue(torch.all(dist.std >= 0.09999))
        self.assertTrue(torch.all(dist.std <= 0.80001))
        self.assertTrue(torch.isfinite(dist.log_prob(dist.sample())).all())

    def test_actual_cnn_ppo_update_and_checkpoint(self):
        torch.set_num_threads(1)
        cfg = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
        cfg.num_steps_per_env = 4
        cfg.algorithm.num_learning_epochs = 1
        cfg.algorithm.num_mini_batches = 2
        config = cfg.to_dict()
        config["multi_gpu"] = None
        obs = TensorDict(
            {"policy": torch.randn(4, 25), "critic": torch.randn(4, 51), "image": torch.rand(4, 3, 128, 128)},
            batch_size=[4],
        )
        env = NS(num_envs=4, num_actions=6)
        alg = ExplorationPPO.construct_algorithm(obs, env, copy.deepcopy(config), "cpu")
        for _ in range(4):
            actions = alg.act(obs)
            self.assertEqual(actions.shape, (4, 6))
            alg.process_env_step(obs, torch.ones(4), torch.tensor([0, 0, 0, 1]), {})
        alg.compute_returns(obs)
        metrics = alg.update()
        self.assertLess(alg.entropy_coef, 0.01)  # Noise is above the entropy target.
        self.assertTrue(all(math.isfinite(float(v)) for v in metrics.values()))
        for name in ["td_error_abs", "return_mean", "explained_variance", "action_saturation", "noise_std_5"]:
            self.assertIn(name, metrics)
        saved = copy.deepcopy(alg.save())
        other = ExplorationPPO.construct_algorithm(obs, env, copy.deepcopy(config), "cpu")
        other.load(saved)
        self.assertEqual(other.entropy_coef, alg.entropy_coef)
        self.assertTrue(torch.equal(alg.actor(obs), other.actor(obs)))
        saved.pop("so101_agent_schema")
        with self.assertRaisesRegex(ValueError, "fresh run"):
            other.load(saved)
        previous_coefficient = alg.entropy_coef
        with torch.no_grad():
            alg.actor.distribution.log_std_param.fill_(math.log(0.1))
        for _ in range(4):
            alg.act(obs)
            alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
        alg.compute_returns(obs)
        alg.update()
        self.assertGreater(alg.entropy_coef, previous_coefficient)


    def test_four_stage_reward_modes_update_and_round_trip_checkpoints(self):
        from unittest.mock import patch
        for mode in ('bounded','overlap'):
            with patch.dict('os.environ',{'SO101_REWARD_PROFILE':'four-stage',
                                         'SO101_GRASP_REWARD_MODE':mode,'SO101_TRAINING_GOAL':'lift'}):
                self.test_actual_cnn_ppo_update_and_checkpoint()

    def test_td_gae_terminal_and_bootstrap_returns(self):
        alg = ExplorationPPO.__new__(ExplorationPPO)
        alg.gamma, alg.lam = 0.99, 0.95
        alg.normalize_advantage_per_mini_batch = False

        class Critic:
            def get_hidden_state(self):
                return None

            def reset(self, **kwargs):
                pass

            def __call__(self, obs):
                return torch.ones(2, 1)

        alg.critic = Critic()
        alg.storage = NS(
            num_transitions_per_env=2,
            values=torch.tensor([[[0.5], [0.5]], [[0.8], [0.8]]]),
            rewards=torch.tensor([[[1.0], [1.0]], [[2.0], [2.0]]]),
            dones=torch.tensor([[[0], [0]], [[1], [0]]]),
            actions=torch.zeros(2, 2, 6),
            returns=torch.zeros(2, 2, 1),
        )
        alg.compute_returns(None)
        torch.testing.assert_close(alg.storage.returns, torch.tensor([[[2.9206], [3.851695]], [[2.0], [2.99]]]))

    def test_placement_requires_lift_release_withdrawal_and_hold(self):
        env = placement_env()
        success = PlaceSuccess(NS(), env)
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.4300
        self.assertFalse(success(env).any())
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.52
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.53
        for _ in range(math.ceil(0.5 / env.step_dt)):
            success(env)
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.4300
        env.scene["robot"].data.body_pos_w.torch[:, :, 2] = 0.55
        for _ in range(10):
            self.assertFalse(success(env).any())  # still closed
        env.scene["robot"].data.joint_pos.torch[:] = 1.2
        for _ in range(9):
            self.assertFalse(success(env).any())
        self.assertEqual(success(env).tolist(), [True, False])
        success.reset(torch.tensor([0]))
        self.assertFalse(success(env).any())

    def test_cube_below_box_floor_is_not_placed(self):
        env = placement_env()
        env.scene["cube"].data.root_pos_w.torch[:, :] = torch.tensor([0.0, 0.0, 0.425])
        self.assertFalse(cube_in_box(env).any())
        env.scene["cube"].data.root_pos_w.torch[:, 2] = 0.430
        self.assertTrue(cube_in_box(env).all())

    def test_rotated_cube_cannot_overlap_box_wall(self):
        env = placement_env()
        env.scene["cube"].data.root_pos_w.torch[:, :] = torch.tensor([0.068, 0.0, 0.4300])
        self.assertTrue(cube_in_box(env).all())
        env.scene["cube"].data.root_quat_w.torch[:, :] = torch.tensor(
            [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]
        )
        self.assertFalse(cube_in_box(env).any())

    def test_critic_state_is_invariant_to_environment_grid_offset(self):
        env = placement_env()

        class Scene(dict):
            pass

        env.scene = Scene(env.scene)
        env.scene.env_origins = torch.zeros(2, 3)
        env.episode_length_buf = torch.zeros(2)
        env.max_episode_length = 150
        before = critic_task_state(env)
        offset = torch.tensor([10.0, -20.0, 0.0])
        env.scene.env_origins += offset
        for key in ["cube", "box"]:
            env.scene[key].data.root_pos_w.torch += offset
        env.scene["robot"].data.body_pos_w.torch += offset
        torch.testing.assert_close(critic_task_state(env), before, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
