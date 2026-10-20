"""Unit tests for Behavior Cloning (BC) dataset, training step, and PPO warm-start loading."""

import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from isaac_vision_rl.utils.demonstrations import Demonstration, DemonstrationDataset, find_eligible_demonstrations
from isaac_vision_rl.utils.rl_agent import ExplorationPPO


class BehaviorCloningTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_demo(self, name, eligible=True, steps=10, schema="so101_rgb_ik_demo_v1"):
        demo_path = self.test_dir / name
        demo = Demonstration(demo_path, chunk_size=steps)
        for i in range(steps):
            pol = np.zeros(29, dtype=np.float32)
            pol[0] = float(i) * 0.01
            img = np.full((3, 128, 128), 0.5, dtype=np.float32)
            act = np.full(6, 0.2, dtype=np.float32)
            demo.add(pol, img, act, reward=1.0, terminated=False, phase="reach")
        demo.finish(success=eligible, reason="test_mock", checkpoint_schema="four_stage_v2_bounded")
        if schema != "so101_rgb_ik_demo_v1":
            meta_path = demo_path / "metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["schema"] = schema
            meta_path.write_text(json.dumps(meta))
        return demo_path

    def test_dataset_validation_and_loading(self):
        # 1. Eligible demonstration loads cleanly
        d1 = self._create_mock_demo("demo_good", eligible=True, steps=8)
        ds = DemonstrationDataset(d1)
        self.assertEqual(len(ds), 8)
        sample = ds[0]
        self.assertEqual(sample["policy"].shape, (29,))
        self.assertEqual(sample["image"].shape, (3, 128, 128))
        self.assertEqual(sample["action"].shape, (6,))

        # 2. Ineligible demonstration is rejected
        d2 = self._create_mock_demo("demo_ineligible", eligible=False, steps=5)
        with self.assertRaisesRegex(ValueError, "not marked eligible_for_cloning"):
            DemonstrationDataset(d2)

        # 3. Schema mismatch is rejected
        d3 = self._create_mock_demo("demo_wrong_schema", eligible=True, steps=5, schema="wrong_schema_v99")
        with self.assertRaisesRegex(ValueError, "schema 'wrong_schema_v99' != expected"):
            DemonstrationDataset(d3)

        # 4. find_eligible_demonstrations discovers only eligible runs
        eligible = find_eligible_demonstrations(self.test_dir)
        self.assertEqual(len(eligible), 2)  # d1 and d3 are marked eligible in metadata
        self.assertIn(d1, eligible)
        self.assertNotIn(d2, eligible)

    def test_bc_training_loss_descent(self):
        torch.set_num_threads(1)
        demo_path = self._create_mock_demo("demo_train", eligible=True, steps=32)
        dataset = DemonstrationDataset(demo_path)

        cfg = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
        config = cfg.to_dict()
        config["multi_gpu"] = None

        mock_obs = TensorDict(
            {"policy": torch.zeros(4, 29), "critic": torch.zeros(4, 51), "image": torch.zeros(4, 3, 128, 128)},
            batch_size=[4],
        )
        mock_env = SimpleNamespace(num_envs=4, num_actions=6)
        alg = ExplorationPPO.construct_algorithm(mock_obs, mock_env, copy.deepcopy(config), "cpu")
        actor = alg.actor

        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        losses = []
        for epoch in range(10):
            actor.train()
            epoch_loss = 0.0
            for batch in loader:
                obs = TensorDict({"policy": batch["policy"], "image": batch["image"]}, batch_size=[batch["policy"].shape[0]])
                actor.update_normalization(obs)
                pred = actor(obs, stochastic_output=False)
                loss = loss_fn(pred, batch["action"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        # Check that loss has strictly decreased after 10 epochs
        self.assertLess(losses[-1], losses[0])

    def test_bc_checkpoint_loading_into_ppo_warmstart(self):
        torch.set_num_threads(1)
        cfg = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
        config = cfg.to_dict()
        config["multi_gpu"] = None

        mock_obs = TensorDict(
            {"policy": torch.zeros(4, 29), "critic": torch.zeros(4, 51), "image": torch.zeros(4, 3, 128, 128)},
            batch_size=[4],
        )
        mock_env = SimpleNamespace(num_envs=4, num_actions=6)

        # Alg 1: Creates initial BC weights
        alg1 = ExplorationPPO.construct_algorithm(mock_obs, mock_env, copy.deepcopy(config), "cpu")
        with torch.no_grad():
            for param in alg1.actor.parameters():
                param.fill_(0.42)
        alg1.exploration_distribution().project_noise()

        bc_checkpoint = {
            "actor_state_dict": copy.deepcopy(alg1.actor.state_dict()),
            "critic_state_dict": copy.deepcopy(alg1.critic.state_dict()),
            "optimizer_state_dict": {},
            "iter": 0,
            "infos": {"source": "behavior_cloning"},
            "is_bc_checkpoint": True,
            "so101_agent_schema": "four_stage_v2_bounded",
            "so101_policy_variant": "rgb",
            "so101_reward_profile": "four-stage",
            "so101_reward_style": "continuous",
            "so101_control_hz": 30,
            "so101_rgb_profile": "1280x720_letterbox128_v1",
            "exploration_entropy_coef": 0.01,
        }

        # Alg 2: Brand new instance representing fresh PPO initialization
        alg2 = ExplorationPPO.construct_algorithm(mock_obs, mock_env, copy.deepcopy(config), "cpu")

        # Record critic weight in alg2 before warm-start load
        head_critic_weight_before = alg2.critic.mlp[-1].weight.clone()

        # Load warm-start BC checkpoint
        with unittest.mock.patch.dict("os.environ", {"SO101_REWARD_PROFILE": "four-stage", "SO101_POLICY": "rgb"}):
            loaded_iter = alg2.load(bc_checkpoint)

        # 1. Warm-start load must return False (iteration is not resumed, stays 0)
        self.assertFalse(loaded_iter)

        # 2. Actor weights must strictly match alg1
        for p1, p2 in zip(alg1.actor.parameters(), alg2.actor.parameters(), strict=True):
            self.assertTrue(torch.equal(p1, p2))

        # 3. Critic weights must NOT be overwritten from checkpoint (they remain fresh)
        self.assertTrue(torch.equal(alg2.critic.mlp[-1].weight, head_critic_weight_before))

        # 4. Entropy coefficient initialized
        self.assertEqual(alg2.entropy_coef, 0.01)

        # 5. Deterministic inference policy forward pass works
        obs = TensorDict({"policy": torch.randn(2, 29), "image": torch.rand(2, 3, 128, 128)}, batch_size=[2])
        out = alg2.actor(obs, stochastic_output=False)
        self.assertEqual(out.shape, (2, 6))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
