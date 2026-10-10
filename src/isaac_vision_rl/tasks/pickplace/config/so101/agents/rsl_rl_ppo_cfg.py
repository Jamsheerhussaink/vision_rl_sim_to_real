# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any, cast

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ExplorationDistributionCfg(RslRlCNNModelCfg.GaussianDistributionCfg):
    class_name = "isaac_vision_rl.utils.rl_agent:ExplorationGaussian"
    init_std = 0.4
    std_type = "log"
    min_std = 0.1
    max_std = 0.8


@configclass
class ExplorationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "isaac_vision_rl.utils.rl_agent:ExplorationPPO"
    target_std = 0.2
    entropy_adaptation_rate = 0.05
    entropy_coef_min = 0.001
    entropy_coef_max = 0.05


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    clip_actions = 1.0
    experiment_name = "so101_pickplace_vision"

    # Actor sees camera-derived cube geometry, proprioception and optional RGB.
    # Simulator object state is restricted to the training critic and rewards.
    obs_groups = {
        "actor": ["policy", "image"],
        "critic": ["critic"],
    }

    actor: RslRlCNNModelCfg | RslRlMLPModelCfg = RslRlCNNModelCfg(
        class_name="isaac_vision_rl.utils.rl_agent:InitialPoseCNN",
        obs_normalization=True,
        hidden_dims=[256, 128, 64],
        activation="elu",
        distribution_cfg=ExplorationDistributionCfg(),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = ExplorationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=cast(Any, None),  # Fixed learning rate; PPO clipping constrains policy updates.
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        variant = os.environ.get("SO101_POLICY", "vision")
        if variant not in ("vision", "state"):
            raise ValueError(f"Unknown SO101_POLICY: {variant}")
        if variant == "state" and self.actor.class_name != "isaac_vision_rl.utils.rl_agent:InitialPoseMLP":
            self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
            self.actor = RslRlMLPModelCfg(
                class_name="isaac_vision_rl.utils.rl_agent:InitialPoseMLP",
                obs_normalization=True,
                hidden_dims=[256, 128, 64],
                activation="elu",
                distribution_cfg=ExplorationDistributionCfg(),
            )
        if os.environ.get("SO101_CURRICULUM") == "grasp":
            # Local practice around the verified grasp pose; large whole-arm
            # perturbations otherwise disrupt closure before a stable lift.
            if self.actor.distribution_cfg is not None:
                cast(ExplorationDistributionCfg, self.actor.distribution_cfg).init_std = 0.15
            cast(ExplorationAlgorithmCfg, self.algorithm).target_std = 0.15
