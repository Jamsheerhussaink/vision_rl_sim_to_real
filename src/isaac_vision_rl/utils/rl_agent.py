"""SO-101 PPO exploration controls and diagnostics; upstream TD/GAE is retained."""

import math
import os
from typing import cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import CNNModel, MLPModel
from rsl_rl.modules.distribution import GaussianDistribution


def _initialize_grasp_actor(model: MLPModel):
    if os.environ.get("SO101_CURRICULUM") != "grasp":
        return
    from isaac_vision_rl.utils.pipeline import read_physical_report

    pose = read_physical_report()["grasp_action"]
    head = cast(torch.nn.Linear, model.mlp[-1])
    if head.bias is None:
        raise TypeError("Grasp actor requires a linear output bias")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([*pose, 0.0], device=head.bias.device))


class InitialPoseCNN(CNNModel):
    """RGB plus state policy with a verified initial action mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _initialize_grasp_actor(self)


class InitialPoseMLP(MLPModel):
    """State-only comparison using the same initial mean and action mapping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _initialize_grasp_actor(self)


class ExplorationGaussian(GaussianDistribution):
    """Learn noise in log space, with an explicit normalized-action noise floor."""

    def __init__(self, output_dim, init_std=0.4, min_std=0.1, max_std=0.8, std_type="log"):
        if not 0 < min_std <= init_std <= max_std:
            raise ValueError("Require 0 < min_std <= init_std <= max_std")
        super().__init__(output_dim, init_std, (min_std, max_std), std_type="log")

    @torch.no_grad()
    def project_noise(self):
        # Project the parameter too, so it can recover after reaching a bound.
        self.log_std_param.clamp_(*self.log_std_range)


class ExplorationPPO(PPO):
    """PPO with bounded entropy adaptation and rollout diagnostics.

    Rewards, discounting, terminal masks, value loss and TD/GAE remain upstream.
    Entropy is that of the raw Gaussian, before environment action clipping.
    """

    def __init__(
        self,
        *args,
        target_std=0.2,
        entropy_adaptation_rate=0.05,
        entropy_coef_min=0.001,
        entropy_coef_max=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if target_std <= 0 or not 0 < entropy_coef_min <= self.entropy_coef <= entropy_coef_max:
            raise ValueError("Invalid entropy exploration configuration")
        self.target_std = target_std
        self.entropy_adaptation_rate = entropy_adaptation_rate
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_max = entropy_coef_max
        self._diagnostics = {}

    def exploration_distribution(self) -> ExplorationGaussian:
        distribution = cast(MLPModel, self._raw_actor).distribution
        if not isinstance(distribution, ExplorationGaussian):
            raise TypeError("SO-101 PPO requires the configured ExplorationGaussian actor distribution")
        return distribution

    def compute_returns(self, obs):
        # Match upstream terminal masking; timeout rewards were already bootstrapped.
        st = self.storage
        with torch.no_grad():
            hidden = self.critic.get_hidden_state()
            last_values = self.critic(obs).detach()
            self.critic.reset(hidden_state=hidden)
            next_values = torch.cat((st.values[1:], last_values.unsqueeze(0)))
            td = st.rewards + self.gamma * (1 - st.dones.float()) * next_values - st.values
            self._diagnostics = {
                "td_error_abs": td.abs().mean().item(),
                "action_saturation": (st.actions.abs() >= 1).float().mean().item(),
            }
        super().compute_returns(obs)
        with torch.no_grad():
            variance = st.returns.var(unbiased=False)
            explained = 1 - (st.returns - st.values).var(unbiased=False) / variance.clamp_min(1e-8)
            self._diagnostics.update(
                {
                    "value_mean": st.values.mean().item(),
                    "return_mean": st.returns.mean().item(),
                    "explained_variance": explained.item() if variance > 1e-8 else 0.0,
                }
            )

    def update(self):
        metrics = super().update()
        distribution = self.exploration_distribution()
        distribution.project_noise()
        with torch.no_grad():
            std = distribution.log_std_param.exp()
            # Target per-action Gaussian entropy, expressed as a target noise std.
            target = math.log(self.target_std) + 0.5 * math.log(2 * math.pi * math.e)
            actual = std.log().mean().item() + 0.5 * math.log(2 * math.pi * math.e)
            multiplier = math.exp(max(-0.1, min(0.1, self.entropy_adaptation_rate * (target - actual))))
            self.entropy_coef = min(self.entropy_coef_max, max(self.entropy_coef_min, self.entropy_coef * multiplier))
            metrics.update({f"noise_std_{i}": value.item() for i, value in enumerate(std)})
        metrics.update(self._diagnostics)
        metrics["entropy_coefficient"] = self.entropy_coef
        return metrics

    def save(self):
        saved = super().save()
        saved["so101_policy_variant"] = "state" if isinstance(self._raw_actor, InitialPoseMLP) else "vision"
        saved["so101_agent_schema"] = "camera_grasp_assist_v7"
        saved["so101_gripper_controller"] = "camera_capture_latch"
        saved["so101_cube_observation"] = "global_rgbd_estimator"
        saved["exploration_entropy_coef"] = self.entropy_coef
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("so101_agent_schema") != "camera_grasp_assist_v7":
            raise ValueError(
                "This checkpoint predates the camera grasp assist and its observed controller state; "
                "start a fresh run."
            )
        expected = "state" if isinstance(self._raw_actor, InitialPoseMLP) else "vision"
        if loaded_dict.get("so101_policy_variant", "vision") != expected:
            raise ValueError(
                f"Checkpoint policy does not match --policy {expected}; use a matching checkpoint or start fresh."
            )
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.entropy_coef = min(
                self.entropy_coef_max,
                max(self.entropy_coef_min, float(loaded_dict.get("exploration_entropy_coef", self.entropy_coef))),
            )
        self.exploration_distribution().project_noise()
        return result
