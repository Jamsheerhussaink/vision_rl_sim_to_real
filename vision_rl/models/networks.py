"""CNN image encoder + actor-critic (Flax).

obs = {"pixels": uint8 [..., S, S, 3k], "proprio": float [..., P]}
  pixels  -> conv stack -> Dense(features) -> LayerNorm -> ReLU
  concat proprio -> MLP (LayerNorm + tanh) -> policy mean (tanh) and value

The policy is a diagonal Gaussian with a learned, state-independent log-std,
clamped to [log_std_min, log_std_max] so exploration can neither collapse nor
blow up.
"""
from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def conv_layout(image_size: int):
    """(channels, kernels, strides) sized so the last conv map is ~6-12 px wide."""
    if image_size >= 160:
        return (32, 64, 64, 64), (8, 4, 3, 3), (4, 2, 2, 1)
    return (32, 64, 64), (8, 4, 3), (4, 2, 1)


class DiagGaussian:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def log_prob(self, x):
        return jnp.sum(-0.5 * (jnp.square((x - self.mean) / self.std) + 2 * jnp.log(self.std) + _LOG_2PI), -1)

    def entropy(self):
        return jnp.sum(0.5 * (_LOG_2PI + 1.0) + jnp.log(self.std), -1)


def _ortho(scale):
    return nn.initializers.orthogonal(scale)


class Encoder(nn.Module):
    channels: Sequence[int]
    kernels: Sequence[int]
    strides: Sequence[int]
    features: int = 256

    @nn.compact
    def __call__(self, pixels):
        *batch, h, w, c = pixels.shape
        x = pixels.reshape((-1, h, w, c)).astype(jnp.float32) / 255.0
        for ch, k, s in zip(self.channels, self.kernels, self.strides):
            x = nn.relu(nn.Conv(ch, (k, k), (s, s), padding="VALID", kernel_init=_ortho(jnp.sqrt(2.0)))(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.LayerNorm()(nn.Dense(self.features, kernel_init=_ortho(jnp.sqrt(2.0)))(x)))
        return x.reshape((*batch, self.features))


class ActorCritic(nn.Module):
    action_dim: int
    image_size: int = 128
    features: int = 256
    hidden: Sequence[int] = (256, 256)
    init_log_std: float = -0.5
    log_std_min: float = -3.0          # std >= 0.05
    log_std_max: float = 0.3           # std <= 1.35

    @nn.compact
    def __call__(self, obs):
        ch, ks, st = conv_layout(self.image_size)
        h = Encoder(ch, ks, st, self.features)(obs["pixels"])
        h = jnp.concatenate([h, obs["proprio"]], axis=-1)
        for n in self.hidden:
            h = nn.tanh(nn.LayerNorm()(nn.Dense(n, kernel_init=_ortho(jnp.sqrt(2.0)))(h)))
        mean = nn.tanh(nn.Dense(self.action_dim, kernel_init=_ortho(0.01))(h))
        log_std = self.param("log_std", lambda _, s: jnp.full(s, self.init_log_std), (self.action_dim,))
        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max)) * jnp.ones_like(mean)
        value = nn.Dense(1, kernel_init=_ortho(1.0))(h)[..., 0]
        return DiagGaussian(mean, std), value
