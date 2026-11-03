"""PPO: GAE + clipped policy/value loss + minibatch updates. Pure JAX, jit-able.

Time limits: the observation carries "time left", so the end of an episode by
time limit is a real terminal state for the critic, and GAE can cut at every
`done` without biasing the value.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 256
    rollout_steps: int = 32           # per env per iteration
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    lr_final_frac: float = 0.3        # linear anneal to 30% (not 0: late discoveries still learn)
    weight_decay: float = 1e-4
    total_steps: int = 50_000_000
    seed: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def num_iterations(self):
        return max(1, self.total_steps // self.batch_size)


def make_optimizer(cfg: PPOConfig):
    n = cfg.num_iterations * cfg.update_epochs * cfg.num_minibatches
    lr = optax.linear_schedule(cfg.learning_rate, cfg.learning_rate * cfg.lr_final_frac, n)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                       optax.adamw(lr, eps=1e-5, weight_decay=cfg.weight_decay))


def gae(rewards, values, dones, last_value, gamma, lam):
    """[T, B] arrays; dones[t] = 1 if the episode ended after step t."""
    def body(carry, x):
        adv, next_v = carry
        r, v, d = x
        nd = 1.0 - d
        delta = r + gamma * next_v * nd - v
        adv = delta + gamma * lam * nd * adv
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value),
                          (rewards, values, dones), reverse=True)
    return adv, adv + values


def _loss(params, apply_fn, mb, cfg: PPOConfig):
    pi, v = apply_fn(params, mb["obs"])
    logp = pi.log_prob(mb["actions"])
    ratio = jnp.exp(logp - mb["log_probs"])
    adv = mb["advantages"]
    pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)).mean()
    v_clip = mb["values"] + jnp.clip(v - mb["values"], -cfg.clip_eps, cfg.clip_eps)
    vf = 0.5 * jnp.maximum(jnp.square(v - mb["returns"]), jnp.square(v_clip - mb["returns"])).mean()
    ent = pi.entropy().mean()
    loss = pg + cfg.value_coef * vf - cfg.entropy_coef * ent
    return loss, {"pg_loss": pg, "vf_loss": vf, "entropy": ent,
                  "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
                  "clip_frac": (jnp.abs(ratio - 1) > cfg.clip_eps).mean(),
                  "std": pi.std.mean()}


def ppo_update(state: TrainState, batch: dict, key, cfg: PPOConfig):
    """`update_epochs` passes of minibatch SGD over a flat batch (leading dim N)."""
    n = batch["advantages"].shape[0]
    mb = n // cfg.num_minibatches
    adv = jnp.nan_to_num(batch["advantages"])
    batch = {**batch, "advantages": (adv - adv.mean()) / (adv.std() + 1e-8),
             "returns": jnp.nan_to_num(batch["returns"])}
    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    def mb_step(state, idx):
        part = jax.tree.map(lambda x: x[idx], batch)
        (_, m), g = grad_fn(state.params, state.apply_fn, part, cfg)
        g = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), g)
        return state.apply_gradients(grads=g), m

    def epoch(state, k):
        perm = jax.random.permutation(k, n)[: mb * cfg.num_minibatches]
        return jax.lax.scan(mb_step, state, perm.reshape(cfg.num_minibatches, mb))

    state, m = jax.lax.scan(epoch, state, jax.random.split(key, cfg.update_epochs))
    return state, jax.tree.map(jnp.mean, m)
