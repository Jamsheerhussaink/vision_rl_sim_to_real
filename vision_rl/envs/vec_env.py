"""All worlds at once on the GPU: vmapped physics + camera images + frame stack.

Observation dict:
    "pixels":  uint8 [B, S, S, 3 * frame_stack]   (S = image_size)   (oldest frame first)
    "proprio": float [B, PickEnv.proprio_size]

Finished worlds are reset inside `step`, so rollouts never stop. Optional image
domain randomisation (`image_dr`) jitters brightness, contrast, colour and adds
pixel noise per world and per episode, on the GPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from vision_rl.envs.pick_env import PickEnv, PickState
from vision_rl.rendering import WarpRenderer


@struct.dataclass
class VecState:
    env: PickState                # batched [B, ...]
    frames: jax.Array             # uint8 [B, S, S, 3k]
    dr: jax.Array                 # [B, 7] per-world image jitter (gain rgb, bias rgb, noise)
    rng: jax.Array
    # Stats of episodes that ended on the LAST step (valid where env.done == 1).
    last_return: jax.Array
    last_success: jax.Array
    last_held: jax.Array
    last_len: jax.Array

    @property
    def obs(self):
        return {"pixels": self.frames, "proprio": self.env.obs}


class VecEnv:
    def __init__(self, env: PickEnv, num_envs: int, frame_stack: int = 3, image_size: int = 128,
                 shadows: bool = True, exact_render: bool = False, image_dr: bool = False):
        self.env, self.num_envs, self.frame_stack, self.image_dr = env, num_envs, frame_stack, image_dr
        self.image_size = image_size
        self.renderer = WarpRenderer(env.mj_model, env.mjx_model, num_envs, size=image_size,
                                     shadows=shadows, exact=exact_render)
        self._reset = jax.vmap(env.reset)
        self._step = jax.vmap(env.step)

    # ------------------------------------------------------------------ helpers
    @property
    def action_size(self):
        return self.env.action_size

    @property
    def pixels_shape(self):
        return (self.image_size, self.image_size, 3 * self.frame_stack)

    def sample_obs(self):
        return {"pixels": jnp.zeros((1, *self.pixels_shape), jnp.uint8),
                "proprio": jnp.zeros((1, self.env.proprio_size), jnp.float32)}

    def _sample_dr(self, key):
        if not self.image_dr:
            return jnp.zeros((self.num_envs, 7))
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n = self.num_envs
        bright = jax.random.uniform(k1, (n, 1), minval=-0.2, maxval=0.2)   # overall gain
        tint = jax.random.uniform(k2, (n, 3), minval=-0.06, maxval=0.06)   # per-channel gain
        bias = jax.random.uniform(k3, (n, 3), minval=-12.0, maxval=12.0)   # per-channel offset
        noise = jax.random.uniform(k4, (n, 1), minval=0.0, maxval=4.0)     # pixel noise std
        return jnp.concatenate([bright + tint, bias, noise], axis=-1)

    def _images(self, data, dr, key):
        img = self.renderer(data)                                          # uint8 [B,S,S,3]
        if not self.image_dr:
            return img
        x = img.astype(jnp.float32)
        gain = (1.0 + dr[:, None, None, 0:3])
        x = x * gain + dr[:, None, None, 3:6]
        x = x + dr[:, None, None, 6:7] * jax.random.normal(key, x.shape)
        return jnp.clip(jnp.round(x), 0, 255).astype(jnp.uint8)

    @staticmethod
    def _select(done, a, b):
        """Per world: a where done else b. Leaves not shaped [B, ...] (Warp's global
        solver buffers) are kept: step recomputes them."""
        if b.ndim == 0 or b.shape[0] != done.shape[0]:
            return b
        return jnp.where(done.reshape((-1,) + (1,) * (b.ndim - 1)) > 0, a, b)

    # ------------------------------------------------------------------ API
    def reset(self, key) -> VecState:
        key, k_env, k_dr, k_img = jax.random.split(key, 4)
        env = self._reset(jax.random.split(k_env, self.num_envs))
        dr = self._sample_dr(k_dr)
        img = self._images(env.data, dr, k_img)
        z = jnp.zeros(self.num_envs)
        return VecState(env=env, frames=jnp.tile(img, (1, 1, 1, self.frame_stack)), dr=dr, rng=key,
                        last_return=z, last_success=z, last_held=z, last_len=z)

    def step(self, vs: VecState, action) -> VecState:
        stepped = self._step(vs.env, action)
        done = stepped.done
        rng, k_env, k_dr, k_img = jax.random.split(vs.rng, 4)
        fresh = self._reset(jax.random.split(k_env, self.num_envs))
        # Finished worlds continue from a fresh episode; keep the finished step's
        # reward/done/metrics (for PPO and logging) and its episode stats.
        env = jax.tree.map(lambda a, b: self._select(done, a, b), fresh, stepped)
        env = env.replace(reward=stepped.reward, done=done, metrics=stepped.metrics)
        dr = jnp.where(done[:, None] > 0, self._sample_dr(k_dr), vs.dr)
        img = self._images(env.data, dr, k_img)
        frames = jnp.where(done.reshape(-1, 1, 1, 1) > 0,
                           jnp.tile(img, (1, 1, 1, self.frame_stack)),
                           jnp.concatenate([vs.frames[..., 3:], img], axis=-1))
        return VecState(env=env, frames=frames, dr=dr, rng=rng,
                        last_return=stepped.ep_return, last_success=stepped.ep_success,
                        last_held=stepped.ep_held, last_len=stepped.step.astype(jnp.float32))
