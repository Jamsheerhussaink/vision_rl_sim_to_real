"""Batched camera images on the GPU with MJX's Warp ray tracer.

The real robot squashes the D455's 424x240 frame to a size x size policy image
(`vision_rl.camera.to_policy_image`). Two ways to make the same image in sim:

  exact=True   render 424x240, then run the same `to_policy_image`.
  exact=False  (default) render size x size directly through a camera whose
               horizontal and vertical field of view are those of the 424x240
               frame (separate fx / fy). Same geometry, about half the rays;
               differs from the exact path by < 1/255 on average (edge pixels
               are point-sampled instead of area-averaged).

Everything stays on the GPU and inside jit.
"""
from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME, to_policy_image


def squashed_camera_model(mj_model: mujoco.MjModel, cam: int) -> mujoco.MjModel:
    """Copy of the model whose camera renders a square image with the 424x240 FOV."""
    m = copy.copy(mj_model)
    tan_y = np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
    tan_x = tan_y * CAM_WIDTH / CAM_HEIGHT
    m.cam_sensorsize[cam] = [1.0, 1.0]                       # square sensor: no aspect crop
    m.cam_intrinsic[cam] = [0.5 / tan_x, 0.5 / tan_y, 0.0, 0.0]
    return m


class WarpRenderer:
    def __init__(self, mj_model: mujoco.MjModel, mjx_model, num_envs: int, size: int = 128,
                 camera: str = CAMERA_NAME, shadows: bool = True, exact: bool = False):
        from mujoco.mjx._src import bvh, render, render_util
        from mujoco.mjx.warp import io

        self._bvh, self._render, self._util = bvh, render, render_util
        self._mx = mjx_model                          # must be put_model(..., impl="warp")
        self.size, self.exact = size, exact
        self._cam = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self._cam < 0:
            raise ValueError(f"camera {camera!r} not in model")
        ctx_model = mj_model if exact else squashed_camera_model(mj_model, self._cam)
        res = (CAM_WIDTH, CAM_HEIGHT) if exact else (size, size)
        # Geom groups 0-2 only: the red collision shapes (groups 3, 4) stay hidden.
        self._ctx = io.create_render_context(
            mjm=ctx_model, nworld=num_envs, cam_res=res,
            use_textures=True, use_shadows=shadows, render_rgb=True,
            render_depth=False, render_seg=False, enabled_geom_groups=[0, 1, 2])

    def _rgb(self, data) -> jax.Array:
        rc = self._ctx.pytree()
        data = jax.vmap(self._bvh.refit_bvh, in_axes=(None, 0, None))(self._mx, data, rc)
        rgb = jax.vmap(self._render.render, in_axes=(None, 0, None))(self._mx, data, rc)[0]
        rgb = jax.vmap(self._util.get_rgb, in_axes=(None, None, 0))(rc, self._cam, rgb)
        return (jnp.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(jnp.uint8)

    def __call__(self, data) -> jax.Array:
        """uint8 [B, size, size, 3] policy images for a batched mjx.Data."""
        rgb = self._rgb(data)
        return to_policy_image(rgb, xp=jnp, size=self.size) if self.exact else rgb
