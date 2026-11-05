"""Pick -> carry -> released, settled cube inside a real collision box."""
from dataclasses import dataclass, fields

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from vision_rl.envs.pick_env import PickEnv, PickState


@dataclass(frozen=True)
class PlaceConfig:
    box_range: float = .015
    light_range: float = .12
    settle_seconds: float = .5
    carry_scale: float = 2.
    carry_progress: float = 3.  # Madhan's place_progress_scale
    carry_radius: float = .15  # Madhan's place_radius
    lower_scale: float = 2.
    place_bonus: float = 20.
    finger_impedance_low: float = .99
    finger_impedance_high: float = .999


def placement_geometry(cube, cube_rotation, box, box_rotation, velocity, grasp, jaw_angle):
    """Use the full rotated cube footprint, not just its centre."""
    local = (cube - box) @ box_rotation
    extent = .025 * jnp.sum(jnp.abs(box_rotation.T @ cube_rotation), axis=1)
    inside = jnp.all(jnp.abs(local[:2]) + extent[:2] < jnp.array([.096, .074]) - .002)
    bottom = local[2] - extent[2]
    resting = (jnp.abs(bottom - .004) < .006) & (jnp.linalg.norm(velocity) < .05)
    withdrawn = (jnp.linalg.norm(grasp - cube) > .065) & (jaw_angle > .8)
    return inside, resting, withdrawn, local, extent


@struct.dataclass
class PlaceState(PickState):
    lift_seen: jax.Array
    previous_box_distance: jax.Array
    over_seen: jax.Array
    settled_steps: jax.Array


class PickPlaceEnv(PickEnv):
    def __init__(self, model, cfg, place_cfg=PlaceConfig(), **kwargs):
        self.place_cfg = place_cfg
        # New placement task only: use the previously measured firmer contact
        # variant. Do not alter robot.xml or the original pick scene/force limit.
        finger_bodies = {model.body("gripper").id, model.body("moving_jaw_so101_v1").id}
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] in finger_bodies and model.geom_contype[geom] and model.geom_priority[geom] == 1:
                model.geom_solimp[geom, :2] = [place_cfg.finger_impedance_low, place_cfg.finger_impedance_high]
        super().__init__(model, cfg, **kwargs)
        self.box_body = model.body("target_box").id
        self.box_mocap = int(model.body_mocapid[self.box_body])
        self.light_body = model.body("domain_light").id
        self.light_mocap = int(model.body_mocapid[self.light_body])
        self.box_origin = jnp.asarray(model.body_pos[self.box_body].copy())
        self.box_quat = jnp.asarray(model.body_quat[self.box_body].copy())
        self.light_origin = jnp.asarray(model.body_pos[self.light_body].copy())
        self.cube_dof = int(model.jnt_dofadr[model.body("cube").jntadr[0]])
        self.avoid_geoms = jnp.asarray([model.geom(f"fixed_jaw_box{i}").id for i in (4, 5, 6, 7)])
        self.avoid_radii = jnp.asarray(model.geom_rbound[np.asarray(self.avoid_geoms)] + .003)
        self.required_settle_steps = int(np.ceil(place_cfg.settle_seconds / cfg.ctrl_dt))

    def _place_metrics(self):
        return {k: jnp.float32(0) for k in ("place_success", "pick_success", "box_distance",
                "inside_box", "settled", "withdrawn", "carry_r", "lower_r", "place_r", "drop_pen")}

    def reset(self, key):
        base = super().reset(key)
        k_box, k_light, k_cube = jax.random.split(jax.random.fold_in(key, 710), 3)
        box = self.box_origin.at[:2].add(jax.random.uniform(k_box, (2,),
                    minval=-self.place_cfg.box_range, maxval=self.place_cfg.box_range))
        light = self.light_origin + jax.random.uniform(k_light, (3,),
                    minval=-self.place_cfg.light_range, maxval=self.place_cfg.light_range)
        data = base.data.replace(mocap_pos=base.data.mocap_pos.at[self.box_mocap].set(box)
                                .at[self.light_mocap].set(light),
                                mocap_quat=base.data.mocap_quat.at[self.box_mocap].set(self.box_quat))
        # Reject starts intersecting the initial fixed finger's lower collision
        # pieces. Candidate sampling stays within the original +/-3 cm area.
        candidates = jnp.asarray(self.cube_xy0) + jax.random.uniform(k_cube, (64, 2),
                    minval=-self.cfg.cube_range, maxval=self.cfg.cube_range)
        centres = jnp.concatenate([candidates, jnp.full((64, 1), self.cube_z0)], axis=1)
        rotation = data.xmat[self.cube_body].reshape(3, 3)
        relative = (data.geom_xpos[self.avoid_geoms][None] - centres[:, None]) @ rotation
        distances = jnp.linalg.norm(jnp.maximum(jnp.abs(relative) - .025, 0), axis=-1)
        clear = jnp.all(distances > self.avoid_radii, axis=1)
        box_rotation = data.xmat[self.box_body].reshape(3, 3)
        cube_extent_xy = .025 * jnp.sum(jnp.abs(box_rotation.T @ rotation), axis=1)[:2]
        relative_box = (centres - box) @ box_rotation
        clear = clear & jnp.any(jnp.abs(relative_box[:, :2]) >
                       jnp.array([.10, .08]) + cube_extent_xy + .004, axis=1)
        fallback = jnp.asarray(self.cube_xy0) + jnp.array([self.cfg.cube_range, -self.cfg.cube_range])
        xy = jnp.where(jnp.any(clear), candidates[jnp.argmax(clear)], fallback)
        data = data.replace(qpos=data.qpos.at[self.cube_q:self.cube_q+2].set(xy))
        data = mjx.forward(self.mjx_model, data)
        # Reinitialize pick reward memory for the final randomized reset.
        from vision_rl.rewards.pick import init_pick_state
        x = self._inputs(data, jnp.zeros(6), base.cube_z0)
        base = base.replace(data=data, obs=self._obs(data, base.step), rstate=init_pick_state(x))
        values = {f.name: getattr(base, f.name) for f in fields(PickState)}
        values["metrics"] = {**base.metrics, **self._place_metrics()}
        distance = jnp.linalg.norm(self._cube(data)[:2] - box[:2])
        return PlaceState(**values, lift_seen=jnp.float32(0), previous_box_distance=distance,
                          over_seen=jnp.float32(0), settled_steps=jnp.int32(0))

    def step(self, state, action):
        nxt = super().step(state, action)
        data, pc = nxt.data, self.place_cfg
        cube = self._cube(data)
        box = data.xpos[self.box_body]
        distance = jnp.linalg.norm(cube[:2] - box[:2])
        held = nxt.metrics["held"]
        lifted = nxt.metrics["cube_lift"] >= self.cfg.reward.pick_height
        lift_seen = jnp.maximum(state.lift_seen, held * lifted)
        inside, resting, withdrawn, local, extent = placement_geometry(
            cube, data.xmat[self.cube_body].reshape(3, 3), box,
            data.xmat[self.box_body].reshape(3, 3), data.qvel[self.cube_dof:self.cube_dof+3],
            self._grasp(data), data.qpos[self.jaw_q])
        over = inside.astype(jnp.float32) * lift_seen * held
        over_seen = jnp.maximum(state.over_seen, over)
        # Do not continue paying the old stationary-lift reward after pickup.
        recovery = (held < .5) & ~inside
        pick_gate = jnp.maximum(1 - state.lift_seen, recovery.astype(jnp.float32))
        safety_keys = ("vis_pen", "rate_pen", "limit_pen", "low_pen", "ctrl_cost")
        safety = sum(nxt.metrics[k] for k in safety_keys)
        pickup = (nxt.reward - safety) * pick_gate
        # Madhan's normalized approach-to-target shaping, shifted down by its
        # maximum so hovering above the box cannot farm a positive carry reward.
        carry = lift_seen * (-pc.carry_scale * jnp.clip(distance / pc.carry_radius, 0., 1.) * held + pc.carry_progress * held *
                            (state.previous_box_distance - distance))
        lower = -pc.lower_scale * jnp.maximum(local[2] - extent[2] - .004, 0) * over
        dropped = (lift_seen > .5) & (held < .5) & ~inside
        penalty = dropped.astype(jnp.float32) * .5
        valid = inside & resting & withdrawn & (lift_seen > .5) & (over_seen > .5)
        settled = jnp.where(valid, state.settled_steps + 1, 0)
        success = (settled >= self.required_settle_steps).astype(jnp.float32)
        place_r = pc.place_bonus * success
        reward = pickup + safety + carry + lower + place_r - penalty
        reward = jnp.where(nxt.metrics["unstable"] > .5, 0., jnp.nan_to_num(reward))
        pick_terms = {k: (v if k in safety_keys else v * pick_gate)
                      for k, v in nxt.metrics.items() if k.endswith(("_r", "_pr", "_pen", "_cost"))}
        metrics = {**nxt.metrics, **pick_terms, **self._place_metrics(), "success": success, "place_success": success,
                   "pick_success": nxt.metrics["success"], "box_distance": distance,
                   "inside_box": inside.astype(jnp.float32), "settled": resting.astype(jnp.float32),
                   "withdrawn": withdrawn.astype(jnp.float32), "carry_r": carry,
                   "lower_r": lower, "place_r": place_r, "drop_pen": -penalty}
        return nxt.replace(reward=reward, metrics=metrics, done=jnp.maximum(nxt.done, success),
                 ep_success=jnp.maximum(state.ep_success, success), ep_return=state.ep_return + reward,
                 lift_seen=lift_seen, previous_box_distance=distance, over_seen=over_seen,
                 settled_steps=settled)
