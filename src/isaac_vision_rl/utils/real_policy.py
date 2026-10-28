"""Read-only checkpoint loading and simulator-free SO-101 actor/controller state."""
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from isaac_vision_rl.utils.calibration import JOINT_ORDER
from isaac_vision_rl.utils.home_pose import home_joint_positions
from isaac_vision_rl.utils.rgb_image import actor_rgb

RGB_PROFILES = {
    "424x240_letterbox128_v1": (424, 240),
    "1280x720_letterbox128_v1": (1280, 720),
}
REAL_POLICY_AGENT_SCHEMAS = frozenset({
    "rgb_policy_gripper_v1",
    "four_stage_v1",
    "four_stage_v2_bounded",
})


def validate_real_checkpoint_metadata(saved: dict) -> None:
    """Allow RGB checkpoint labels whose actor can be checked against this adapter."""
    schema = saved.get("so101_agent_schema")
    if schema not in REAL_POLICY_AGENT_SCHEMAS or saved.get("so101_policy_variant") != "rgb":
        raise ValueError("Unsupported real-policy checkpoint schema or policy variant")


def validate_actor_state_dict(expected: dict, saved: dict) -> None:
    """Reject actor weights that do not match the deployed RGB+proprio actor."""
    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(set(saved) - set(expected))
    mismatched = [
        (name, tuple(saved[name].shape), tuple(expected[name].shape))
        for name in set(expected) & set(saved)
        if tuple(saved[name].shape) != tuple(expected[name].shape)
    ]
    if missing or unexpected or mismatched:
        detail = []
        if missing:
            detail.append(f"missing keys: {missing[:3]}")
        if unexpected:
            detail.append(f"unexpected keys: {unexpected[:3]}")
        if mismatched:
            name, checkpoint_shape, runtime_shape = mismatched[0]
            detail.append(f"{name} checkpoint shape {checkpoint_shape}, runtime shape {runtime_shape}")
        raise ValueError(
            "Checkpoint actor does not match the real RGB-and-joint inputs (" + "; ".join(detail) + "). "
            "Train without a simulator-only actor stage input before sim-to-real use."
        )


def camera_profile_dimensions(profile: str) -> tuple[int, int]:
    """Return the native RGB dimensions required by a saved policy profile."""
    try:
        return RGB_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"Unsupported RGB profile: {profile}") from error


def deployed_policy_home() -> np.ndarray:
    """Return the joint reference used by configure_curriculum() at runtime.

    Isaac Lab writes ``params/env.yaml`` before the curriculum hook mutates
    the arm pose.  That file therefore contains the base asset pose, not the
    calibrated physical arm reference.  The gripper starts open in simulation;
    the five arm joints use the measured photo-home calibration.
    """
    values = home_joint_positions()
    values["gripper"] = 1.2
    return np.asarray([values[joint] for joint in JOINT_ORDER], dtype=float)


class RealPolicy:
    def __init__(self, checkpoint, device="cpu"):
        from pxr import Usd, UsdPhysics
        from rsl_rl.models import CNNModel
        checkpoint = Path(checkpoint).resolve()
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        self.rgb_profile = saved.get("so101_rgb_profile")
        self.camera_width, self.camera_height = camera_profile_dimensions(self.rgb_profile)
        validate_real_checkpoint_metadata(saved)
        self.hz = int(saved["so101_control_hz"])
        if self.hz not in (30, 60):
            raise ValueError("This real-camera adapter currently requires a 30 Hz or 60 Hz checkpoint")
        # BaseLoader does not construct arbitrary Python objects from saved YAML.
        env = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
        agent = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
        self.home = deployed_policy_home()
        self.duration = float(env["episode_length_s"])
        self.device = device
        arm = env["actions"]["arm_action"]
        if arm["rescale_to_limits"] != "true" or float(arm["scale"]) != 1:
            raise ValueError("Unsupported arm action mapping")
        self.alpha, self.speed = float(arm["alpha"]), float(arm["max_target_speed"])
        self.motion_profile = saved.get("so101_motion_profile", "legacy_velocity_ema")
        self.acceleration = float(arm["max_target_acceleration"])
        self.wrist_roll_speed = float(arm["wrist_roll_speed"])
        if self.motion_profile == "four_stage_smooth_v1":
            if self.acceleration <= 0 or self.wrist_roll_speed <= 0:
                raise ValueError("Incomplete four-stage smooth controller settings")
        elif self.acceleration != 0 or self.wrist_roll_speed != 0:
            raise ValueError("Unsupported saved arm motion controller")
        self.speed_limits = np.full(5, self.speed, dtype=float)
        self.speed_limits[4] = self.wrist_roll_speed if self.wrist_roll_speed > 0 else self.speed
        self.arm_velocity = np.zeros(5, dtype=float)
        jaw = env["actions"]["gripper_action"]
        self.grip_scale, self.grip_offset = float(jaw["scale"]), float(jaw["offset"])
        self.grip_speed = float(jaw.get("max_target_speed", 0))
        self.grip_min_close = float(jaw.get("min_close_target", 0))
        self.grip_acceleration = float(jaw.get("max_target_acceleration", 0))
        self.grip_velocity = 0.0
        stage = Usd.Stage.Open(env["scene"]["robot"]["spawn"]["usd_path"])
        limits = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in JOINT_ORDER:
                joint = UsdPhysics.RevoluteJoint(prim)
                limits[prim.GetName()] = np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
        factor = float(env["scene"]["robot"].get("soft_joint_pos_limit_factor", 1))
        raw_limits = np.array([limits[j] for j in JOINT_ORDER[:5]])
        centre = raw_limits.mean(axis=1)
        half = (raw_limits[:, 1] - raw_limits[:, 0]) * factor / 2
        self.lower, self.upper = centre - half, centre + half
        obs = TensorDict({"policy": torch.zeros(1, 25, device=device),
                          "image": torch.zeros(1, 3, 128, 128, device=device)}, batch_size=[1])
        cfg = agent["actor"]
        cnn = cfg["cnn_cfg"]
        cnn_cfg = {k: [int(x) for x in cnn[k]] for k in ("output_channels", "kernel_size", "stride")}
        cnn_cfg["activation"] = cnn["activation"]
        distribution = {"class_name": "isaac_vision_rl.utils.rl_agent:ExplorationGaussian",
                        "init_std": .4, "std_type": "log", "min_std": .1, "max_std": .8}
        self.actor = CNNModel(obs, {"actor": ["policy", "image"]}, "actor", 6,
                              hidden_dims=[int(v) for v in cfg["hidden_dims"]], activation=cfg["activation"],
                              obs_normalization=True, distribution_cfg=distribution, cnn_cfg=cnn_cfg).to(device)
        actor_state = saved["actor_state_dict"]
        validate_actor_state_dict(self.actor.state_dict(), actor_state)
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()
        self.reset(self.home)

    def reset(self, measured):
        self.last_action = np.zeros(6)
        self.filtered = np.array([*measured[:5], 1.2], dtype=float)
        self.arm_velocity[:] = 0
        self.grip_velocity = 0.0
        self.steps = 0

    def set_executed_target(self, target):
        """Store the calibration-limited target sent to the motor worker.

        The next actor observation includes this controller memory.  Without
        this update it could describe an unreachable policy proposal while the
        hardware was commanded to a different, clipped pose.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite executed joint targets")
        self.filtered = target.copy()

    def observation(self, rgb, q, velocity):
        state = np.concatenate([np.asarray(q) - self.home, velocity, self.last_action,
                                self.filtered, [max(0, 1 - self.steps / (self.duration * self.hz))]])
        if state.shape != (25,) or not np.isfinite(state).all():
            raise ValueError("Invalid real-robot observation")
        return TensorDict({"policy": torch.tensor(state, dtype=torch.float32, device=self.device)[None],
                           "image": actor_rgb(torch.as_tensor(rgb.copy(), device=self.device)[None])}, batch_size=[1])

    def step(self, rgb, q, velocity):
        obs = self.observation(rgb, q, velocity)
        with torch.inference_mode():
            raw = self.actor(obs)[0].cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("Policy produced nonfinite actions")
        action = np.clip(raw, -1, 1)
        desired = self.lower + (action[:5] + 1) * .5 * (self.upper - self.lower)
        ema = np.clip(self.alpha * desired + (1 - self.alpha) * self.filtered[:5], self.lower, self.upper)
        dt = 1.0 / self.hz
        if self.acceleration > 0:
            error = ema - self.filtered[:5]
            brake_speed = np.sqrt((self.acceleration * dt) ** 2 + 2 * self.acceleration * np.abs(error)) - self.acceleration * dt
            desired_velocity = np.sign(error) * np.minimum(brake_speed, self.speed_limits)
            velocity = self.arm_velocity + np.clip(
                desired_velocity - self.arm_velocity,
                -self.acceleration * dt,
                self.acceleration * dt,
            )
            arm_unclipped = self.filtered[:5] + velocity * dt
            arm = np.clip(arm_unclipped, self.lower, self.upper)
            self.arm_velocity[:] = np.where(np.isclose(arm, arm_unclipped, atol=1e-10), velocity, 0.0)
        else:
            arm = self.filtered[:5] + np.clip(ema - self.filtered[:5], -self.speed * dt, self.speed * dt)
        jaw = np.clip(self.grip_scale * action[5] + self.grip_offset, self.grip_min_close, 1.55)
        if self.motion_profile == "four_stage_smooth_v1" and self.grip_acceleration > 0 and self.grip_speed > 0:
            jaw_error = jaw - self.filtered[5]
            jaw_brake_speed = np.sqrt((self.grip_acceleration * dt) ** 2 + 2 * self.grip_acceleration * abs(jaw_error)) - self.grip_acceleration * dt
            jaw_desired_velocity = float(np.sign(jaw_error) * min(jaw_brake_speed, self.grip_speed))
            jaw_velocity = self.grip_velocity + float(np.clip(
                jaw_desired_velocity - self.grip_velocity,
                -self.grip_acceleration * dt,
                self.grip_acceleration * dt,
            ))
            jaw_unclipped = self.filtered[5] + jaw_velocity * dt
            jaw = float(np.clip(jaw_unclipped, 0, 1.55))
            self.grip_velocity = jaw_velocity if abs(jaw - jaw_unclipped) < 1e-10 else 0.0
        self.filtered = np.array([*arm, jaw])
        self.last_action = action.copy()
        self.steps += 1
        return raw, self.filtered.copy(), obs["policy"][0].cpu().numpy()
