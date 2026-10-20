"""Shared curriculum settings and evidence gates for staged SO-101 training."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("grasp", "fixed", "lift", "place")
SEEDS = (101, 202, 303)






def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = [
        "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/config/so101/agents/rsl_rl_ppo_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/rewards.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/terminations.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/observations.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/actions.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/actions_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/health.py",
        "src/isaac_vision_rl/utils/rl_agent.py",
        "src/isaac_vision_rl/utils/pipeline.py",
        "src/isaac_vision_rl/utils/grasp_geometry.py",
        "src/isaac_vision_rl/utils/approach_curriculum.py",
        "scripts/eval_pipeline.py",
        "scripts/assets/build_robot_asset.py",
        "scripts/checks/check_workcell.py",
        "scripts/checks/check_resets.py",
        "usd/box_reference.usda",
        "usd/box.usd",
        "usd/table_reference.usda",
        "usd/table.usd",
        "usd/robot.usd",
    ]
    calibrated = ROOT / "usd/robot_grasp_surface_v3.usda"
    if calibrated.exists():
        paths.append(str(calibrated.relative_to(ROOT)))
    return {p: sha256(ROOT / p) for p in paths}


def read_physical_report(strict=True):
    workcell_path = ROOT / "outputs/pipeline/workcell_physics.json"
    if not workcell_path.exists():
        raise ValueError("Table/box collision evidence is missing; rerun --stage preflight.")
    workcell = json.loads(workcell_path.read_text())
    if not workcell.get("passed"):
        raise ValueError("Table/box collision evidence is failed; rerun --stage preflight.")
    stale_workcell = any(
        workcell.get(key) != sha256(ROOT / path)
        for key, path in (
            ("env_config_hash", "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py"),
            ("box_reference_hash", "usd/box_reference.usda"),
            ("box_source_hash", "usd/box.usd"),
        )
    )
    if stale_workcell and strict:
        raise ValueError("Table/box collision evidence is failed or stale; rerun --stage preflight.")
    path = ROOT / "outputs/pipeline/physical_grasp.json"
    if not path.exists():
        raise ValueError("Run train_pipeline.py --stage preflight first: physical grasp evidence is missing.")
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise ValueError("Physical grasp failed; training is blocked until grasp geometry is corrected.")
    if not report.get("table_safe") or report.get("minimum_hand_table_clearance_m", 0) < 0.005:
        raise ValueError("Table-safe grasp evidence is missing or failed; rerun --stage preflight.")
    required = (
        "asset_hash",
        "robot_asset",
        "source_asset_hash",
        "env_config_hash",
        "above_joint_pos",
        "grasp_joint_pos",
        "base_pos",
        "base_rot",
        "above_action",
        "grasp_action",
    )
    if any(key not in report for key in required) or not Path(report["robot_asset"]).is_file():
        raise ValueError("Physical grasp evidence is incomplete; rerun --stage preflight.")
    stale_grasp = (
        report["asset_hash"] != sha256(report["robot_asset"])
        or report["source_asset_hash"] != sha256(ROOT / "usd/robot.usd")
        or report["env_config_hash"] != sha256(ROOT / "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py")
    )
    if stale_grasp and strict:
        raise ValueError("Physical grasp evidence is stale; rerun --stage preflight.")
    return report


def configure_curriculum(cfg, stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown curriculum: {stage}")
    if os.environ.get("SO101_POLICY", "rgb") != "rgb":
        raise ValueError("Only RGB policies are supported")
    # No depth sensor, geometry cue, capture latch, or privileged release.
    fps = int(os.environ.get("SO101_RGB_FPS", "30"))
    if fps not in (25, 30, 60):
        raise ValueError("SO101_RGB_FPS must be 25 (legacy), 30 or 60")
    cfg.decimation = 4
    cfg.sim.dt = 1.0 / (4 * fps)
    cfg.sim.render_interval = 4
    cfg.scene.camera.update_period = 0.0  # Render fresh at every control tick.
    # Preserve the filter's approximate physical-time response.
    cfg.actions.arm_action.alpha = 1.0 - 0.8 ** (25.0 / fps)
    cfg.scene.camera.data_types = ["rgb"]
    cfg.rewards.reach_progress.params["time_normalized"] = True
    # Dense bounded shaping bridges approach -> alignment -> supported lift.
    # Actor inputs and strict termination gates remain unchanged.
    from isaaclab.managers import RewardTermCfg
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import dense_rgb_approach
    cfg.rewards.dense_reach = RewardTermCfg(func=dense_rgb_approach, weight=2.0, params={"component": "reach"})
    cfg.rewards.dense_alignment = RewardTermCfg(func=dense_rgb_approach, weight=3.0, params={"component": "align"})
    cfg.rewards.approach_open = RewardTermCfg(func=dense_rgb_approach, weight=0.5, params={"component": "open"})
    cfg.rewards.stable_support = RewardTermCfg(func=dense_rgb_approach, weight=10.0, params={"component": "support"})
    cfg.rewards.displacement_penalty.weight = float(os.environ.get("SO101_DISPLACEMENT_PENALTY", "-12.0"))
    cfg.rewards.action_rate.weight = float(os.environ.get("SO101_ACTION_RATE_PENALTY", "-0.06"))

    cam_w = int(os.environ.get("SO101_CAMERA_WIDTH", "1280"))
    cam_h = int(os.environ.get("SO101_CAMERA_HEIGHT", "720"))
    cfg.scene.camera.width = cam_w
    cfg.scene.camera.height = cam_h
    # Keep the original close viewpoint, with a small upward framing shift.
    # Explicit OpenGL axes: -Z forward, +Y up.
    cfg.scene.camera.offset.pos = (-0.05, -0.22, 0.77)
    cfg.scene.camera.offset.convention = "opengl"
    cfg.scene.camera.offset.rot = (0.3131639, 0.0, 0.0, 0.9496994)
    cfg.scene.camera.spawn.horizontal_aperture = 24.88
    cfg.scene.camera.spawn.focal_length = 12.44  # Nominal 90-degree horizontal FOV.
    # 30 cm horizontal camera/base distance, on the image's left side.
    # Base offset from camera: (-0.20, sqrt(0.30**2 - 0.20**2)).
    cfg.scene.robot.init_state.pos = (-0.25, 0.015, 0.405)
    cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    # Move the objects with the base, preserving the arm-relative layout.
    # Align cube and box to match real workspace photo pixel measurements.
    cfg.scene.cube.init_state.pos = (-0.002, -0.052, 0.426)
    cfg.scene.box.init_state.pos = (0.055, 0.120, 0.401)
    # Actual photo-reference home, captured from all six real encoders.
    from isaac_vision_rl.utils.home_pose import home_joint_positions
    cfg.scene.robot.init_state.joint_pos.update(home_joint_positions())
    cfg.scene.robot.init_state.joint_pos["gripper"] = 1.20
    # Axis-align the base; compensate shoulder pan to retain arm heading.
    from isaac_vision_rl.utils.cube_material import spawn_grain_material, spawn_cardboard_material
    cfg.scene.cube.spawn.visual_material.func = spawn_grain_material
    cfg.scene.box.spawn.visual_material.func = spawn_cardboard_material
    cfg.scene.cube.spawn.visual_material.diffuse_color_constant = (0.85, 0.70, 0.35)
    cfg.scene.box.spawn.visual_material.diffuse_color_constant = (0.75, 0.55, 0.40)
    cfg.scene.cube.spawn.visual_material.reflection_roughness_constant = 0.55
    cfg.scene.box.spawn.visual_material.reflection_roughness_constant = 0.85
    cfg.scene.table.spawn.usd_path = str(ROOT / "usd/table_real_rgb.usda")
    cfg.scene.table.spawn.scale = (3.5, 3.5, 1.0)
    cfg.scene.table.init_state.pos = (0.0, 0.35, 0.0)
    # Soft, natural ambient illumination matching the real physical room.
    cfg.scene.dome_light.spawn.intensity = 1200.0
    cfg.scene.dome_light.spawn.color = (0.95, 0.98, 1.0)
    cfg.scene.overhead_front.spawn.intensity = 1500.0
    cfg.scene.overhead_front.spawn.radius = 0.80
    cfg.scene.overhead_front.init_state.pos = (0.02, 0.51, 1.45)
    cfg.scene.overhead_front.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    cfg.scene.overhead_front.spawn.color = (0.98, 0.98, 1.0)
    cfg.scene.overhead_left.spawn.radius = 0.80
    cfg.scene.overhead_left.init_state.pos = (-0.40, 0.65, 1.35)
    cfg.scene.overhead_left.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    cfg.scene.overhead_left.spawn.intensity = 3500.0
    cfg.scene.overhead_left.spawn.color = (0.98, 0.99, 1.0)
    cfg.scene.overhead_table.spawn.intensity = 3000.0
    cfg.scene.overhead_table.spawn.radius = 0.90
    cfg.scene.overhead_table.init_state.pos = (0.15, 0.85, 1.45)
    cfg.scene.camera.background_color = (0.04, 0.07, 0.05)
    cfg.scene.camera.background_color = (0.08, 0.12, 0.09)
    cfg.sim.default_visualizer_cfg.eye = cfg.scene.camera.offset.pos
    cfg.sim.default_visualizer_cfg.lookat = (-0.08, 0.10, 0.33)
    cfg.scene.camera.spawn.vertical_aperture = cfg.scene.camera.spawn.horizontal_aperture * cam_h / cam_w
    # Symmetric mapping: -1 closed, 0 half open, +1 fully open.
    cfg.actions.gripper_action.scale = 0.6
    cfg.actions.gripper_action.offset = 0.6
    if stage == "grasp":
        report = read_physical_report()
        if any(abs(a - b) > 1e-6 for a, b in zip(report["base_pos"], cfg.scene.robot.init_state.pos, strict=True)) or any(
            abs(a - b) > 1e-6 for a, b in zip(report["base_rot"], cfg.scene.robot.init_state.rot, strict=True)
        ):
            raise ValueError("Grasp reset base pose differs from physical evidence; rerun --stage preflight.")
        pose_mode = os.environ.get("SO101_GRASP_POSE", "enclosed")
        if pose_mode not in ("above", "enclosed"):
            raise ValueError("SO101_GRASP_POSE must be 'above' or 'enclosed'")
        if pose_mode == "above":
            cfg.scene.robot.init_state.joint_pos.update(report["above_joint_pos"])
        else:
            cfg.scene.robot.init_state.joint_pos.update(report["grasp_joint_pos"])
    if stage in ("grasp", "fixed"):
        # Fixed conditions still need an explicit pose AND velocity reset.
        # RigidObject.reset() clears buffers/wrenches, not the object's pose.
        if cfg.events.reset_cube_pose is None:
            raise ValueError("Cube pose reset must remain enabled in fixed curricula")
        cfg.events.reset_cube_pose.params["pose_range"] = {}
        cfg.events.reset_cube_pose.params["velocity_range"] = {}
        if getattr(cfg.events, "reset_box_pose", None) is not None:
            cfg.events.reset_box_pose.params["pose_range"] = {}
            cfg.events.reset_box_pose.params["velocity_range"] = {}
        cfg.events.randomize_cube_mass = None
        cfg.events.randomize_cube_friction = None
    jitter = float(os.environ.get("SO101_CUBE_JITTER", "0"))
    if os.environ.get("SO101_APPROACH_CURRICULUM", "0") == "1":
        if stage != "fixed" or jitter != 0:
            raise ValueError("Approach reset curriculum requires fixed cube conditions")
        import numpy as np
        from isaaclab.managers import EventTermCfg
        from isaac_vision_rl.utils.approach_curriculum import reset_approach_curriculum
        from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES
        report = read_physical_report()
        if not np.allclose(report["base_pos"], cfg.scene.robot.init_state.pos, atol=1e-6) or not np.allclose(
            report["base_rot"], cfg.scene.robot.init_state.rot, atol=1e-6
        ):
            raise ValueError("Approach grasp evidence uses a different base pose")
        names = ARM_NAMES + ["gripper"]
        near = np.array([report["grasp_joint_pos"][n] for n in ARM_NAMES] + [1.2])
        home = np.array([cfg.scene.robot.init_state.joint_pos[n] for n in ARM_NAMES] + [1.2])
        poses = [(1 - t) * near + t * home for t in np.linspace(0, 1, 41)]
        for pose in poses:
            pose[-1] = 1.2
        ramp = int(os.environ.get("SO101_APPROACH_RAMP_UPDATES", "1000"))
        if ramp < 1:
            raise ValueError("Approach ramp must be positive")
        cfg.events.reset_robot_joints = EventTermCfg(func=reset_approach_curriculum, mode="reset", params={
            "joint_names": names, "start_poses": [p.tolist() for p in poses],
            "ramp_steps": ramp * round(32 * fps / 25),
        })
    if not 0 <= jitter <= 0.05:
        raise ValueError("Cube jitter must be between 0 and 0.05 metres")
    if jitter:
        cfg.events.reset_cube_pose.params["pose_range"] = {
            "x": (-jitter, jitter),
            "y": (-jitter, jitter),
            "yaw": (-0.7854, 0.7854),
        }
    box_jitter = float(os.environ.get("SO101_BOX_JITTER", "0"))
    if getattr(cfg.events, "reset_box_pose", None) is not None and box_jitter > 0:
        cfg.events.reset_box_pose.params["pose_range"] = {
            "x": (-box_jitter, box_jitter),
            "y": (-box_jitter, box_jitter),
            "yaw": (-0.35, 0.35),
        }
    if os.environ.get("SO101_REWARD_PROFILE", "legacy") == "four-stage":
        if stage != "place" or os.environ.get("SO101_APPROACH_CURRICULUM", "0") == "1":
            raise ValueError("Four-stage task requires --task place --curriculum place and home resets")
        from isaac_vision_rl.utils.four_stage import configure
        configure(cfg)


def evaluations_pass(reports, episodes=50):
    return len(reports) == len(SEEDS) and all(
        r.get("seed") == seed
        and r.get("episodes") == episodes
        and r.get("success_rate", 0) >= 0.90
        and r.get("mean_table_slide_cm", float("inf")) <= 1.0
        for seed, r in zip(SEEDS, reports, strict=True)
    )


def write_gate(stage, checkpoint, reports, episodes=50):
    checkpoint = Path(checkpoint).resolve()
    if any(Path(r.get("checkpoint", "")).resolve() != checkpoint for r in reports):
        raise ValueError("Evaluation checkpoints do not match the candidate.")
    if any(r.get("curriculum") != stage or r.get("task") != ("place" if stage == "place" else "lift") for r in reports):
        raise ValueError("Evaluation curriculum or task does not match the gate.")
    policy = reports[0].get("policy", "rgb") if reports else "rgb"
    if any(r.get("policy", "rgb") != policy for r in reports):
        raise ValueError("Mixed policy variants cannot certify one checkpoint")
    gate = {
        "policy": policy,
        "stage": stage,
        "passed": evaluations_pass(reports, episodes),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": sha256(checkpoint),
        "fingerprint": fingerprint(),
        "episodes_per_seed": episodes,
        "threshold": 0.90,
        "max_mean_table_slide_cm": 1.0,
        "evaluations": reports,
    }
    target = ROOT / "outputs/pipeline" / f"{stage}_gate.json"
    variant_target = ROOT / "outputs/pipeline" / f"{stage}_{policy}_gate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(gate, indent=2) + "\n")
    variant_target.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def require_gate(stage, policy=None):
    path = ROOT / "outputs/pipeline" / f"{stage}_gate.json"
    if not path.exists():
        raise ValueError(f"{stage} mastery evidence is missing. Train and evaluate that stage first.")
    gate = json.loads(path.read_text())
    if policy is not None and gate.get("policy", "rgb") != policy:
        raise ValueError(f"{stage} gate belongs to another policy; use matching --policy and checkpoint")
    checkpoint = Path(gate["checkpoint"])
    if not gate["passed"] or not evaluations_pass(gate["evaluations"], gate["episodes_per_seed"]):
        raise ValueError(f"{stage} has not passed its mastery gate; do not advance.")
    if gate.get("episodes_per_seed", 0) < 50 or gate.get("stage") != stage:
        raise ValueError(f"{stage} evidence does not meet the required evaluation protocol.")
    if (
        gate["fingerprint"] != fingerprint()
        or not checkpoint.is_file()
        or gate["checkpoint_hash"] != sha256(checkpoint)
    ):
        raise ValueError(f"{stage} evidence is stale or checkpoint changed; reevaluate before advancing.")
    return str(checkpoint)
