"""Shared curriculum settings and evidence gates for staged SO-101 training."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("grasp", "fixed", "lift", "place")
SEEDS = (101, 202, 303)


def camera_fingerprint():
    paths = (
        "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py",
        "src/isaac_vision_rl/perception/cube.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/observations.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/rewards.py",
        "src/isaac_vision_rl/utils/grasp_geometry.py",
        "src/isaac_vision_rl/utils/grasp_assist.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/actions.py",
        "scripts/camera_pick_lift.py",
        "scripts/verify_camera_perception.py",
        "usd/robot_grasp_calibrated.usda",
    )
    return {path: sha256(ROOT / path) for path in paths}


def require_camera_preflight():
    """Require scored camera geometry and a physical camera-guided reference lift."""
    for name in ("sim_validation.json", "camera_lift.json"):
        path = ROOT / "outputs/perception" / name
        if not path.exists():
            raise ValueError("Camera evidence is missing; rerun --stage preflight before training.")
        report = json.loads(path.read_text())
        if not report.get("passed") or report.get("camera_fingerprint") != camera_fingerprint():
            raise ValueError("Camera evidence is failed or stale; rerun --stage preflight before training.")
        if name == "camera_lift.json" and not report.get("grasp_assist_used"):
            raise ValueError("Camera grasp assistance must pass preflight before training.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = [
        "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/config/so101/agents/rsl_rl_ppo_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/rewards.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/observations.py",
        "src/isaac_vision_rl/perception/cube.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/actions.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/actions_cfg.py",
        "src/isaac_vision_rl/tasks/pickplace/mdp/health.py",
        "src/isaac_vision_rl/utils/rl_agent.py",
        "src/isaac_vision_rl/utils/pipeline.py",
        "src/isaac_vision_rl/utils/grasp_geometry.py",
        "src/isaac_vision_rl/utils/grasp_assist.py",
        "scripts/eval_pipeline.py",
        "scripts/build_robot_asset.py",
        "scripts/check_workcell.py",
        "scripts/check_resets.py",
        "usd/box_reference.usda",
        "usd/box.usd",
        "usd/table_reference.usda",
        "usd/table.usd",
        "usd/robot.usd",
    ]
    calibrated = ROOT / "usd/robot_grasp_calibrated.usda"
    if calibrated.exists():
        paths.append(str(calibrated.relative_to(ROOT)))
    return {p: sha256(ROOT / p) for p in paths}


def read_physical_report():
    workcell_path = ROOT / "outputs/pipeline/workcell_physics.json"
    if not workcell_path.exists():
        raise ValueError("Table/box collision evidence is missing; rerun --stage preflight.")
    workcell = json.loads(workcell_path.read_text())
    if not workcell.get("passed") or any(
        workcell.get(key) != sha256(ROOT / path)
        for key, path in (
            ("env_config_hash", "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py"),
            ("box_reference_hash", "usd/box_reference.usda"),
            ("box_source_hash", "usd/box.usd"),
        )
    ):
        raise ValueError("Table/box collision evidence is failed or stale; rerun --stage preflight.")
    path = ROOT / "outputs/pipeline/physical_grasp.json"
    if not path.exists():
        raise ValueError("Run train_pipeline.py --stage preflight first: physical grasp evidence is missing.")
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise ValueError("Physical grasp failed; training is blocked until grasp geometry is corrected.")
    required = (
        "asset_hash",
        "robot_asset",
        "source_asset_hash",
        "env_config_hash",
        "above_joint_pos",
        "above_action",
        "grasp_action",
    )
    if any(key not in report for key in required) or not Path(report["robot_asset"]).is_file():
        raise ValueError("Physical grasp evidence is incomplete; rerun --stage preflight.")
    if (
        report["asset_hash"] != sha256(report["robot_asset"])
        or report["source_asset_hash"] != sha256(ROOT / "usd/robot.usd")
        or report["env_config_hash"] != sha256(ROOT / "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py")
    ):
        raise ValueError("Physical grasp evidence is stale; rerun --stage preflight.")
    return report


def configure_curriculum(cfg, stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown curriculum: {stage}")
    if stage == "grasp":
        report = read_physical_report()
        cfg.scene.robot.init_state.joint_pos.update(report["above_joint_pos"])
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
    if not 0 <= jitter <= 0.05:
        raise ValueError("Cube jitter must be between 0 and 0.05 metres")
    if jitter:
        cfg.events.reset_cube_pose.params["pose_range"] = {
            "x": (-jitter, jitter),
            "y": (-jitter, jitter),
            "yaw": (-0.52, 0.52),
        }
    box_jitter = float(os.environ.get("SO101_BOX_JITTER", "0"))
    if getattr(cfg.events, "reset_box_pose", None) is not None and box_jitter > 0:
        cfg.events.reset_box_pose.params["pose_range"] = {
            "x": (-box_jitter, box_jitter),
            "y": (-box_jitter, box_jitter),
            "yaw": (-0.35, 0.35),
        }


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
    policy = reports[0].get("policy", "vision") if reports else "vision"
    if any(r.get("policy", "vision") != policy for r in reports):
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
    if policy is not None and gate.get("policy", "vision") != policy:
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
