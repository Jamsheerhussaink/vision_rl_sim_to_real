"""Fail early on invalid dynamics and preserve the offending physical state."""

import json
from pathlib import Path

import torch


def assert_finite_physics(env):
    fields = {}
    for asset_name, names in (
        ("robot", ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")),
        ("cube", ("root_pos_w", "root_lin_vel_w", "root_quat_w")),
    ):
        data = env.scene[asset_name].data
        for name in names:
            if hasattr(data, name):
                value = getattr(data, name)
                fields[f"{asset_name}/{name}"] = value.torch if hasattr(value, "torch") else value
    combined = torch.cat([value.reshape(env.num_envs, -1) for value in fields.values()], dim=-1)
    invalid = ~torch.isfinite(combined).all(dim=-1)
    if invalid.any():
        ids = torch.nonzero(invalid).flatten()
        bad_fields = [name for name, value in fields.items() if not torch.isfinite(value[ids]).all()]
        report = {
            "env_ids": ids.cpu().tolist(),
            "bad_fields": bad_fields,
            "states": {name: value[ids[:2]].detach().cpu().tolist() for name, value in fields.items()},
            "reason": "Non-finite articulation or object dynamics; values were not replaced or hidden",
        }
        target = Path(__file__).resolve().parents[5] / "outputs/pipeline/invalid_physics_state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        raise FloatingPointError(f"Invalid physics in environments {report['env_ids']}: {bad_fields}. See {target}")
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
