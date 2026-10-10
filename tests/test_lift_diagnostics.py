"""Diagnostic thresholds must identify failure without loosening task success."""

import runpy
from pathlib import Path

diagnose = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/eval_pipeline.py"))[
    "lift_condition_diagnostics"
]


def test_wide_jaw_isolated_from_other_valid_hold_conditions():
    result = diagnose(0.55, 0.01, 0.02, 1.21, 0.01, {"fixed": 1.0, "moving": 1.0})
    assert result["all_except_jaw_angle"]
    assert not result["jaw_angle"]
    assert not result["all_conditions"]


def test_other_condition_failures_are_not_mislabelled_as_jaw_only():
    for height, xy, z, speed, forces, failed in [
        (0.47, 0.01, 0.02, 0.01, {"a": 1.0, "b": 1.0}, "height"),
        (0.55, 0.035, 0.02, 0.01, {"a": 1.0, "b": 1.0}, "xy_separation"),
        (0.55, 0.01, 0.045, 0.01, {"a": 1.0, "b": 1.0}, "z_separation"),
        (0.55, 0.01, 0.02, 0.25, {"a": 1.0, "b": 1.0}, "cube_speed"),
        (0.55, 0.01, 0.02, 0.01, {"a": 1.0, "b": 0.02}, "bilateral_contact"),
    ]:
        result = diagnose(height, xy, z, 0.97, speed, forces)
        assert not result[failed]
        assert not result["all_except_jaw_angle"]


def test_valid_hold_and_missing_contact_data():
    assert diagnose(0.55, 0.01, 0.02, 0.97, 0.01, {"a": 1.0, "b": 1.0})["all_conditions"]
    assert not diagnose(0.55, 0.01, 0.02, 0.97, 0.01, {})["all_conditions"]
