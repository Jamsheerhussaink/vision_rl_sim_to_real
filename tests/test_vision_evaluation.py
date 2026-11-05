"""Check image interventions and uninterrupted hold accounting without physics."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "vision_evaluation", Path(__file__).resolve().parents[1] / "scripts/evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_camera_interventions_preserve_stack_and_have_no_self_donors():
    current = np.arange(4 * 2 * 2 * 9).reshape(4, 2, 2, 9)
    initial = current + 1000
    np.testing.assert_array_equal(module.camera_input(0, current, initial, np), current)
    np.testing.assert_array_equal(module.camera_input(1, current, initial, np), initial)
    shuffled = module.camera_input(2, current, initial, np)
    for w in range(4):
        np.testing.assert_array_equal(shuffled[w], current[(w - 1) % 4])
        assert not np.array_equal(shuffled[w], current[w])


def test_hold_requires_consecutive_steps_and_ignores_auto_reset_episode():
    streak, longest = np.zeros(2), np.zeros(2)
    for condition, active in [([1, 1], [1, 1]), ([0, 1], [1, 1]),
                              ([1, 1], [1, 0]), ([1, 1], [1, 0])]:
        streak, longest = module.advance_streak(
            streak, longest, np.array(condition, bool), np.array(active, bool), np)
    np.testing.assert_array_equal(longest, [2, 2])
