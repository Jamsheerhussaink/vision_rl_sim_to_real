"""Capture trigger, occlusion latch and per-environment reset regressions."""

import unittest

import torch

from isaac_vision_rl.utils.grasp_assist import CameraGraspLatch


class GraspAssistTests(unittest.TestCase):
    def test_gate_diagnostics_identify_alignment_and_occlusion_separately(self):
        latch = CameraGraspLatch(2, "cpu")
        latch.update(torch.tensor([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                     torch.ones(2), torch.tensor([1, 0]), torch.ones(2))
        self.assertEqual(latch.gates["xy_alignment"].tolist(), [False, True])
        self.assertEqual(latch.gates["visible"].tolist(), [True, False])
        self.assertFalse(latch.latched.any())

    def test_centre_closes_after_confirmation_and_occlusion_does_not_release(self):
        latch = CameraGraspLatch(2, "cpu")
        positions = torch.tensor([[0.002, 0.003, 0.005], [0.04, 0.0, 0.0]])
        confidence, valid, down = torch.ones(2), torch.ones(2), torch.ones(2)
        self.assertEqual(latch.update(positions, confidence, valid, down).tolist(), [False, False])
        self.assertEqual(latch.update(positions, confidence, valid, down).tolist(), [True, False])
        self.assertEqual(latch.update(positions, confidence, torch.zeros(2), down).tolist(), [True, False])
        latch.reset(torch.tensor([0]))
        self.assertEqual(latch.latched.tolist(), [False, False])

    def test_invalid_or_high_or_low_confidence_or_tilted_view_cannot_trigger(self):
        for positions, confidence, valid, down in (
            ([0, 0, 0.04], 1, 1, 1),
            ([0, 0, 0], 0.2, 1, 1),
            ([0, 0, 0], 1, 0, 1),
            ([0, 0, 0], 1, 1, 0),
            ([float("nan"), 0, 0], 1, 1, 1),
        ):
            latch = CameraGraspLatch(1, "cpu")
            for _ in range(3):
                state = latch.update(
                    torch.tensor([positions], dtype=torch.float32),
                    torch.tensor([confidence]),
                    torch.tensor([valid]),
                    torch.tensor([down]),
                )
                self.assertFalse(state.any())
