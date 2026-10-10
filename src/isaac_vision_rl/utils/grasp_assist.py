"""Camera-only capture trigger with episode-local grasp memory."""

import torch


class CameraGraspLatch:
    def __init__(self, count, device, xy_tolerance=0.012, z_tolerance=0.015, min_confidence=0.75, confirm_steps=2):
        if min(xy_tolerance, z_tolerance) <= 0 or confirm_steps < 1 or not 0 <= min_confidence <= 1:
            raise ValueError("Invalid camera grasp thresholds")
        self.xy_tolerance = xy_tolerance
        self.z_tolerance = z_tolerance
        self.min_confidence = min_confidence
        self.confirm_steps = confirm_steps
        self.latched = torch.zeros(count, dtype=torch.bool, device=device)
        self.ready_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.hold_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.closure_hold_steps = 15
        self.gates = {}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.latched[ids] = False
        self.ready_steps[ids] = 0
        self.hold_steps[ids] = 0

    def update(self, cube_tool_xyz, confidence, visible, tool_downward):
        self.gates = {
            "finite": torch.isfinite(cube_tool_xyz).all(dim=-1),
            "visible": visible.bool(),
            "confidence": confidence >= self.min_confidence,
            "downward": tool_downward.bool(),
            "xy_alignment": torch.linalg.vector_norm(cube_tool_xyz[:, :2], dim=-1) <= self.xy_tolerance,
            "z_alignment": cube_tool_xyz[:, 2].abs() <= self.z_tolerance,
        }
        ready = torch.stack(list(self.gates.values())).all(dim=0)
        self.hold_steps[:] = (self.hold_steps - 1).clamp_min(0)
        self.ready_steps[:] = torch.where(ready, self.ready_steps + 1, 0)
        captured = ~self.latched & (self.ready_steps >= self.confirm_steps)
        self.hold_steps[captured] = self.closure_hold_steps
        self.latched |= self.ready_steps >= self.confirm_steps
        return self.latched
