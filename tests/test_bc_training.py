import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_bc import weighted_bc_loss


def test_bc_loss_gives_gripper_action_more_weight_during_grasp():
    target = torch.zeros(2, 6)
    predicted = torch.zeros_like(target)
    predicted[:, 5] = 0.5
    policy = torch.zeros(2, 29)
    reach_loss = weighted_bc_loss(predicted[:1], target[:1], policy[:1])
    policy[1, 26] = 1.0  # Grasp phase one-hot input.
    grasp_loss = weighted_bc_loss(predicted[1:], target[1:], policy[1:])
    assert grasp_loss > reach_loss
