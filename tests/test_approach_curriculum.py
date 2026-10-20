"""Approach training retains grasp practice and increases full-home exposure."""
import os
from unittest.mock import patch
import torch
from isaac_vision_rl.utils.approach_curriculum import sample_start_indices


def test_early_and_late_reset_distributions():
    torch.manual_seed(101)
    early = sample_start_indices(20000, 0, 41, "cpu")
    late = sample_start_indices(20000, 1, 41, "cpu")
    assert 0.08 < (early == 40).float().mean() < 0.12
    assert 0.47 < (late == 40).float().mean() < 0.54
    assert (late == 0).float().mean() > 0.18
    assert not ((early > 4) & (early < 40)).any()
    assert ((late > 4) & (late < 40)).any()


def test_opt_in_configuration_preserves_home_and_open_gripper():
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
    with patch.dict(os.environ, {"SO101_POLICY": "rgb", "SO101_CURRICULUM": "fixed",
                                "SO101_APPROACH_CURRICULUM": "1", "SO101_CUBE_JITTER": "0"}):
        cfg = PickLiftEnvCfg()
    params = cfg.events.reset_robot_joints.params
    assert len(params["start_poses"]) == 41
    assert all(p[-1] == 1.2 for p in params["start_poses"])
    home = params["start_poses"][-1]
    for name, value in zip(params["joint_names"][:-1], home[:-1]):
        assert value == cfg.scene.robot.init_state.joint_pos[name]
    assert cfg.terminations.unsafe_table_contact is not None
