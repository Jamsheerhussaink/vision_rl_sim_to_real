import os
from unittest.mock import patch
from isaac_vision_rl.utils.manual_inspection import configure_manual_inspection
from isaac_vision_rl.utils.four_stage import StageMachine
from test_four_stage import values


def test_manual_disables_all_terminations_and_uses_one_environment_only():
    from isaaclab.managers import TerminationTermCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    with patch.dict(os.environ,{'SO101_REWARD_PROFILE':'four-stage','SO101_CURRICULUM':'place','SO101_POLICY':'rgb'}):
        manual=PickplaceEnvCfg();training=PickplaceEnvCfg()
    configure_manual_inspection(manual)
    assert manual.scene.num_envs==1
    assert not any(isinstance(v,TerminationTermCfg) for v in vars(manual.terminations).values())
    assert training.terminations.unsafe_table_contact is not None
    assert training.terminations.time_out is not None
    assert training.terminations.placed is not None


def test_manual_failure_does_not_latch_and_safe_stage_can_recover():
    s=StageMachine(1,'cpu')
    for _ in range(8):s.update(values(unsafe=True),.05,inspection=True)
    assert not s.failure.item() and s.stage.item()==0
    for _ in range(4):s.update(values(),.05,inspection=True)
    assert s.stage.item()==1
    normal=StageMachine(1,'cpu');normal.update(values(unsafe=True),.05)
    assert normal.failure.item()
