from types import SimpleNamespace as NS
from unittest.mock import Mock
import torch
from isaac_vision_rl.utils.training_reward_charts import update_training_charts


def test_training_charts_read_env_zero_without_changing_rewards():
    viewer=NS(register_ui_callback=Mock())
    env=NS(sim=NS(visualizers=[NS(_viewer=viewer)]),episode_length_buf=torch.tensor([5,8]),common_step_counter=1)
    s=NS(stage=torch.tensor([0,1]),reward_stage=torch.tensor([0,1]),
         raw=torch.tensor([[.1,.2,.3,0.,0.,0.],[.9,.8,.7,0.,0.,0.]]),
         contribution=torch.tensor([[.8,.4,-.6,0.,0.,0.],[3.,4.,-1.,0.,0.,0.]]),
         failure=torch.zeros(2,dtype=torch.bool),success=torch.zeros(2,dtype=torch.bool))
    s.group_contribution=torch.zeros(2,4,6)
    s.group_contribution[:,0]=s.contribution
    previous=s.contribution.clone()
    update_training_charts(env,s)
    p=env._training_reward_panel
    assert p.snapshot['contribution']==s.contribution[0].tolist()
    assert p.sample_count==1 and p.names==[]
    assert abs(p.approach_running_total-.8)<1e-6
    assert abs(p.readiness_running_total+.6)<1e-6
    assert torch.equal(s.contribution,previous)
    viewer.register_ui_callback.assert_called_once()
    env.episode_length_buf[0]=1
    update_training_charts(env,s)
    assert abs(p.total-float(previous[0].sum()))<1e-6
    assert p.sample_count==2  # Training histories persist across episode resets.
    assert abs(p.approach_running_total-.8)<1e-6
    assert abs(p.readiness_running_total+.6)<1e-6
    # A stationary step adds zero; plotted totals retain their existing values.
    env.episode_length_buf[0]=2
    s.contribution.zero_();s.group_contribution.zero_()
    update_training_charts(env,s)
    assert abs(p.term_history['overhead_approach'][-1]-.8)<1e-6
    assert abs(p.term_history['wrist_down'][-1]+.6)<1e-6


def test_standalone_charts_work_without_newton_and_send_environment_zero(monkeypatch):
    monkeypatch.setenv('SO101_REWARD_WINDOW','1')
    window=Mock()
    monkeypatch.setattr('isaac_vision_rl.utils.reward_window.RewardWindow',lambda:window)
    env=NS(sim=NS(visualizers=[]),episode_length_buf=torch.tensor([2,3]),common_step_counter=1)
    groups=torch.zeros(2,4,6);groups[0,0,0]=.12;groups[1,0,0]=.99
    s=NS(stage=torch.tensor([0,0]),reward_stage=torch.tensor([0,0]),
         raw=groups.sum(1),contribution=groups.sum(1),group_contribution=groups,
         failure=torch.zeros(2,dtype=torch.bool),success=torch.zeros(2,dtype=torch.bool))
    update_training_charts(env,s)
    window.send.assert_called_once()
    sent=window.send.call_args.args[0]
    assert abs(sent['group_contribution'][0][0]-.12)<1e-6
