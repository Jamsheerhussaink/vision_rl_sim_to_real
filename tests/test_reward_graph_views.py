import pytest
from isaac_vision_rl.utils.reward_panel import RewardPanel
from isaac_vision_rl.utils.four_stage import TERMS


def sample(age, reward, height=0.):
    groups=[[0. for _ in row] for row in TERMS]
    groups[0][2]=reward
    return dict(stage=0,reward_stage=0,episode_step=age,
                group_contribution=groups,cube_rise_cm=height)


def test_view_switch_totals_steps_and_height():
    panel=RewardPanel([])
    for age,reward,height in [(1,1.,0.),(2,0.,10.),(3,-2.,10.)]:
        panel.record(sample(age,reward,height))
    assert panel.term_history['wrist_down']==[1.,1.,-1.]
    panel.episode_total_view=False
    assert panel.chart_samples('wrist_down')[1].tolist()==[1.,0.,-2.]
    assert panel.chart_samples('cube_rise_cm')[1].tolist()==[0.,10.,10.]
    panel.episode_total_view=True
    assert panel.chart_samples('wrist_down')[1].tolist()==[1.,1.,-1.]
    panel.record(sample(0,0.))
    panel.record(sample(1,.5))
    assert panel.term_history['wrist_down'][-1]==pytest.approx(.5)
    panel.clear_history()
    assert not panel.step_full_history['wrist_down']
