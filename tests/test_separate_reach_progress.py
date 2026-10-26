import pytest
from test_opposite_side_grasp import sample
from isaac_vision_rl.utils.four_stage import StageMachine


def test_separate_posture_deltas_no_reset_or_stationary_credit():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    def update(wrist,jaw,center):
        s.update(sample(wrist_down=wrist,jaws_open=jaw,centered_xy=center),1/30)
        return s.group_raw[0,0,2:5].tolist()
    assert update(.5,.5,.5)==[0.,0.,0.]
    assert update(.5,.5,.5)==[0.,0.,0.]
    assert update(.6,.4,.5)==pytest.approx([.1,-.1,0.])
    assert update(.4,.4,.7)==pytest.approx([-.2,0.,.2])
    s.reset()
    assert update(.8,.8,.8)==[0.,0.,0.]
