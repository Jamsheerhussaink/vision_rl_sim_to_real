import torch
from test_four_stage import values
from isaac_vision_rl.utils.four_stage import StageMachine


def attempt(**changes):
    m=dict(static_finger_reach=.3,moving_finger_reach=.3,bilateral=False,fixed_contact_force_N=1.,moving_contact_force_N=1.,
           fixed_contact_in_pad=True,moving_contact_in_pad=False,
           opposing_contact_forces=False,enclosed=True,jaw=.99,reach_ready=False)
    m.update(changes)
    return values(**m)


def test_imperfect_two_finger_hold_enters_lift_after_dwell():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=1
    s.update(attempt(),.1)
    assert s.stage.item()==1
    for _ in range(6):s.update(attempt(),.1)
    assert s.stage.item()==2
    assert not s.failure.item()


def test_one_finger_open_jaws_no_enclosure_or_unsafe_cannot_pass():
    for changes in [dict(moving_contact_force_N=0.),dict(fixed_contact_force_N=0.),
                    dict(enclosed=False),dict(jaw=1.2),dict(unsafe=True)]:
        s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous');s.stage[:]=1
        for _ in range(8):s.update(attempt(**changes),.1)
        assert s.stage.item()!=2


def test_contact_loss_resets_hold_and_still_fails_after_lift():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous');s.stage[:]=1
    s.update(attempt(),.2)
    s.update(attempt(moving_contact_force_N=0.),.1)
    assert s.acquisition_dwell.item()==0
    s.update(attempt(),.2)
    assert s.stage.item()==1
    for _ in range(5):s.update(attempt(),.1)
    assert s.stage.item()==2
    for _ in range(3):s.update(attempt(moving_contact_force_N=0.),.1)
    assert s.failure.item()
