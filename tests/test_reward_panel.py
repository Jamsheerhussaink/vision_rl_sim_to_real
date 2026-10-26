from isaac_vision_rl.utils.reward_panel import RewardPanel

class UI:
    class Cond_: first_use_ever=0
    def __init__(self,clicked=(),active=''):self.clicked=clicked;self.active=active;self.last=''
    def begin(self,*args):return True,None
    def button(self,label):self.last=label;return label in self.clicked
    def is_item_active(self):return self.last==self.active
    def slider_float(self,label,value,*args):return False,value
    def __getattr__(self,name):return lambda *args:None


def test_jog_is_held_not_latched_and_safety_lock_blocks_motion():
    p=RewardPanel(['joint']*6)
    p.draw(UI(active='+##joint5'));assert p.velocity==[0.,0.,0.,0.,0.,.25]
    p.draw(UI());assert not any(p.velocity)
    p.locked=True;p.paused=True
    p.draw(UI(clicked=['Run'],active='+##joint5'))
    assert p.paused and not any(p.velocity)


def test_stop_and_explicit_home_reset_are_distinct_requests():
    p=RewardPanel(['joint']*6);p.paused=False
    p.draw(UI(clicked=['STOP / HOLD']))
    assert p.stop_requested and p.paused and not p.reset_requested
    p.draw(UI(clicked=['Reset to HOME']))
    assert p.reset_requested


def test_each_reward_has_a_separate_signed_history_and_reset():
    from isaac_vision_rl.utils.four_stage import TERMS
    p=RewardPanel(['joint']*6)
    p.record(dict(reward_stage=0,contribution=[.2,-.3,0.]))
    assert p.term_history['table_clearance'][-1]==-.3
    p.record(dict(reward_stage=1,contribution=[.4,.5,-.6]))
    assert p.term_history['table_clearance'][-1]==0
    assert p.term_history['table_slide_speed'][-1]==-.6
    assert len(p.term_history)==12
    p.clear_history()
    assert all(not any(h) for h in p.term_history.values())


def test_overlapping_rewards_record_separate_families():
    p=RewardPanel([])
    p.record(dict(reward_stage=1,contribution=[.5,.7,-.2],
                  group_contribution=[[.1,.2,-.1],[.4,.5,-.1],[0.,0.,0.],[0.,0.,0.]]))
    assert p.term_history['overhead_approach']==[.1]
    assert p.term_history['gripper_readiness']==[-.1]
    assert p.term_history['current_enclosure']==[.4]
    assert p.term_history['supported_lift_progress']==[0.]


def test_camera_margin_records_as_its_own_reach_graph():
    from isaac_vision_rl.utils.four_stage import TERMS
    from isaac_vision_rl.utils.reward_panel import display_name
    p=RewardPanel([])
    groups=[[0.,0.,0.,0.,0.,-.1],[0.]*5,[0.]*3,[0.]*3]
    p.record(dict(reward_stage=0,group_contribution=groups,contribution=[-.1,0.,0.,0.,0.,0.]))
    assert 'camera_visibility' in TERMS[0]
    assert display_name('camera_visibility')=='Camera margin'
    assert p.term_history['camera_visibility']==[-.1]


def test_approach_chart_is_running_total_and_can_go_below_zero():
    p=RewardPanel([])
    for total in (0.,.3,.1,-.2):
        p.record(dict(reward_stage=0,contribution=[0.,0.,0.],
                      group_contribution=[[.3,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
                      approach_progress_total=total))
    assert p.term_history['overhead_approach']==[0.,.3,.1,-.2]


def test_approach_chart_accumulates_manual_step_rewards_from_zero():
    p=RewardPanel([])
    for step in (0.,.3,0.,-.5):
        p.record(dict(reward_stage=0,contribution=[step,0.,0.]))
    assert p.term_history['overhead_approach']==[0.,.3,.3,-.2]
    p.clear_history()
    assert p.approach_running_total==0.


def test_readiness_graph_holds_its_total_while_stationary():
    p=RewardPanel([])
    for step in (0.,.1,0.,0.,-.2):
        p.record(dict(reward_stage=0,contribution=[0.,0.,step]))
    assert p.term_history['gripper_readiness']==[0.,.1,.1,.1,-.1]
    p.clear_history()
    assert p.readiness_running_total==0.


def test_confirmed_reward_table_names_and_continuous_weights_match():
    from isaac_vision_rl.utils.reward_panel import display_name
    from isaac_vision_rl.utils.four_stage import TERMS, CONTINUOUS_WEIGHTS
    expected=[('Approach distance',.30),('Table clearance',-1.),
              ('Wrist downward',.05),('Jaws open',.05),('Centered XY',.10),
              ('Camera margin',-.25),('Fixed Finger Near Cube Side',.05),
              ('Moving Finger Near Opposite Side',.05),('Cube enclosure',.10),
              ('Valid two-finger grasp',.30),('Table dragging',-4.),
              ('Lift reward',.15),('Stable held lift',.15),
              ('Cube speed during Lift',-1.),('Carry toward box',.15),
              ('Lower into box',.15),('Successful placement',1.)]
    actual=[(display_name(term),weight) for terms,weights in zip(TERMS,CONTINUOUS_WEIGHTS)
            for term,weight in zip(terms,weights)]
    assert actual==expected


def test_chart_defaults_to_live_window_and_can_show_full_session():
    p=RewardPanel([])
    for i in range(3100):p.record(dict(reward_stage=0,contribution=[0.,float(i),0.]))
    x,y=p.chart_samples('table_clearance')
    assert not p.show_full_history and len(y)==150
    assert x[0]==2950 and y[-1]==3099
    p.show_full_history=True
    x,y=p.chart_samples('table_clearance')
    assert x[0]==0 and len(y)==3100 and y[0]==0 and y[-1]==3099
    p.clear_history()
    assert not p.full_history['table_clearance']


def test_full_history_preserves_isolated_penalty_spikes_when_reduced():
    p=RewardPanel([])
    for i in range(12000):p.record(dict(reward_stage=0,contribution=[0.,-1. if i==7777 else 0.,0.]))
    p.show_full_history=True
    x,y=p.chart_samples('table_clearance')
    assert len(y)<=5002 and y.min()==-1. and x[-1]==11999


def test_coordinated_pose_apply_and_stop_clears_pending_and_gripper_hold():
    p=RewardPanel(['joint']*6)
    p.joints=[.1,.2,.3,.4,.5,.6]
    p.draw(UI(clicked=['Copy current pose']))
    assert p.pose_draft==p.joints
    p.pose_draft=[.2,.3,.4,.5,.6,.7]
    p.draw(UI(clicked=['Apply all joint targets','Hold current gripper target']))
    assert p.pending_pose==p.pose_draft and p.hold_gripper
    p.draw(UI(clicked=['STOP / HOLD']))
    assert p.pending_pose is None and not p.hold_gripper and p.paused


def test_coordinated_pose_cannot_apply_when_safety_locked():
    p=RewardPanel(['joint']*6);p.pose_draft=[0.]*6;p.locked=True
    p.draw(UI(clicked=['Apply all joint targets']))
    assert p.pending_pose is None
