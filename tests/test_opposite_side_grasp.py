"""Current trainer: side progress must not pay for waiting, dragging or squeezing."""
import torch
from test_four_stage import values
from isaac_vision_rl.utils.four_stage import StageMachine, gated_open_jaw_quality


def sample(**changes):
    defaults=dict(reach_ready=False,static_finger_reach=.3,moving_finger_reach=.3,
                  fixed_finger_distance=.04,moving_finger_distance=.04,on_table=True)
    defaults.update(changes)
    return values(**defaults)


def test_independent_signed_side_progress_and_stationary_baseline():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=1
    s.update(sample(),1/30)
    assert not s.group_raw[0,1,:2].any()
    s.update(sample(fixed_finger_distance=.03,moving_finger_distance=.05),1/30)
    assert s.group_raw[0,1,0]>0 and s.group_raw[0,1,1]<0
    s.update(sample(fixed_finger_distance=.03,moving_finger_distance=.05),1/30)
    assert not s.group_raw[0,1,:2].any()


def test_dragging_cannot_earn_positive_finger_progress():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous');s.stage[:]=1
    s.update(sample(),1/30)
    s.update(sample(fixed_finger_distance=.02,moving_finger_distance=.06,slide_speed=.02),1/30)
    assert s.group_raw[0,1,0]==0 and s.group_raw[0,1,1]<0
    assert s.group_contribution[0,1,4]<0


def test_real_grasp_and_reset_do_not_create_squeezing_income():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous');s.stage[:]=1
    s.update(sample(),1/30)
    s.update(sample(fixed_finger_distance=.001,moving_finger_distance=.001,
                    bilateral=True,enclosed=True,jaw=.8),1/30)
    assert not s.group_raw[0,1,:2].any()
    s.finger_sides_valid[:]=True;s.reset()
    assert not s.finger_sides_valid.any() and not s.finger_distance_valid.any()
    s.stage[:]=1;s.update(sample(fixed_finger_distance=.001),1/30)
    assert not s.group_raw[0,1,:2].any()


def test_opening_only_scores_in_reach_and_closing_scores_no_grasp_in_reach():
    q=gated_open_jaw_quality(torch.ones(1),torch.ones(1),torch.tensor([1.2]))
    assert q>gated_open_jaw_quality(torch.ones(1),torch.ones(1),torch.tensor([.5]))
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='progress')
    s.update(sample(jaws_open=1.,enclosure=1.),1/30)
    assert s.group_raw[0,0,3]>0 and not s.group_raw[0,1,:4].any()
    s.stage[:]=1;s.update(sample(jaws_open=1.),1/30)
    assert s.group_raw[0,0,3]==0


def test_separate_finger_graphs_hold_totals_without_motion():
    from isaac_vision_rl.utils.reward_panel import RewardPanel
    panel=RewardPanel([])
    for fixed,moving in ((.1,-.2),(0.,0.),(-.3,.1)):
        groups=[[0.]*5 for _ in range(4)]
        groups[1][0]=fixed;groups[1][1]=moving
        panel.record(dict(stage=1,reward_stage=1,group_contribution=groups))
    assert torch.allclose(torch.tensor(panel.term_history['static_finger_reach']),torch.tensor([.1,.1,-.2]))
    assert torch.allclose(torch.tensor(panel.term_history['moving_finger_reach']),torch.tensor([-.2,-.2,-.1]))


def test_valid_grasp_dwell_crosses_reach_handoff_without_restarting_timer():
    from isaac_vision_rl.utils.four_stage import GRASP_HOLD_SECONDS
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    dt=.05
    for _ in range(round(GRASP_HOLD_SECONDS/dt)):
        s.update(sample(bilateral=True,enclosed=True,jaw=.8,tool_cube_distance=.03),dt)
    assert s.stage.item()==2
    s.reset()
    assert s.acquisition_dwell.item()==0 and s.stage.item()==0


def test_overhead_or_one_finger_cannot_claim_grasp_success():
    for changes in ({'bilateral':False},{'bilateral':True,'enclosed':False}):
        s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
        for _ in range(25):s.update(sample(jaw=.8,**changes),.05)
        assert s.stage.item()==0


def test_reach_gate_requires_cube_between_open_fingers_not_just_overhead():
    from isaac_vision_rl.utils.four_stage import pinch_reach_ready
    def ready(between=True,distance=.015,down=1.,clearance=.01,jaw=1.2):
        return pinch_reach_ready(*[torch.tensor([x]) for x in (between,distance,down,clearance,jaw)]).item()
    assert ready()
    assert not ready(between=False)
    assert not ready(distance=.06)
    assert not ready(down=.2)
    assert not ready(clearance=.002)
    assert not ready(jaw=.7)


def test_live_snapshot_follows_stage_and_reset_without_waiting_for_next_tick():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from isaac_vision_rl.utils.reward_panel import RewardPanel
    from isaac_vision_rl.utils.training_reward_charts import update_training_charts,reset_training_charts
    panel=RewardPanel([]);panel.last_episode_step=-1;panel.reward_window=Mock()
    env=NS(_training_reward_panel=panel,common_step_counter=45,episode_length_buf=torch.tensor([17]))
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=3;s.reward_stage[:]=2
    update_training_charts(env,s)
    assert panel.snapshot['stage']==3 and panel.snapshot['reward_stage']==2
    reset_training_charts(env,[1])
    assert panel.snapshot['stage']==3
    reset_training_charts(env,[0])
    assert panel.snapshot['stage']==0 and panel.snapshot['episode_step']==0
    assert panel.snapshot['step_reward']==0 and panel.total==0
    assert panel.reward_window.send.call_args.args[0]['stage']==0
    count=panel.sample_count
    reset_training_charts(env,[0])
    assert panel.sample_count==count  # reward-manager slots cannot duplicate reset samples


def test_default_ten_to_twelve_cm_held_lift_passes_but_low_high_or_unheld_do_not():
    from isaac_vision_rl.utils.four_stage import LIFT_RISE_MIN_M,LIFT_RISE_MAX_M
    assert (LIFT_RISE_MIN_M,LIFT_RISE_MAX_M)==(.10,.12)
    for rise,bilateral,expected in ((.09,True,2),(.11,True,3),(.13,True,2),(.11,False,2)):
        s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous',lift_hold_seconds=.5)
        s.stage[:]=2
        for _ in range(11):
            s.update(sample(height=.425+rise,bilateral=bilateral,enclosed=True,jaw=.8,
                            tool_cube_distance=.03,speed=.05),.05)
        assert s.stage.item()==expected


def test_newton_one_visible_robot_is_environment_zero_not_random_world():
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    from scripts.train import newton_visualizer_overrides
    from isaaclab.visualizers.base_visualizer import BaseVisualizer
    options=newton_visualizer_overrides(1)
    assert 'env.sim.default_visualizer_cfg.visible_env_indices=[0]' in options
    assert 'env.sim.default_visualizer_cfg.randomly_sample_visible_envs=false' in options
    viewer=NS(_scene_data_provider=NS(num_envs=16),
              cfg=NS(visible_env_indices=[0],max_visible_envs=1,randomly_sample_visible_envs=False))
    with patch('random.sample',side_effect=AssertionError('Wrong environment could be displayed')):
        assert BaseVisualizer._compute_visualized_env_ids(viewer)==[0]


def test_grasp_diagnostics_identify_failed_checks_without_advancing_stage():
    from isaac_vision_rl.utils.training_reward_charts import grasp_check_snapshot
    m=values(fixed_contact_in_pad=True,moving_contact_in_pad=False,
             fixed_contact_force_N=.2,moving_contact_force_N=.005,
             opposing_contact_forces=False,enclosed=True,jaw=.83,unsafe=False)
    report=grasp_check_snapshot(m,.0,.5)
    assert report['fixed_pad'] and not report['moving_pad'] and not report['opposing']
    assert report['jaw_ready'] and report['enclosed'] and report['safe']
    assert report['held_seconds']==0 and report['required_seconds']==.5


def test_pushing_ungrasped_cube_cannot_earn_other_positive_task_rewards():
    def aligned_sample(**changes):
        return sample(grasp_target_xy_distance=.005,downward_cosine=.99,**changes)
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=1;s.descent_ready[:]=True
    s.update(aligned_sample(grasp_pose_distance=.10,enclosure=.1),1/30)
    s.update(aligned_sample(grasp_pose_distance=.08,enclosure=.4,fixed_finger_distance=.02,
                    moving_finger_distance=.02,slide_speed=.02),1/30)
    assert not (s.group_contribution>0).any()
    assert s.group_contribution[0,1,4]<0
    s.update(aligned_sample(grasp_pose_distance=.12,enclosure=.1,slide_speed=.02),1/30)
    assert s.group_contribution[0,0,0]<0  # Retreat remains negative.
    s.update(aligned_sample(grasp_pose_distance=.12,enclosure=.1),1/30)
    assert s.group_contribution[0,0,0]==0  # Stopping does not pay back pushed progress.


def test_verified_airborne_carry_still_earns_box_reward():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=3
    s.update(sample(bilateral=True,enclosed=True,jaw=.8,tool_cube_distance=.03,
                    height=.535,on_table=False,box_distance=.10),1/30)
    assert s.group_contribution[0,3,0]>0


def test_descent_requires_small_target_xy_error_and_nearly_downward_wrist():
    from isaac_vision_rl.utils.four_stage import descent_alignment_ready
    xy=torch.tensor([.008,.03,.008]);down=torch.tensor([.99,.99,.91])
    assert descent_alignment_ready(xy,down).tolist()==[True,False,True]


def test_losing_alignment_changes_target_without_fake_progress():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.stage[:]=1;s.descent_ready[:]=True
    s.update(sample(downward_cosine=.99,grasp_target_xy_distance=.005),1/30)
    assert s.using_descent_target.item()
    s.update(sample(downward_cosine=.75,grasp_target_xy_distance=.05),1/30)
    assert not s.using_descent_target.item()
    assert s.group_raw[0,0,0]==0


def test_all_five_reward_columns_reach_ppo_including_grasp_and_dragging(monkeypatch):
    from types import SimpleNamespace as NS
    from isaac_vision_rl.utils import four_stage_terms
    # Distinct signed columns detect omissions, duplicates and lost penalties.
    state=NS(contribution=torch.tensor([[.1,-.2,.3,.4,-.5],[0.,0.,0.,.7,-.9]]))
    monkeypatch.setattr(four_stage_terms,'tick',lambda env:state)
    env=NS(step_dt=1/30)
    sent=sum(four_stage_terms.StageReward.__call__(None,env,slot)*env.step_dt for slot in range(3))
    assert torch.allclose(sent,state.contribution.sum(-1))
    assert sent[1]<0  # Grasp and dragging must both survive the adapter.
