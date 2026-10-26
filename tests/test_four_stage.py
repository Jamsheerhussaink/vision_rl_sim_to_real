import torch
from isaac_vision_rl.utils.four_stage import StageMachine, TERMS, WEIGHTS


def values(**changes):
    d=dict(bilateral=False,enclosed=True,jaw=1.2,height=.425,cube_start_height=.425,speed=0.,unsafe=False,
           reach_ready=True,pregrasp_distance=.01,alignment=1.,enclosure=0.,clearance=.01,
           slide_speed=0.,box_distance=.15,lowering=0.,inside_box=False,placed=False,
           tool_height_above_cube=.02,xy_distance=.20,grasp_pose_distance=.20,
           top_push=False,top_approach_push=False,side_push=False)
    d.update(changes)
    return {k:torch.tensor([v]) for k,v in d.items()}


def test_exact_budget_and_no_implicit_global_terms():
    assert all(len(t)<=3 for t in TERMS)
    assert all(len(w)==3 for w in WEIGHTS)
    assert TERMS[0]==('overhead_approach','table_clearance','gripper_readiness')
    assert len(TERMS[0])==3 and WEIGHTS[0][2]>0.


def test_pregrasp_waypoint_is_above_cube_and_alignment_rewards_open_jaws():
    from isaac_vision_rl.utils.four_stage import (
        pregrasp_target,pregrasp_alignment,approach_from_above_quality,
    )
    cube=torch.tensor([[.2,-.1,.43]])
    target=pregrasp_target(cube)
    assert torch.allclose(target,torch.tensor([[.2,-.1,.51]]))
    xy=torch.tensor([0.]);down=torch.tensor([1.])
    overhead=approach_from_above_quality(xy,torch.tensor([.08]))
    side=approach_from_above_quality(torch.tensor([.10]),torch.tensor([.02]))
    assert overhead>side
    assert side.item()==0.
    opened=pregrasp_alignment(xy,down,torch.tensor([1.10]),overhead)
    closed=pregrasp_alignment(xy,down,torch.tensor([0.]),overhead)
    assert opened>closed
    assert torch.allclose(opened,torch.tensor([1.]))


def test_all_stages_require_dwell_then_single_success_bonus():
    s=StageMachine(1,'cpu')
    for _ in range(3): s.update(values(),.05)
    assert s.stage.item()==1
    m=values(jaw=.97,bilateral=True,enclosure=1.)
    for _ in range(4): s.update(m,.05)
    assert s.stage.item()==2
    for _ in range(11): s.update(values(jaw=.97,bilateral=True,height=.535,enclosure=1.),.05)
    assert s.stage.item()==3 and not s.success.item()
    for _ in range(9): s.update(values(placed=True,inside_box=True),.05)
    assert s.success.item()
    assert s.update(values(placed=True,inside_box=True),.05).item()==0
    s.reset(); assert s.stage.item()==0 and not s.success.item()


def test_verified_lift_remains_valid_after_cube_is_lowered_for_placement():
    s=StageMachine(1,'cpu',reward_style='continuous',lift_hold_seconds=.5)
    s.stage[:]=2
    s.cube_start_height[:]=.425
    s.cube_start_height_valid[:]=True
    lifted=values(jaw=.97,bilateral=True,height=.535,enclosure=1.)
    for _ in range(11):s.update(lifted,.05)
    assert s.stage.item()==3 and s.lift_verified.item()
    lowered=values(jaw=1.2,bilateral=False,height=.43,placed=True,inside_box=True)
    for _ in range(8):s.update(lowered,.05)
    assert s.success.item()
    s.reset()
    assert not s.lift_verified.item()


def test_top_push_and_dragging_are_penalties_not_reward():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.stage[:]=1
    s.update(values(top_push=True),.05)
    assert s.group_raw[0,1,4] > 0
    assert s.group_contribution[0,1,4] < 0
    s.reset();s.stage[:]=1
    s.update(values(slide_speed=.05),.05)
    assert s.group_contribution[0,1,4] < 0


def test_downward_top_push_requires_closing_jaws_and_real_tool_descent():
    from isaac_vision_rl.utils.four_stage import top_descent_push
    result=top_descent_push(
        torch.tensor([.05,.05,.05]),
        torch.tensor([.04,.00,.04]),
        torch.tensor([.04,.04,.12]),
        torch.tensor([.80,.80,.80]),
    )
    assert result.tolist()==[True,False,False]


def test_clearance_shaping_encourages_safe_raise_but_allows_aligned_descent():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.stage[:]=0
    s.update(values(clearance=.005),.05)
    assert s.group_contribution[0,0,1] < 0
    s.descent_ready[:]=True
    s.update(values(clearance=.005),.05)
    assert s.group_contribution[0,0,1] == 0


def test_camera_margin_is_a_separate_reach_only_penalty():
    assert TERMS[0][-1] == 'camera_visibility'
    assert WEIGHTS[0][-1] < 0.
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.update(values(camera_visibility_cost=.5),.05)
    reach_cost=s.group_contribution[0,0,-1].item()
    assert reach_cost < 0.
    s.stage[:]=1
    s.update(values(camera_visibility_cost=1.),.05)
    assert s.group_contribution[0,1,-1].item() == 0.


def test_one_finger_and_broken_contact_do_not_grasp_or_hold():
    s=StageMachine(1,'cpu');s.stage[:]=1
    for _ in range(20): s.update(values(jaw=.97,enclosure=1.),.05)
    assert s.stage.item()==1
    s.stage[:]=2
    for _ in range(5): s.update(values(jaw=.97,height=.52),.05)
    assert s.failure.item() and not s.success.item()


def test_unsafe_table_and_dropped_cube_cannot_advance():
    s=StageMachine(1,'cpu')
    for _ in range(20):s.update(values(unsafe=True),.05)
    assert s.failure.item() and s.stage.item()==0


def test_reach_rewards_raise_first_then_3d_pregrasp_and_penalizes_overshoot():
    from isaac_vision_rl.utils.four_stage import approach_from_above_quality
    s=StageMachine(1,'cpu')
    # While low, moving sideways earns nothing; raising toward the 8 cm
    # clearance earns progress.
    s.update(values(reach_ready=False,tool_height_above_cube=.01,pregrasp_distance=.15),.05)
    sideways=s.update(values(reach_ready=False,tool_height_above_cube=.01,xy_distance=.10,pregrasp_distance=.15),.05).item()
    assert sideways==0
    raised=s.update(values(reach_ready=False,tool_height_above_cube=.08,
                           pregrasp_distance=.12),.05).item()
    assert raised>0
    s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.12),.05)
    switched=s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.12),.05)
    assert torch.isfinite(switched).all()
    closer=s.update(values(reach_ready=False,tool_height_above_cube=.08,
                           pregrasp_distance=.05),.05).item()
    assert closer>0
    farther=s.update(values(reach_ready=False,tool_height_above_cube=.08,
                            pregrasp_distance=.12),.05).item()
    assert farther<0
    assert s.update(values(reach_ready=False,tool_height_above_cube=.08,
                           pregrasp_distance=.12),.05).item()==0
    # Losing overhead clearance while misaligned restarts the raise objective
    # and suppresses sideways reward until the gripper rises again.
    s.update(values(reach_ready=False,tool_height_above_cube=.02,xy_distance=.10),.05)
    assert s.reach_route.item()==0
    assert approach_from_above_quality(torch.tensor([.10]),torch.tensor([.02])).item()==0.


def test_continuous_reach_reward_is_signed_distance_change_and_pose_independent():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.reach_route[:]=1  # test the overhead-centering subgoal
    good=dict(approach_from_above=.95,downward_cosine=1.,jaw=1.1,
              tool_height_above_cube=.08,reach_ready=False)
    s.update(values(pregrasp_distance=.05,**good),.05)
    initial=s.group_contribution[0,0,0].item()
    readiness_good=s.group_contribution[0,0,2].item()
    s.update(values(pregrasp_distance=.05,**good),.05)
    same=s.group_contribution[0,0,0].item()
    s.update(values(pregrasp_distance=.20,**good),.05)
    farther=s.group_contribution[0,0,0].item()
    s.update(values(
        pregrasp_distance=.05,approach_from_above=.05,
        downward_cosine=-.5,jaw=0.,tool_height_above_cube=.08,
        reach_ready=False),.05)
    same_distance_different_pose=s.group_contribution[0,0,0].item()
    readiness_bad=s.group_contribution[0,0,2].item()
    s.update(values(
        pregrasp_distance=.05,approach_from_above=.05,
        downward_cosine=-.5,jaw=0.,tool_height_above_cube=.08,
        reach_ready=False),.05)
    stationary_bad_pose=s.group_contribution[0,0,0].item()
    assert initial==0
    assert same==0  # waiting produces no approach reward
    assert farther<0  # backing away produces negative reward
    assert same_distance_different_pose>0  # distance change counts regardless of pose
    assert stationary_bad_pose==0  # pose alone gives no distance reward
    assert readiness_good>readiness_bad  # independent centered/downward alignment


def test_open_jaw_reward_requires_downward_wrist_and_cube_overhead_alignment():
    from isaac_vision_rl.utils.four_stage import gated_open_jaw_quality
    jaw=torch.tensor([1.10])
    sideways=gated_open_jaw_quality(torch.tensor([0.]),torch.tensor([1.]),jaw)
    down_but_misaligned=gated_open_jaw_quality(torch.tensor([1.]),torch.tensor([0.]),jaw)
    correctly_aligned=gated_open_jaw_quality(torch.tensor([1.]),torch.tensor([1.]),jaw)
    below_angle_threshold=gated_open_jaw_quality(torch.tensor([.79]),torch.tensor([1.]),jaw)
    assert sideways.item()==0.
    assert down_but_misaligned.item()==0.
    assert below_angle_threshold.item()==0.
    assert torch.isclose(correctly_aligned,torch.tensor([1.]))


def test_continuous_distance_progress_tracks_raise_then_overhead_route():
    s=StageMachine(1,'cpu',reward_style='continuous')
    # First waypoint: reward only movement toward the safe-height target.
    s.update(values(reach_ready=False,tool_height_above_cube=0.),.05)
    initial=s.group_contribution[0,0,0].item()
    s.update(values(reach_ready=False,tool_height_above_cube=.02),.05)
    raised=s.group_contribution[0,0,0].item()
    assert initial==0 and raised>0
    # Reaching overhead changes the target without a fake reward jump.
    s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.10),.05)
    s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.10),.05)
    assert s.reach_route.item()==1
    assert s.group_contribution[0,0,0].item()==0
    s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.05,
                    approach_from_above=.8),.05)
    above=s.group_contribution[0,0,0].item()
    s.update(values(reach_ready=False,tool_height_above_cube=.08,pregrasp_distance=.10,
                    approach_from_above=0.),.05)
    retreat=s.group_contribution[0,0,0].item()
    assert above>0 and retreat<0  # alignment is separate from distance progress.


def test_reach_distance_readiness_and_table_safety_have_separate_terms():
    s=StageMachine(1,'cpu',reward_mode='overlap')
    s.update(values(reach_ready=False,tool_height_above_cube=.08,
                    pregrasp_distance=.15,alignment=.2),.05)
    s.update(values(reach_ready=False,tool_height_above_cube=.08,
                    pregrasp_distance=.15,alignment=.2),.05)  # switch subgoals
    s.update(values(reach_ready=False,tool_height_above_cube=.08,
                    pregrasp_distance=.10,alignment=.6,
                    clearance=.006),.05)
    # Distance and pose feedback now occupy separate chart/reward slots.
    assert torch.allclose(s.group_contribution[0,0,0],torch.tensor(.5))
    assert s.group_contribution[0,0,2]==0.
    assert s.group_contribution[0,0,1]<0  # Clearance stays its own penalty line.


def test_grasp_current_quality_persists_and_drag_is_negative():
    s=StageMachine(1,'cpu',reward_mode='overlap');s.stage[:]=1
    m=values(reach_ready=False,enclosure=.5)
    s.update(m,.05)
    before=s.group_contribution[:,1].clone()
    s.update(m,.05)
    assert torch.equal(before,s.group_contribution[:,1])
    assert before[0,0]>0
    s.update(values(reach_ready=False,slide_speed=.1),.05)
    assert s.group_contribution[0,1,2]<0


def test_overlapping_activation_and_aggregated_reward_for_each_stage():
    from isaac_vision_rl.utils.four_stage import active_reward_groups
    for stage,expected in enumerate(((0,1), (0,1), (1,2), (1,3))):
        s=StageMachine(1,'cpu');s.stage[:]=stage
        total=s.update(values(reach_ready=False,jaw=.97,bilateral=True,
                              enclosure=.8,height=.48,speed=.1),.05)
        assert active_reward_groups(stage)==expected
        for group in range(4):
            if group not in expected:assert not s.group_contribution[:,group].any()
        if stage in (0,1,2):assert s.group_contribution[0,1,1]>0
        elif stage==3:assert s.group_contribution[0,1,:2].sum()==0
        else:
            # First observation establishes the progress baseline. Moving closer
            # on the following observation must then earn positive reach reward.
            assert s.group_contribution[0,0,0]==0
            total=s.update(values(reach_ready=False,tool_height_above_cube=.08,
                                  xy_distance=.0,pregrasp_distance=.0),.05)
            assert s.group_contribution[0,0,0]>0
        assert torch.allclose(total,s.group_contribution.sum((1,2)))
        assert torch.allclose(s.contribution,s.group_contribution.sum(1))
    s=StageMachine(1,'cpu');s.stage[:]=3
    s.update(values(inside_box=True,jaw=1.2,enclosure=0.),.05)
    assert s.group_contribution[0,1,:2].sum()==0  # Release earns no grasp bonus.


def test_dense_reach_return_is_control_rate_independent():
    totals=[]
    for fps in (30,60):
        s=StageMachine(1,'cpu')
        rewards=[]
        for step in range(fps*2):
            height=.02+.06*(step/(fps*2-1))
            rewards.append(s.update(values(reach_ready=False,tool_height_above_cube=height),1/fps).item())
        totals.append(sum(rewards))
    # The 0.1 s raise dwell is sampled at 30/60 Hz, so one control tick of
    # quantization is expected when the route switches subgoals.
    assert abs(totals[0]-totals[1])<.01


def test_invalid_release_and_lost_contact_fail_but_valid_release_can_settle():
    s=StageMachine(1,'cpu');s.stage[:]=3
    for _ in range(5):s.update(values(placed=False),.05)
    assert s.failure.item() and not s.success.item()
    s.reset();s.stage[:]=3
    for _ in range(9):s.update(values(placed=True,inside_box=True),.05)
    assert s.success.item()


def test_profile_config_has_only_three_terms_and_no_lift_termination():
    import os
    from unittest.mock import patch
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    with patch.dict(os.environ,{'SO101_REWARD_PROFILE':'four-stage','SO101_CURRICULUM':'place','SO101_POLICY':'rgb'}):
        cfg=PickplaceEnvCfg()
    assert set(vars(cfg.rewards))=={'slot_1','slot_2','slot_3'}
    assert not hasattr(cfg.terminations,'lift_success')
    assert cfg.terminations.stage_failure is not None
    assert cfg.terminations.unsafe_table_contact is not None
    assert cfg.events.reset_cube_pose.params['pose_range']=={}
    assert cfg.observations.policy.phase is not None


def test_four_stage_uses_smooth_arm_wrist_and_gripper_references():
    import os
    from unittest.mock import patch
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    with patch.dict(os.environ,{'SO101_REWARD_PROFILE':'four-stage','SO101_CURRICULUM':'place','SO101_POLICY':'rgb'}):
        cfg=PickplaceEnvCfg()
    assert cfg.actions.arm_action.max_target_speed==.5
    assert cfg.actions.arm_action.max_target_acceleration==1.
    assert cfg.actions.arm_action.wrist_roll_speed==.2
    assert cfg.scene.robot.init_state.joint_pos["gripper"]==0.0
    assert cfg.events.reset_robot_joints.params["position_range"]==(0.0,0.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"]==(0.0,0.0)
    assert cfg.actions.gripper_action.max_target_speed==.45
    assert cfg.actions.gripper_action.max_target_acceleration==.9


def test_direct_grasp_completes_reach_without_open_pregrasp():
    for manual in (False,True):
        s=StageMachine(1,'cpu')
        m=values(reach_ready=False,jaw=.97,bilateral=True,enclosure=1.)
        s.update(m,.05,inspection=manual)
        assert s.stage.item()==0  # Single contact tick is not sufficient.
        for _ in range(2):s.update(m,.05,inspection=manual)
        assert s.stage.item()==1  # Display stage 2: GRASP, not skipped to LIFT.
        for _ in range(4):s.update(m,.05,inspection=manual)
        assert s.stage.item()==2


def test_direct_grasp_rejects_one_finger_bad_enclosure_open_jaw_and_unsafe():
    for changes in (dict(bilateral=False),dict(enclosed=False),dict(jaw=1.2),dict(unsafe=True)):
        s=StageMachine(1,'cpu')
        params=dict(reach_ready=False,jaw=.97,bilateral=True,enclosure=1.)
        params.update(changes)
        for _ in range(8):s.update(values(**params),.05,inspection=True)
        assert s.stage.item()==0


def test_direct_grasp_requires_continuous_dwell():
    s=StageMachine(1,'cpu')
    grasp=values(reach_ready=False,jaw=.97,bilateral=True)
    for _ in range(8):
        s.update(grasp,.05)
        s.update(values(reach_ready=False),.05)
    assert s.stage.item()==0


def test_bounded_grasp_cannot_farm_holding_or_regrasping():
    s=StageMachine(1,'cpu',reward_mode='bounded');s.stage[:]=1
    grasp=values(reach_ready=False,jaw=.97,bilateral=True,enclosure=1.)
    earned=0.
    for _ in range(200):
        s.update(grasp,.05,inspection=True)
        earned+=float(s.group_contribution[0,1,:2].sum())
    assert abs(earned-12.)<1e-5
    for _ in range(6):s.update(values(reach_ready=False),.05,inspection=True)
    for _ in range(20):
        s.update(grasp,.05,inspection=True)
        assert s.group_contribution[0,1,:2].sum()==0
    s.reset();s.stage[:]=1;s.update(grasp,.05)
    assert s.group_contribution[0,1,:2].sum()>0


def test_lift_only_goal_finishes_on_held_lift_and_never_enters_place():
    s=StageMachine(1,'cpu',reward_mode='bounded',goal='lift');s.stage[:]=2
    m=values(jaw=.97,bilateral=True,enclosure=1.,height=.575)
    for _ in range(11):s.update(m,.05)
    assert s.success.item() and s.stage.item()==2
    assert s.update(m,.05)==0


def test_finger_contact_capacity_above_observed_truncation_limit():
    import os
    from unittest.mock import patch
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    with patch.dict(os.environ,{'SO101_REWARD_PROFILE':'four-stage','SO101_CURRICULUM':'place','SO101_POLICY':'rgb'}):
        cfg=PickplaceEnvCfg()
    for name in ('fixed_finger_contact','moving_finger_contact'):
        assert getattr(cfg.scene,name).max_contact_data_count_per_prim==1024


def test_bounded_critic_sees_remaining_reward_history_and_partial_reset():
    from types import SimpleNamespace as NS
    from isaac_vision_rl.utils.four_stage import critic_observation
    s=StageMachine(2,'cpu',reward_mode='bounded')
    s.grasp_best[:]=.5;s.contact_dwell[:]=.1
    robot=NS(find_bodies=lambda _:([0],None),data=NS(body_pos_w=NS(torch=torch.zeros(2,1,3))))
    cube=NS(data=NS(root_pos_w=NS(torch=torch.zeros(2,3)),root_lin_vel_w=NS(torch=torch.zeros(2,3))))
    box=NS(data=NS(root_pos_w=NS(torch=torch.zeros(2,3))))
    class Scene(dict):pass
    scene=Scene(robot=robot,cube=cube,box=box);scene.env_origins=torch.zeros(2,3)
    env=NS(_four_stage=s,scene=scene)
    obs=critic_observation(env)
    assert obs.shape==(2,23)
    assert torch.allclose(obs[:,-3:],torch.tensor([[.5,.5,.1],[.5,.5,.1]]))
    s.reset(torch.tensor([0]));assert s.grasp_best[0].sum()==0 and s.grasp_best[1].sum()==1
    s.reward_mode='overlap';assert critic_observation(env).shape==(2,20)


def test_lift_height_reward_peaks_in_band_and_falls_above_it():
    from isaac_vision_rl.utils.four_stage import lift_height_quality
    h=torch.tensor([0.,.075,.10,.12,.13,.15,.20])
    q=lift_height_quality(h)
    assert q[0]==0 and 0<q[1]<q[2]
    assert torch.allclose(q[2:4],torch.ones(2))
    assert q[4]<q[3] and q[5]<q[4] and q[6]<q[5]
    s=StageMachine(1,'cpu',reward_mode='bounded');s.stage[:]=2
    for _ in range(15):s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.675),.05)
    assert s.stage.item()==2 and not s.success.item()


def test_over_lifting_is_negative_even_after_maximum_lift_credit():
    s=StageMachine(1,'cpu',reward_mode='bounded');s.stage[:]=2
    s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.425),.05)
    s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.535),.05)
    assert s.group_contribution[0,2,0]>0
    s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.560),.05)
    s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.575),.05)
    assert s.group_contribution[0,2,0]<0 and s.dwell.item()==0
    s.stage[:]=3;s.started[:]=False
    s.update(values(jaw=.97,bilateral=True,enclosure=1.,height=.635),.05)
    assert s.group_contribution[0,3,1]<0


def test_old_overlap_checkpoint_uses_requested_lift_rise_threshold():
    s=StageMachine(1,'cpu',reward_mode='overlap');s.stage[:]=2
    for _ in range(11):s.update(values(jaw=.97,bilateral=True,height=.51),.05)
    assert s.stage.item()==2
    for _ in range(11):s.update(values(jaw=.97,bilateral=True,height=.575),.05)
    assert s.stage.item()==3


def test_placement_allows_lowering_below_carry_height_and_success():
    s=StageMachine(1,'cpu',reward_mode='bounded',goal='place');s.stage[:]=3
    for _ in range(9):s.update(values(placed=True,inside_box=True,height=.43,lowering=1.),.05)
    assert s.success.item()  # Carry band must not prevent release onto box floor.


def test_slow_opening_inside_box_is_permitted_but_does_not_create_success():
    s=StageMachine(1,'cpu',reward_mode='bounded');s.stage[:]=3
    for _ in range(20):
        s.update(values(jaw=.85,bilateral=False,inside_box=True,opening_for_release=True),.05)
    assert not s.failure.item() and not s.success.item()
    outside=StageMachine(1,'cpu',reward_mode='bounded');outside.stage[:]=3
    for _ in range(6):outside.update(values(jaw=.85,inside_box=False,opening_for_release=True),.05)
    assert outside.failure.item()
    stopped=StageMachine(1,'cpu',reward_mode='bounded');stopped.stage[:]=3
    for _ in range(6):stopped.update(values(jaw=.85,inside_box=True,opening_for_release=False),.05)
    assert stopped.failure.item()


def test_dragging_proxy_excludes_box_floor_but_keeps_source_table_motion():
    from isaac_vision_rl.utils.four_stage import table_slide_mask
    height=torch.tensor([.425,.430,.54,.425,.430])
    inside_xy=torch.tensor([False,True,False,True,True])
    bottom=torch.tensor([.004,.004,.10,-.01,.05])
    # Source table / actual box interior / airborne / under floor / outside floor zone.
    assert table_slide_mask(height,inside_xy,bottom).tolist()==[True,False,False,True,True]
