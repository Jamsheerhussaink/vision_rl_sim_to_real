"""Isolated experimental reward behavior; no simulator or active-run changes."""
import hashlib
import json
from pathlib import Path
import torch
from test_four_stage import values
from isaac_vision_rl.experiments.simple_rewards.stages import (
    StageMachine, TERMS, active_reward_groups, GRIPPER_ACTION_CLASS,
    INITIAL_GRIPPER_POSITION,
)


def sample(**kwargs):
    m=values(reach_ready=False,pinch_center_distance=.04,pinch_overhead_distance=.10,
             downward_cosine=.5,jaw_gap_score=-.4,closing_allowed=True,
             fixed_finger_distance=.04,moving_finger_distance=.04)
    m.update({k:torch.tensor([v]) for k,v in kwargs.items()})
    return m


def test_original_sources_are_untouched():
    base=Path(__file__).resolve().parents[1]
    manifest=json.loads((base/'src/isaac_vision_rl/experiments/simple_rewards/source_manifest.json').read_text())
    for path,digest in manifest.items():
        assert hashlib.sha256((base/path).read_bytes()).hexdigest()==digest


def test_three_separate_terms_and_only_current_stage():
    assert all(len(t)==3 for t in TERMS)
    assert TERMS[0]==('reach_distance','point_down','open_jaws')
    for i in range(4):assert active_reward_groups(i)==(i,)
    s=StageMachine(1,'cpu',reward_style='continuous')
    for stage in range(4):
        s.reset();s.stage[:]=stage
        s.update(sample(),1/30,inspection=True)
        assert not s.group_contribution[0,[i for i in range(4) if i!=stage]].any()


def test_distance_and_angle_change_independently_and_waiting_pays_nothing():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.update(sample(),1/30)
    s.update(sample(pinch_overhead_distance=.08),1/30)
    assert s.group_contribution[0,0,0]>0
    assert s.group_contribution[0,0,1]==0
    s.update(sample(pinch_overhead_distance=.08),1/30)
    assert not s.group_contribution.any()
    s.update(sample(pinch_overhead_distance=.12),1/30)
    assert s.group_contribution[0,0,0]<0
    s.update(sample(pinch_overhead_distance=.12,downward_cosine=.8),1/30)
    assert s.group_contribution[0,0,0]==0
    assert s.group_contribution[0,0,1]>0


def test_open_jaw_reward_only_grows_when_centered_above_cube():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.update(sample(approach_from_above=0.,jaw=0.),1/30)
    s.update(sample(approach_from_above=0.,jaw=1.1),1/30)
    assert s.group_contribution[0,0,2]==0
    s.reset()
    s.update(sample(approach_from_above=1.,downward_cosine=.95,jaw=0.),1/30)
    s.update(sample(approach_from_above=1.,downward_cosine=.95,jaw=1.1),1/30)
    assert s.group_contribution[0,0,2]>0
    s.update(sample(approach_from_above=1.,downward_cosine=.95,jaw=1.1),1/30)
    assert s.group_contribution[0,0,2]==0


def test_fresh_policy_controls_jaw_without_stage_based_override():
    assert INITIAL_GRIPPER_POSITION in (1.2, 1.55)
    assert GRIPPER_ACTION_CLASS.endswith(':PolicyGripperAction')
    assert 'GuidedGripperAction' not in GRIPPER_ACTION_CLASS


def test_each_finger_earns_independent_signed_progress_without_waiting_income():
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=1
    s.update(sample(),1/30)
    assert not s.group_contribution.any()
    s.update(sample(fixed_finger_distance=.03),1/30)
    assert s.group_contribution[0,1,0]>0 and s.group_contribution[0,1,1]==0
    s.update(sample(fixed_finger_distance=.03,moving_finger_distance=.06),1/30)
    assert s.group_contribution[0,1,0]==0 and s.group_contribution[0,1,1]<0
    s.update(sample(fixed_finger_distance=.03,moving_finger_distance=.06),1/30)
    assert not s.group_contribution.any()


def test_unsafe_pose_cannot_earn_positive_reward_and_touch_does_not_advance():
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=1
    s.update(sample(),1/30)
    s.update(sample(pinch_center_distance=.01,unsafe=True),1/30)
    assert (s.group_contribution<=0).all()
    s.reset();s.stage[:]=1
    for _ in range(10):s.update(sample(bilateral=False,enclosed=True,jaw=.8),1/30)
    assert s.stage.item()==1


def test_outcome_paid_once_and_reset_clears_baselines():
    s=StageMachine(1,'cpu',reward_style='continuous')
    s.stage[:]=3
    s.update(sample(timed_out=True,inside_box=True,opening_for_release=True),1/30)
    assert torch.isclose(s.group_contribution.sum(),torch.tensor(-.1))
    s.update(sample(timed_out=True,inside_box=True,opening_for_release=True),1/30)
    assert s.group_contribution.sum()==0
    s.reset();assert not s.simple_valid.any()


def test_experiment_panel_totals_are_independent():
    from isaac_vision_rl.experiments.simple_rewards.panel import RewardPanel
    p=RewardPanel([])
    groups=[[.2,.1,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
    p.record({'reward_stage':0,'group_contribution':groups})
    groups[0]=[0.,0.,0.]
    p.record({'reward_stage':0,'group_contribution':groups})
    assert p.term_history['reach_distance']==[.2,.2]
    assert p.term_history['point_down']==[.1,.1]
    p.clear_history()
    assert p.reach_distance_running_total==0


def test_module_activation_does_not_replace_default_in_parent_process():
    import subprocess,sys,os
    code='''import isaac_vision_rl.utils.four_stage as original
from isaac_vision_rl.experiments.simple_rewards import activate
activate()
from isaac_vision_rl.utils.four_stage import VERSION
assert VERSION.startswith("simple_rewards_")
assert original.VERSION != VERSION
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=os.environ.copy())


def test_charts_follow_current_stage_and_stage_two_has_reach_distance():
    from isaac_vision_rl.experiments.simple_rewards.panel import visible_reward_terms,display_name
    for stage in range(4):
        visible,terms=visible_reward_terms({'stage':stage,'reward_stage':max(0,stage-1)})
        assert visible==stage
        assert terms==TERMS[stage] and len(terms)==3
    _,terms=visible_reward_terms({'stage':1,'reward_stage':1})
    assert display_name(terms[0])=='Fixed Finger Near Cube Side'
    assert display_name(terms[1])=='Moving Finger Near Opposite Side'
    assert 'reach_distance' not in terms  # no stale overhead graph or fourth term


def test_contact_alignment_continues_but_extra_squeezing_does_not_pay():
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=1
    s.update(sample(fixed_finger_distance=.04,moving_finger_distance=.04),1/30)
    # First contact is not a finished 0.2-second grasp.
    s.update(sample(bilateral=True,enclosed=True,jaw=.8,
                    fixed_finger_distance=.02,moving_finger_distance=.02),1/30)
    assert (s.group_contribution[0,1,:2]>0).all()
    # Further closing after contact earns no positive credit.
    s.update(sample(bilateral=True,enclosed=True,jaw=.7,
                    fixed_finger_distance=.01,moving_finger_distance=.01),1/30)
    assert not s.group_contribution[0,1,:2].any()
    # With jaw held, better alignment can still be learned before the dwell passes.
    s.update(sample(bilateral=True,enclosed=True,jaw=.7,
                    fixed_finger_distance=.008,moving_finger_distance=.008),1/30)
    assert (s.group_contribution[0,1,:2]>0).all()
    for _ in range(5):
        s.update(sample(bilateral=True,enclosed=True,jaw=.7,
                        fixed_finger_distance=.008,moving_finger_distance=.008),1/30)
    assert s.stage.item()==2
    assert not s.group_contribution[0,1,:2].any()


def test_experimental_teacher_targets_relative_lift_not_old_absolute_height():
    import numpy as np
    from isaac_vision_rl.experiments.simple_rewards.teacher import IKTeacher,LIFT_RISE_MIN_M
    t=IKTeacher(None,None,np.zeros(6),np.array([0.,0.,.425]),np.array([0.,.1,.4]),1/30)
    assert abs(t.lift_target-(.425+LIFT_RISE_MIN_M+.01))<1e-9


def test_slow_lift_is_not_charged_the_fast_motion_penalty():
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=2
    for speed in (.0,.1,.25):
        s.update(sample(bilateral=True,enclosed=True,jaw=.8,speed=speed),1/30)
        assert s.group_contribution[0,2,2]==0
    s.update(sample(bilateral=True,enclosed=True,jaw=.8,speed=.4),1/30)
    assert s.group_contribution[0,2,2]<0


def test_lift_requires_grasped_cube_inside_height_band_for_full_hold():
    from isaac_vision_rl.experiments.simple_rewards import stages
    dt=1/30
    def machine():
        result=StageMachine(1,'cpu',reward_style='continuous',goal='place',lift_hold_seconds=.5)
        result.stage[:]=2
        result.cube_start_height[:]=.425
        result.cube_start_height_valid[:]=True
        return result

    valid=lambda rise: sample(bilateral=True,enclosed=True,jaw=.8,height=.425+rise,
                              cube_start_height=.425,speed=.1)
    below=machine()
    for _ in range(20):below.update(valid(stages.LIFT_RISE_MIN_M-.001),dt)
    assert below.stage.item()==2 and below.dwell.item()==0

    above=machine()
    for _ in range(20):above.update(valid(stages.LIFT_RISE_MAX_M+.001),dt)
    assert above.stage.item()==2 and above.dwell.item()==0

    short=machine()
    for _ in range(14):short.update(valid(.11),dt)
    assert short.stage.item()==2
    short.update(valid(.11),dt)
    assert short.stage.item()==3


def test_lift_hold_restarts_if_grasp_or_height_band_is_lost():
    from isaac_vision_rl.experiments.simple_rewards import stages
    s=StageMachine(1,'cpu',reward_style='continuous',goal='place',lift_hold_seconds=.5)
    s.stage[:]=2;s.cube_start_height[:]=.425;s.cube_start_height_valid[:]=True
    inside=sample(bilateral=True,enclosed=True,jaw=.8,height=.535,cube_start_height=.425,speed=.1)
    for _ in range(7):s.update(inside,1/30)
    assert s.dwell.item()>0
    outside=sample(bilateral=True,enclosed=True,jaw=.8,
                   height=.425+stages.LIFT_RISE_MAX_M+.001,cube_start_height=.425,speed=.1)
    s.update(outside,1/30)
    assert s.dwell.item()==0 and s.stage.item()==2
    for _ in range(14):s.update(inside,1/30)
    assert s.stage.item()==2
    s.update(inside,1/30)
    assert s.stage.item()==3


def test_lift_stage_still_requires_grasp_and_low_cube_speed():
    s=StageMachine(1,'cpu',reward_style='continuous',lift_hold_seconds=0.)
    s.stage[:]=2
    s.cube_start_height[:]=.425
    s.cube_start_height_valid[:]=True
    high=sample(bilateral=False,enclosed=False,jaw=1.2,height=.55,
                cube_start_height=.425,speed=.1)
    for _ in range(6):s.update(high,1/30)
    assert s.stage.item()==2
    unstable=sample(bilateral=True,enclosed=True,jaw=.8,height=.55,
                    cube_start_height=.425,speed=.3)
    for _ in range(6):s.update(unstable,1/30)
    assert s.stage.item()==2


def test_batch_evaluation_summary_reports_grasp_lift_slide_and_clearance_separately():
    from scripts.evaluate_simple_rewards_batch import summarize
    episodes=[
        {"seed":101,"success":True,"trace":[
            {"measurements":{"bilateral":True,"enclosed":True,"jaw":.8,
                             "lift_rise":.11,"slide_distance":.012,"clearance":.004}}]},
        {"seed":102,"success":False,"trace":[
            {"measurements":{"bilateral":True,"enclosed":False,"jaw":.8,
                             "lift_rise":.03,"slide_distance":.021,"clearance":.001}}]},
    ]
    report=summarize("lift","model.pt",[101,102],episodes)
    assert report["verified_two_finger_grasps"]==1
    assert report["held_lift_successes"]==1
    assert report["lift_gate_required_successes"]==2
    assert report["lift_gate_passed"] is False
    assert abs(report["mean_table_slide_cm"]-1.65)<1e-6
    assert report["minimum_clearance_mm"]==1.0


def test_teacher_records_blocked_ik_instead_of_crashing():
    import numpy as np
    from isaac_vision_rl.experiments.simple_rewards.teacher import IKTeacher
    class Geometry:
        def solve(self,*args,**kwargs):raise ValueError('test unreachable pose')
    t=IKTeacher(Geometry(),None,np.zeros(6),np.array([0.,0.,.425]),np.array([0.,.1,.4]),1/30)
    t.phase=4
    result=t.step(np.zeros(6),np.zeros(3),np.zeros(3),{'unsafe':False})
    assert 'IK plan blocked in lift' in t.failure
    assert np.array_equal(result,np.zeros(6))


def test_experiment_rejects_legacy_progress_mode():
    import pytest
    with pytest.raises(ValueError,match='requires continuous'):
        StageMachine(1,'cpu',reward_style='progress')


def patch():
    return torch.tensor([[[-.04,-.02,-.02],[-.04,.02,-.02],[-.04,.02,.02]],
                         [[-.04,-.02,-.02],[-.04,.02,.02],[-.04,-.02,.02]]])


def test_actual_surface_distance_not_wrist_or_patch_box_center():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import closest_surface_point
    d,q=closest_surface_point(torch.tensor([[-.025,0.,0.],[-.025,.05,.06]]),patch())
    assert torch.isclose(d[0],torch.tensor(.015),atol=1e-6)
    assert torch.allclose(q[0],torch.tensor([-.04,0.,0.]),atol=1e-6)
    assert torch.isclose(d[1],torch.sqrt(torch.tensor(.015**2+.03**2+.04**2)),atol=1e-6)


def test_opposite_cube_faces_follow_rotation_and_same_side_finger_is_not_correct():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import choose_opposite_sides,opposite_targets,closest_surface_point
    cube=torch.zeros(1,3);rot=torch.eye(3)[None]
    fixed=torch.tensor([[-.04,0.,0.]]);moving=-fixed
    axis,sign=choose_opposite_sides(fixed,moving,cube,rot)
    f,m=opposite_targets(cube,rot,axis,sign)
    assert torch.allclose(f,-m) and f[0,0]<0
    # Both fingers on the fixed side: moving finger is still 65 mm from its own target.
    wrong,_=closest_surface_point(m,patch())
    assert torch.isclose(wrong[0],torch.tensor(.065),atol=1e-6)
    turn=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    fr,mr=opposite_targets(cube,turn,axis,sign)
    assert torch.allclose(fr,torch.tensor([[0.,-.025,0.]]))
    assert torch.allclose(mr,-fr)


def test_above_behind_and_penetrating_surfaces_do_not_get_zero_error():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import distance_to_face
    cube=torch.zeros(1,3);rot=torch.eye(3)[None];target=torch.tensor([[-.025,0.,0.]])
    for shift in ([0.,0.,.1],[.08,0.,0.],[.03,0.,0.]):
        d,_,_=distance_to_face(target,torch.tensor([shift]),rot,patch(),cube,rot)
        assert d.item()>.005
    penetrating=torch.tensor([[[-.04,-.015,-.02],[-.01,.015,0.],[-.04,-.015,.02]]])
    d,_,depth=distance_to_face(target,cube,rot,penetrating,cube,rot)
    assert depth.item()>0 and d.item()>0


def test_usd_inward_patches_exist_and_are_not_contact_region_box_proxies():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import FingerSurfaces
    base=Path(__file__).resolve().parents[1]
    g=FingerSurfaces(base/'usd/robot_grasp_surface_v3.usda','cpu',torch.float32)
    assert set(g.triangles)=={'gripper_link','moving_jaw_so101_v1_link'}
    assert all(len(t)>10 for t in g.triangles.values())


def test_side_assignment_is_locked_until_reset():
    from types import SimpleNamespace as NS
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import FingerSurfaces
    g=FingerSurfaces.__new__(FingerSurfaces)
    local=patch()+torch.tensor([.04,0.,0.])
    g.triangles={'gripper_link':local,'moving_jaw_so101_v1_link':local}
    positions=torch.tensor([[[-.04,0.,0.],[.04,0.,0.]]])
    rotations=torch.tensor([[[0.,0.,0.,1.],[0.,0.,0.,1.]]])
    names=list(g.triangles)
    robot=NS(find_bodies=lambda name:([names.index(name)],[name]),
             data=NS(body_pos_w=NS(torch=positions),body_quat_w=NS(torch=rotations)))
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=1
    cube=torch.zeros(1,3);rot=torch.eye(3)[None]
    first=g.measure(robot,cube,rot,s)
    assert first['finger_target_axis'].item()==0
    positions[:]=torch.tensor([[[0.,-.04,0.],[0.,.04,0.]]])
    import math
    rotations[:]=torch.tensor([[[0.,0.,math.sqrt(.5),math.sqrt(.5)],[0.,0.,math.sqrt(.5),math.sqrt(.5)]]])
    again=g.measure(robot,cube,rot,s)
    assert again['finger_target_axis'].item()==0
    assert torch.allclose(first['fixed_finger_target'],again['fixed_finger_target'])
    s.reset();s.stage[:]=1
    assert g.measure(robot,cube,rot,s)['finger_target_axis'].item()==1


def test_open_jaw_center_offset_does_not_choose_wrong_cube_axis():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import choose_opposite_sides
    axis,sign=choose_opposite_sides(torch.tensor([[-.1,0.,0.]]),torch.tensor([[.1,0.,0.]]),
        torch.zeros(1,3),torch.eye(3)[None],fixed_inward=torch.tensor([[0.,-1.,0.]]))
    assert axis.item()==1 and sign.item()==1


def test_small_face_region_accepts_safe_offcenter_contact_but_not_air_gap():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import face_region_targets,distance_to_region
    cube=torch.zeros(1,3);rot=torch.eye(3)[None];axis=torch.tensor([0])
    center=torch.tensor([[-.025,0.,0.]])
    region=face_region_targets(center,rot,axis)
    tri=torch.tensor([[[-.025,.004,-.008],[-.025,.012,-.008],[-.025,.012,.008]],
                      [[-.025,.004,-.008],[-.025,.012,.008],[-.025,.004,.008]]])
    d,_,_,_,old=distance_to_region(region,cube,rot,tri,cube,rot)
    assert d.item()<1e-6 and old.item()>.003
    for shift in (-.005,.005):
        shifted=tri+torch.tensor([shift,0.,0.])
        d,_,_,_,_=distance_to_region(region,cube,rot,shifted,cube,rot)
        assert d.item()>.0049


def test_region_rotates_with_cube_and_stays_on_exact_face_plane():
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import face_region_targets
    turn=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    center=torch.tensor([[.1,.175,.3]])  # cube at (.1,.2,.3), -local-X face
    points=face_region_targets(center,turn,torch.tensor([0]))
    local=torch.bmm(points-torch.tensor([[[.1,.2,.3]]]),turn)
    assert torch.allclose(local[:,:,0],torch.full((1,9),-.025),atol=1e-6)
    assert local[:,:,1:].abs().max()<.00601
    assert torch.allclose(points[:,4],center)


def test_dragging_uses_actual_travel_not_instantaneous_contact_vibration():
    s=StageMachine(1,'cpu',reward_style='continuous');s.stage[:]=1
    s.update(sample(slide_speed=.03,table_step_distance=0.),1/30)
    assert s.group_contribution[0,1,2]==0
    s.update(sample(slide_speed=0.,table_step_distance=.001),1/30)
    assert torch.isclose(s.group_contribution[0,1,2],torch.tensor(-.04),atol=1e-6)
    s.update(sample(slide_speed=0.,table_step_distance=.01),1/30)
    assert s.group_contribution[0,1,2]<0


def test_tuning_reset_clears_contact_memory_and_reward_baselines():
    s=StageMachine(2,'cpu',reward_style='continuous')
    s.previous_finger_grasp[:]=True;s.previous_finger_jaw[:]=.7;s.simple_valid[:]=True
    s.reset(torch.tensor([0]))
    assert not s.previous_finger_grasp[0] and s.previous_finger_grasp[1]
    assert not s.simple_valid[0].any() and s.simple_valid[1].all()


def test_stage_one_graph_names_open_jaws_in_plain_language():
    from isaac_vision_rl.experiments.simple_rewards.panel import display_name
    assert display_name('open_jaws')=='Keep Jaws Open During Descent'
