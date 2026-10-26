"""Measured bottlenecks: tilted retained grasps, lift dwell and cropped wrists."""
from types import SimpleNamespace as NS
import math
import pytest
import torch
from test_four_stage import values
from isaac_vision_rl.utils.four_stage import StageMachine


def acquired(hold=.25):
    state=StageMachine(1,'cpu',reward_style='continuous',lift_hold_seconds=hold)
    state.stage[:]=1
    for _ in range(6):
        state.update(values(reach_ready=False,bilateral=True,jaw=.8,tool_cube_distance=.03),1/30)
    assert state.stage.item()==2
    return state


def test_verified_tilted_grasp_can_complete_short_hold_then_place_stage():
    state=acquired()
    held=values(bilateral=True,enclosed=False,jaw=.8,height=.59,
                tool_cube_distance=.04,speed=.10)
    for _ in range(8):state.update(held,1/30)
    assert state.stage.item()==3 and not state.success.item()
    state.update(held,1/30)
    assert state.group_contribution[0,1,:2].sum()==0  # no grasp income in Place


def test_contact_loss_distance_and_unsafe_pose_are_not_bypassed():
    for change in ({'bilateral':False},{'tool_cube_distance':.15},{'jaw':1.2}):
        state=acquired()
        m=values(bilateral=True,enclosed=False,jaw=.8,height=.59,tool_cube_distance=.04)
        m.update({k:torch.tensor([v]) for k,v in change.items()})
        for _ in range(8):state.update(m,1/30)
        assert state.failure.item() and state.stage.item()==2
    state=acquired()
    state.update(values(bilateral=True,jaw=.8,height=.59,unsafe=True),1/30)
    assert state.failure.item() and state.stage.item()==2


def test_unverified_touch_or_overlift_cannot_force_stage_advancement():
    for changes in ({'bilateral':False},{'enclosed':False}):
        state=StageMachine(1,'cpu',reward_style='continuous',lift_hold_seconds=.25)
        state.stage[:]=1
        m=values(reach_ready=False,bilateral=True,jaw=.8,height=.59,tool_cube_distance=.03)
        m.update({k:torch.tensor([v]) for k,v in changes.items()})
        for _ in range(30):state.update(m,1/30)
        assert state.stage.item()==1
    state=acquired()
    for _ in range(12):state.update(values(bilateral=True,jaw=.8,height=.65),1/30)
    assert state.stage.item()==2


def test_original_hold_duration_remains_available_and_invalid_durations_fail():
    state=acquired(.5)
    for _ in range(8):state.update(values(bilateral=True,jaw=.8,height=.59),1/30)
    assert state.stage.item()==2
    for value in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):StageMachine(1,'cpu',lift_hold_seconds=value)


def test_older_checkpoint_mode_can_release_only_over_box_without_false_success():
    for inside_box in (False,True):
        state=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
        state.stage[:]=3
        for _ in range(12):
            state.update(values(bilateral=False,jaw=.9,inside_box=inside_box,
                                opening_for_release=True),1/30)
        assert state.failure.item()!=inside_box
        assert not state.success.item()


def test_cropped_body_origins_do_not_hide_visible_pads_but_invisible_pads_fail():
    from isaac_vision_rl.tasks.pickplace.mdp.terminations import (
        gripper_outside_camera_view,gripping_area_outside_camera_view,
    )
    positions=torch.tensor([[[.49,0.,1.],[.49,0.,1.],[.49,0.,1.]]])
    s=math.sqrt(.5)
    quats=torch.tensor([[[0.,s,0.,s],[0.,0.,-s,s],[0.,0.,0.,1.]]])
    indices={'gripper_link':0,'moving_jaw_so101_v1_link':1,'gripper_frame_link':2}
    robot=NS(data=NS(body_pos_w=NS(torch=positions),body_quat_w=NS(torch=quats)),
             find_bodies=lambda name:([indices[name]],[name]))
    camera=NS(image_shape=(100,100),data=NS(pos_w=torch.zeros(1,3),
        quat_w_ros=torch.tensor([[0.,0.,0.,1.]]),
        intrinsic_matrices=torch.tensor([[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]]])))
    env=NS(scene={'robot':robot,'camera':camera},device='cpu',num_envs=1)
    assert gripper_outside_camera_view(env).item()
    assert not gripping_area_outside_camera_view(env).item()
    positions[0,1,0]=1.
    assert gripping_area_outside_camera_view(env).item()
