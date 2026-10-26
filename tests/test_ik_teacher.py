from types import SimpleNamespace as NS
import json
import numpy as np
from isaac_vision_rl.utils.ik_teacher import IKTeacher,offline_check
from isaac_vision_rl.utils.four_stage import PREGRASP_CLEARANCE_M
from isaac_vision_rl.utils.demonstrations import Demonstration


def _clearance_stub():
    return NS(check_segment=lambda *a,**k:.01,
              body_poses=lambda joints:{'gripper_link':np.eye(4)})


def test_teacher_waits_for_real_pregrasp_condition_and_times_out():
    g=NS(solve=lambda pos,q,**kw:q)
    guard=_clearance_stub()
    t=IKTeacher(g,guard,np.zeros(6),[0,0,.425],[0,.1,.401],.05)
    t.phase=1
    actual=np.r_[np.zeros(5),1.2]
    m=dict(unsafe=False,reach_ready=False,clearance=.01,downward_cosine=.5)
    for _ in range(20):t.step(actual,[0,0,.505],[0,0,.425],m)
    assert t.phase_name=='reach'  # Passing time alone cannot grant a grasp.
    for _ in range(4):t.step(actual,[0,0,.505],[0,0,.425],dict(m,downward_cosine=1.))
    assert t.phase_name=='descend'
    t.enter=False;t.position=None
    for _ in range(250):t.step(np.ones(6),[0,0,.505],[0,0,.425],m)
    assert 'timeout' in t.failure


def test_teacher_reach_waypoint_matches_reward_stage_pregrasp_height():
    g=NS(solve=lambda pos,q,**kw:q)
    guard=_clearance_stub()
    cube=np.array([.1,-.2,.425])
    t=IKTeacher(g,guard,np.zeros(6),cube,[0,.1,.401],.05)
    t.phase=1
    _,position=t._goal(np.zeros(3),cube)
    assert np.isclose(position[2]-cube[2],PREGRASP_CLEARANCE_M)


def test_descent_cannot_close_until_reference_and_measured_arm_settle():
    g=NS(solve=lambda pos,q,**kw:q)
    guard=_clearance_stub()
    t=IKTeacher(g,guard,np.zeros(6),[0,0,.425],[0,.1,.401],.05)
    t.phase=2;t.enter=False;t.position=np.array([0.,0.,.4345])
    t.goal=np.array([.2,0.,0.,0.,0.,1.2])
    m=dict(unsafe=False)
    actual=t.goal.copy();actual[0]-=.03
    for _ in range(20):t.step(actual,t.position,t.cube,m)
    assert t.phase_name=='descend'  # Error above the new 0.02 rad settled tolerance.
    # A brief pass through the target while moving must not complete descent.
    t.step(t.goal.copy(),t.position,t.cube,m)
    assert t.phase_name=='descend'
    for _ in range(8):
        t.step(t.goal.copy(),t.position,t.cube,m)
        if t.phase_name=='grasp':break
    assert t.phase_name=='grasp'


def test_demonstrations_align_pre_observation_with_command_and_reject_failed_episode(tmp_path):
    d=Demonstration(tmp_path/'episode',chunk_size=2)
    obs=np.array([1.,2.]);image=np.ones((3,4,4));action=np.zeros(6)
    d.add(obs,image,action,0.,False,'reach')
    obs[:]=99;image[:]=99  # Recorder must not retain mutable simulator views.
    d.add([3.,4.],np.zeros((3,4,4)),np.ones(6),1.,True,'place')
    d.finish(False,'lost grasp')
    with np.load(tmp_path/'episode/chunk_00000.npz') as x:
        assert x['policy'][0,0]==1 and x['image'][0,0,0,0]==1
        assert x['action'][0].sum()==0 and x['action'][1].sum()==6
    meta=json.loads((tmp_path/'episode/metadata.json').read_text())
    assert not meta['eligible_for_cloning'] and meta['steps']==2


def test_settled_three_point_one_mm_tracking_residual_does_not_deadlock_teacher():
    g=NS(solve=lambda pos,q,**kw:q);guard=_clearance_stub()
    teacher=IKTeacher(g,guard,np.zeros(6),[0,0,.425],[0,.1,.401],.05)
    teacher.phase=2;teacher.enter=False;teacher.position=np.array([0.,0.,.4345])
    teacher.goal=np.r_[np.zeros(5),1.2];teacher.target=teacher.goal.copy()
    for _ in range(8):
        teacher.step(teacher.goal.copy(),teacher.position+[.0031,0.,0.],teacher.cube,dict(unsafe=False))
        if teacher.phase_name=='grasp':break
    assert teacher.phase_name=='grasp'


def test_offline_nominal_path_is_reachable_and_has_sampled_hand_table_clearance(monkeypatch):
    monkeypatch.setenv('SO101_POLICY','rgb');monkeypatch.setenv('SO101_CURRICULUM','place')
    monkeypatch.setenv('SO101_REWARD_PROFILE','four-stage')
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    result=offline_check(PickplaceEnvCfg())
    assert len(result['waypoints'])==9
    assert min(x['minimum_table_clearance_m'] for x in result['waypoints'])>=.005
