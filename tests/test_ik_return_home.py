import json
from pathlib import Path
import numpy as np
from isaac_vision_rl.utils.ik_return_home import can_return,make_home_plan


def test_recovery_blocks_faults_held_objects_and_unstable_cube():
    good=dict(unsafe=False,height=.43,speed=0.)
    assert can_return(np.zeros(6),good)[0]
    for m in (dict(good,unsafe=True),dict(good,height=.53),dict(good,speed=.1)):
        assert not can_return(np.zeros(6),m)[0]
    assert not can_return(np.full(6,np.nan),good)[0]


def test_home_route_from_recorded_placement_has_table_clearance(monkeypatch):
    monkeypatch.setenv('SO101_POLICY','rgb');monkeypatch.setenv('SO101_CURRICULUM','place')
    monkeypatch.setenv('SO101_REWARD_PROFILE','four-stage')
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    from isaac_vision_rl.utils.grasp_geometry import GraspGeometry,ARM_NAMES
    from isaac_vision_rl.utils.table_clearance import HandTableClearance
    c=PickplaceEnvCfg();r=c.scene.robot
    g=GraspGeometry(r.spawn.usd_path,r.init_state.pos,r.init_state.rot)
    h=HandTableClearance(r.spawn.usd_path,r.init_state.pos,r.init_state.rot)
    q=np.array([r.init_state.joint_pos[n] for n in ARM_NAMES]+[0.])
    # Representative post-withdraw joint pose for the shifted-cube scene.
    actual=np.array([-.311,.22,-.35,1.648,0.,1.2])
    plan=make_home_plan(g,h,actual,q)
    assert [n for n,_ in plan]==['open','clear_box','above_home','fold_home','close_home']
    assert np.allclose(plan[-1][1],q)
