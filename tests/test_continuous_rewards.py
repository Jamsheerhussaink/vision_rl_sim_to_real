import torch
from test_four_stage import values
from isaac_vision_rl.utils.four_stage import StageMachine,CONTINUOUS_WEIGHTS


def state(stage):
    s=StageMachine(1,'cpu',reward_mode='bounded',reward_style='continuous');s.stage[:]=stage
    return s


def test_continuous_reach_score_rises_near_falls_far_and_stays_when_stationary():
    for stage in (0,1):
        s=state(stage)
        if stage==0:s.reach_route[:]=1
        def measure(distance):
            s.update(values(reach_ready=False,pregrasp_distance=distance,
                            grasp_pose_distance=distance,alignment=.5,
                                tool_height_above_cube=.08,
                            grasp_alignment=.5,approach_from_above=1.,
                            grasp_approach_quality=1.),1/30)
            return s.group_contribution[0,0,0].clone()
        far=measure(.20)
        near=measure(.03)
        assert near>far>0
        assert torch.equal(measure(.03),near)
        assert torch.allclose(measure(.20),far)
        assert torch.equal(measure(.20),far)




def test_grasp_stage_does_not_keep_rewarding_jaw_opening():
    seen=[]
    for jaw in (.8,1.2):
        s=state(1)
        s.update(values(reach_ready=False,jaw=jaw,pregrasp_distance=.02,
                        grasp_pose_distance=.02,grasp_alignment=.7,
                        downward_cosine=1.),1/30)
        seen.append(s.group_contribution[0,0,0].clone())
    assert torch.equal(seen[0],seen[1])


def test_grasp_reward_is_continuous_while_valid_not_a_once_only_budget():
    s=state(1);m=values(jaw=.8,bilateral=True,enclosure=1.,height=.54)
    seen=[]
    # Ten steps stay within the grasp/lift phases; placement must stop this bonus.
    for _ in range(10):
        s.update(m,1/30)
        seen.append(s.group_contribution[0,1,:2].clone())
    assert all(torch.allclose(x,seen[0]) for x in seen)
    assert seen[-1].sum()>0


def test_grasp_shaping_stops_in_placement_but_table_slide_penalty_remains():
    s=state(3);m=values(jaw=.8,bilateral=True,enclosure=1.,height=.54,slide_speed=.05)
    s.update(m,1/30)
    assert torch.equal(s.group_contribution[0,1,:2],torch.zeros(2))
    assert s.group_contribution[0,1,2]<0


def test_grasp_shaping_remains_active_during_lift():
    s=state(2);m=values(jaw=.8,bilateral=True,enclosure=1.,height=.54)
    s.update(m,1/30)
    assert s.group_contribution[0,1,:2].sum()>0


def test_lowering_rewards_current_depth_and_maintained_floor_position():
    s=state(3)
    rewards=[]
    for h in [.53,.50,.46,.43,.43]:
        s.update(values(jaw=.8,bilateral=True,enclosure=1.,height=h,inside_box=True,over_box=True),1/30)
        rewards.append(float(s.group_contribution[0,3,1]))
    assert rewards[0]==0 and rewards[0]<rewards[1]<rewards[2]<rewards[3]
    assert abs(rewards[3]-rewards[4])<1e-7


def test_dragging_cannot_earn_box_approach_and_unsafe_tick_cannot_earn_positive():
    s=state(3)
    for d in [.2,.1,.05]:
        s.update(values(jaw=.8,bilateral=True,height=.425,box_distance=d,slide_speed=.05),1/30)
        assert s.group_contribution[0,3,0]==0
        assert s.group_contribution[0,1,2]<0
    s=state(2)
    s.update(values(jaw=.8,bilateral=True,height=.54,unsafe=True,enclosure=1.),1/30)
    assert s.failure.item() and not (s.group_contribution>0).any()


def test_success_remains_single_event_and_rates_are_time_integrated():
    s=state(3);bonuses=[]
    for _ in range(20):
        s.update(values(placed=True,inside_box=True,over_box=True,height=.43),.05)
        bonuses.append(float(s.group_contribution[0,3,2]))
    assert sum(bonuses)==40 and sum(b>0 for b in bonuses)==1
    totals=[]
    for fps in (30,60):
        s=state(0)
        totals.append(sum(float(s.update(values(reach_ready=False,pregrasp_distance=.1),1/fps)) for _ in range(fps*2)))
    assert abs(totals[0]-totals[1])<1e-6



def test_restored_distance_ignores_route_and_pose_gates():
    for route,gate in ((0,0.),(1,1.)):
        s=state(0)
        s.reach_route[:]=route
        s.update(values(reach_ready=False,pregrasp_distance=.05,
                        tool_height_above_cube=.08,alignment=0.,
                        approach_from_above=gate,grasp_approach_quality=gate),1/30)
        expected=.30*torch.exp(torch.tensor(-.5))/30
        assert torch.allclose(s.group_contribution[0,0,0],expected)


def test_overlap_checkpoint_can_select_dense_continuous_rewards():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    assert s.reward_style=='continuous'
    for _ in range(2):
        s.update(values(reach_ready=False,pregrasp_distance=.05),1/30)
        assert torch.allclose(s.group_contribution[0,0,0],.30*torch.exp(torch.tensor(-.5))/30)



def test_grasp_flow_from_reach_through_lift_then_stops_in_place():
    for mode in ('bounded','overlap'):
        for stage in range(4):
            s=StageMachine(1,'cpu',reward_mode=mode,reward_style='continuous')
            s.stage[:]=stage
            s.update(values(reach_ready=False,jaw=.8,bilateral=True,
                            enclosed=True,enclosure=1.,height=.50),1/30)
            reward=s.group_contribution[0,1,:2].sum().item()
            assert (reward>0) if stage<3 else (reward==0)


def test_stable_lift_advances_to_place_then_grasp_bonus_stops():
    s=state(2)
    m=values(jaw=.8,bilateral=True,enclosed=True,enclosure=1.,height=.57,
             cube_start_height=.40,speed=.05)
    for _ in range(16):s.update(m,1/30)
    assert s.stage.item()==3
    s.update(m,1/30)
    assert s.group_contribution[0,1,:2].sum().item()==0


def test_small_unsuccessful_outcome_is_once_per_episode_and_resettable():
    for reason in ('unsafe','timed_out'):
        s=state(0)
        m=values(reach_ready=False,**{reason:True})
        s.update(m,1/30)
        assert abs(s.group_contribution[0,3,2].item()+.1)<1e-6
        s.update(m,1/30)
        assert s.group_contribution[0,3,2].item()==0
        s.reset()
        s.update(m,1/30)
        assert abs(s.group_contribution[0,3,2].item()+.1)<1e-6


def test_reach_and_alignment_are_independent_and_ignore_jaw_and_height():
    def measure(distance,xy,cosine,jaw=0.,height=.08):
        s=state(0)
        s.update(values(reach_ready=False,pregrasp_distance=distance,xy_distance=xy,
                        downward_cosine=cosine,jaw=jaw,tool_height_above_cube=height),1/30)
        return s.group_contribution[0,0].clone()
    base=measure(.1,.02,1.)
    rotated=measure(.1,.02,0.)
    closer=measure(.02,.02,1.)
    assert base[0]==rotated[0] and base[2]>rotated[2]
    assert closer[0]>base[0] and closer[2]==base[2]
    assert measure(.1,.02,1.,jaw=1.2,height=.5)[2]>=base[2]
    assert measure(.1,.10,1.)[2]<base[2]



def test_closed_hover_loses_its_positive_return_after_grace():
    s=state(0)
    m=values(reach_ready=False,pregrasp_distance=.01,xy_distance=0.,
             tool_height_above_cube=.08,downward_cosine=1.,jaw=.65)
    for _ in range(400):s.update(m,1/30)
    assert s.group_contribution[0,0].sum().item()<0
    assert not s.descent_ready.item()


def test_overhead_open_pose_switches_distance_target_to_cube():
    s=state(0)
    s.update(values(reach_ready=False,pregrasp_distance=.005,grasp_pose_distance=.08,
                    xy_distance=0.,tool_height_above_cube=.08,downward_cosine=1.,jaw=1.2),1/30)
    assert s.descent_ready.item()
    assert torch.allclose(s.group_contribution[0,0,0],.30*torch.exp(torch.tensor(-2.))/30)


def test_overhead_opening_progress_prevents_idle_decay():
    s=state(0)
    for i in range(150):
        s.update(values(reach_ready=False,pregrasp_distance=.01,grasp_pose_distance=.08,
                        xy_distance=0.,tool_height_above_cube=.08,downward_cosine=1.,
                        jaw=.5+i*.003),1/30)
    assert s.idle_time.item()<.2 and not s.descent_ready.item()


def test_slow_cumulative_distance_progress_resets_idle_timer():
    s=state(0)
    for i in range(150):
        s.update(values(reach_ready=False,pregrasp_distance=.2-i*.0001,jaw=.5),1/30)
    assert s.idle_time.item()<.5


def test_opening_bonus_is_once_and_cannot_be_refilled_by_closing():
    s=state(0)
    def m(jaw):
        return values(reach_ready=False,pregrasp_distance=.01,grasp_pose_distance=.08,
                      xy_distance=0.,tool_height_above_cube=.08,downward_cosine=1.,jaw=jaw)
    s.update(m(1.2),1/30)
    first=s.group_contribution[0,0,2].item()
    assert first>=1. and s.open_bonus_paid.item() and s.descent_ready.item()
    s.update(m(.5),1/30);s.update(m(1.2),1/30)
    assert s.group_contribution[0,0,2].item()<.02
    s.reset();assert not s.open_bonus_paid.item() and s.open_best.item()==0


def test_opening_progress_and_bonus_are_not_available_far_from_cube():
    s=state(0)
    for i in range(150):
        s.update(values(reach_ready=False,pregrasp_distance=.2,xy_distance=.2,
                        tool_height_above_cube=.08,downward_cosine=1.,jaw=i*.008),1/30)
    assert s.idle_time.item()>3. and not s.open_bonus_paid.item()


def test_stage_two_opens_before_descent_and_still_requires_grasp_for_lift():
    s=state(1)
    closed=values(reach_ready=True,pregrasp_distance=.005,grasp_pose_distance=.08,
                  xy_distance=0.,tool_height_above_cube=.08,downward_cosine=1.,jaw=.75,
                  bilateral=False,enclosed=False)
    for _ in range(12):s.update(closed,1/30)
    assert s.stage.item()==1 and not s.descent_ready.item()
    expected=.30*torch.exp(torch.tensor(-.05))/30
    assert torch.allclose(s.group_contribution[0,0,0],expected)
    closed['jaw']=torch.tensor([1.2]);s.update(closed,1/30)
    assert s.descent_ready.item() and s.stage.item()==1
    assert torch.allclose(s.group_contribution[0,0,0],.30*torch.exp(torch.tensor(-2.))/30)


def test_new_centered_descent_is_rewarded_but_repeating_it_cannot_refill_bonus():
    s=state(1);s.descent_ready[:]=True
    def pose(height,jaw=1.2,xy=0.):
        return values(reach_ready=False,tool_height_above_cube=height,
                      pregrasp_distance=abs(.045-height),grasp_pose_distance=height,
                      xy_distance=xy,downward_cosine=1.,jaw=jaw)
    s.update(pose(.045),1/30)
    s.update(pose(.015),1/30);first=s.group_contribution[0,0,0].item()
    assert first>1.
    s.update(pose(.045),1/30);s.update(pose(.015),1/30)
    assert s.group_contribution[0,0,0].item()<.02


def test_closing_credit_only_near_cube_after_open_overhead_gate():
    def score(height,jaw):
        s=state(1);s.descent_ready[:]=True
        s.update(values(reach_ready=False,tool_height_above_cube=height,
                        grasp_pose_distance=height,pregrasp_distance=abs(.08-height),
                        xy_distance=0.,downward_cosine=1.,jaw=jaw,enclosure=0.),1/30)
        return s.group_contribution[0,1,0].item()
    assert score(.08,.55)==0
    assert score(.015,.55)>score(.015,1.2)


def test_ik_target_accounts_for_fixed_finger_axis_and_height():
    from isaac_vision_rl.utils.four_stage import ik_grasp_target
    cube=torch.tensor([[0.,0.,.425]])
    rotation=torch.eye(3)[None]
    target=ik_grasp_target(cube,rotation)
    assert torch.allclose(target,torch.tensor([[.016,0.,.4345]]))
    rotation=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    assert torch.allclose(ik_grasp_target(cube,rotation),torch.tensor([[0.,.016,.4345]]))


def test_closing_region_uses_teacher_target_error_not_cube_center_error():
    s=state(1);s.descent_ready[:]=True
    s.update(values(reach_ready=False,grasp_pose_distance=.005,xy_distance=.03,
                    tool_height_above_cube=.03,grasp_target_xy_distance=.004,
                    grasp_target_height_error=.004,downward_cosine=1.,jaw=.55,enclosure=0.),1/30)
    assert s.group_contribution[0,1,0].item()>0
    assert s.stage.item()==1  # Position guidance never substitutes for actual contacts.
