"""Versioned four-stage task. Three terms per reward family, with explicit overlapping activation."""
import math
import os
import torch

VERSION = ('four_stage_v25_overlap_ik_grasp_target' if os.environ.get('SO101_GRASP_REWARD_MODE','bounded')=='overlap' and os.environ.get('SO101_REWARD_STYLE','progress')=='continuous'
           else 'four_stage_v8_overlap_topdown_approach' if os.environ.get('SO101_GRASP_REWARD_MODE','bounded')=='overlap'
           else 'four_stage_v25_ik_grasp_target' if os.environ.get('SO101_REWARD_STYLE','progress')=='continuous'
           else 'four_stage_v10_staged_topdown_approach')
# Cube-centre height in metres relative to environment origin, not lift distance.
# Box rim is about 0.491 m; 5 cm cube bottom clears it in this provisional band.
LIFT_MIN=.53
LIFT_MAX=.55
LIFT_RISE_MIN_M=float(os.environ.get('SO101_LIFT_RISE_MIN','0.15'))
LIFT_RISE_MAX_M=float(os.environ.get('SO101_LIFT_RISE_MAX','0.20'))
PREGRASP_CLEARANCE_M=.045
OPEN_JAW_DOWN_COSINE_THRESHOLD=.8


def pregrasp_target(cube_position):
    """Safe first waypoint centered 8 cm above the cube."""
    target=cube_position.clone()
    target[...,2]+=PREGRASP_CLEARANCE_M
    return target


def ik_grasp_target(cube_position, fixed_finger_rotation):
    """Same asymmetric-finger target as IKTeacher._goal; reward-only guidance."""
    target=cube_position+fixed_finger_rotation[..., :, 0]*.016
    target=target.clone()
    target[...,2]+=.0095
    return target


def approach_from_above_quality(xy_distance, tool_height_above_cube):
    """Give lateral-approach credit only when the gripper is nearly overhead."""
    height_quality=((tool_height_above_cube-(PREGRASP_CLEARANCE_M-.02))/.02).clamp(0,1)
    centered_quality=torch.exp(-torch.square(xy_distance/.05))
    return height_quality*centered_quality


def pregrasp_alignment(xy_distance, downward_alignment, jaw_position, approach_quality):
    """Reward centered overhead alignment and opening before descent."""
    open_fraction=(jaw_position/1.10).clamp(0,1)
    return (torch.exp(-torch.square(xy_distance/.035))*downward_alignment
            *(.5+.5*open_fraction)*approach_quality)


def downward_orientation_score(downward_cosine):
    """Continuous wrist feedback: up=-1, sideways=-0.5, fully down=+1."""
    quality=(downward_cosine.clamp(-1.,1.)+1.)*.5
    return 2.*quality.square()-1.


def gated_open_jaw_quality(downward_cosine, approach_quality, jaw_position):
    """Reward open jaws only when the wrist points down over the cube."""
    downward=(downward_cosine-OPEN_JAW_DOWN_COSINE_THRESHOLD).clamp(0.,1.)/(1.-OPEN_JAW_DOWN_COSINE_THRESHOLD)
    open_fraction=(jaw_position/1.10).clamp(0.,1.)
    return open_fraction*downward*approach_quality.clamp(0.,1.)

def lift_height_quality(lift_rise):
    """Score cube rise from its reset height, with a preferred 15-20 cm band."""
    rising=(lift_rise/LIFT_RISE_MIN_M).clamp(0,1)
    falling=(1-(lift_rise-LIFT_RISE_MAX_M).clamp(min=0)/.05).clamp(0,1)
    return rising*falling


def table_slide_mask(height,whole_box_xy,box_bottom):
    """Height proxy for source tabletop motion, excluding cube inside box floor region."""
    within_box=whole_box_xy & (box_bottom>=-.002) & (box_bottom<=.035)
    return (height<.44) & ~within_box


NAMES = ('REACH', 'GRASP', 'LIFT', 'PLACE')
TERMS = (
    ('overhead_approach', 'table_clearance', 'gripper_readiness'),
    ('current_enclosure', 'bilateral_contact', 'table_slide_speed'),
    ('supported_lift_progress', 'held_lift_dwell', 'object_speed'),
    ('box_approach_progress', 'lowering_progress', 'settled_release'),
)
WEIGHTS = ((.5, -1., .1), (4., 8., -2.), (12., 8., -1.), (10., 6., 40.))
CONTINUOUS_WEIGHTS=((.30,-1.,.10),(.10,.10,-2.),(.15,.15,-1.),(.15,.15,40.))
CONDITIONS = (
    'Safe aligned overhead pose OR valid opposing enclosed grasp, for 0.15 s',
    'Cube enclosed, opposing two-finger contact and closed jaws for 0.2 s',
    f'Cube rises {LIFT_RISE_MIN_M:.2f}–{LIFT_RISE_MAX_M:.2f} m from reset, supported, speed <0.25 m/s, for 0.5 s',
    'Whole cube settled in box, released, tool withdrawn >6.5 cm, for 0.4 s',
)


def active_reward_groups(stage):
    """Zero-based stage -> active reward families (including their penalties)."""
    return (0, 1) if stage <= 1 else (1, 2) if stage == 2 else (1, 3)


class StageMachine:
    """Pure tensor state machine; simulator data enters only through measurements."""
    def __init__(self, count, device, reward_mode=None, goal=None, reward_style=None):
        self.reward_mode=reward_mode or os.environ.get("SO101_GRASP_REWARD_MODE","bounded")
        self.reward_style=reward_style or os.environ.get("SO101_REWARD_STYLE","progress")
        if self.reward_style not in ("continuous","progress"):raise ValueError("Invalid reward style")
        self.weights=CONTINUOUS_WEIGHTS if self.reward_style=="continuous" else WEIGHTS
        self.goal=goal or os.environ.get("SO101_TRAINING_GOAL","place")
        if self.reward_mode not in ("bounded","overlap") or self.goal not in ("lift","place"):
            raise ValueError("Invalid grasp reward mode or training goal")
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.reward_stage = self.stage.clone()
        self.dwell = torch.zeros(count, device=device)
        self.lost = self.dwell.clone()
        self.best = torch.zeros(count, 2, device=device)
        self.previous = torch.zeros(count, 2, device=device)
        # Reach has two ordered subgoals: raise to cube clearance, then center XY.
        self.reach_route = torch.zeros(count, dtype=torch.long, device=device)
        self.reach_route_dwell = torch.zeros(count, device=device)
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        self.cube_start_height = torch.zeros(count, device=device)
        self.cube_start_height_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.success = self.started.clone()
        self.failure = self.started.clone()
        self.outcome_paid = self.started.clone()
        self.descent_ready = self.started.clone()
        self.idle_time = self.dwell.clone()
        self.open_best = self.dwell.clone()
        self.descent_best = self.dwell.clone()
        self.closing_best = self.dwell.clone()
        self.open_bonus_paid = self.started.clone()
        self.progress_best = torch.full_like(self.dwell,float('inf'))
        self.grasp_best = torch.zeros(count,2,device=device)
        self.contact_dwell = torch.zeros(count,device=device)
        self.raw = torch.zeros(count, 3, device=device)
        self.contribution = self.raw.clone()
        self.group_raw = torch.zeros(count, 4, 3, device=device)
        self.group_contribution = self.group_raw.clone()
        self.last = None
        self.slide_distance = self.dwell.clone()
        self.cube_seen = self.started.clone()
        self.previous_xy = torch.zeros(count,2,device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in (self.stage, self.reward_stage, self.dwell, self.lost, self.best,
                      self.previous, self.started, self.success, self.failure, self.raw, self.contribution,
                      self.group_raw, self.group_contribution, self.grasp_best, self.contact_dwell, self.outcome_paid, self.descent_ready, self.idle_time, self.open_best, self.open_bonus_paid, self.descent_best, self.closing_best):
            value[ids] = 0
        self.progress_best[ids] = float("inf")
        self.cube_start_height[ids] = 0
        self.cube_start_height_valid[ids] = False
        self.reach_route[ids] = 0
        self.reach_route_dwell[ids] = 0
        self.slide_distance[ids]=0
        self.cube_seen[ids]=False
        self.last = None

    def update(self, m, dt, inspection=False):
        old = self.stage.clone()
        self.reward_stage[:] = old
        first_in_stage = ~self.started
        # Do not reward lateral movement until the gripper has first risen above
        # the cube. Once safely high, keep the route on the centering subgoal.
        tool_height = m.get('tool_height_above_cube',
                            torch.full_like(m['pregrasp_distance'], PREGRASP_CLEARANCE_M))
        raised_enough = tool_height >= PREGRASP_CLEARANCE_M - .005
        in_reach = old == 0
        self.reach_route_dwell[:] = torch.where(
            in_reach & raised_enough, self.reach_route_dwell + dt,
            torch.where(in_reach, 0., self.reach_route_dwell))
        route_switch = in_reach & (self.reach_route == 0) & (self.reach_route_dwell >= .10)
        self.reach_route[:] = torch.where(route_switch, 1, self.reach_route)
        # If the gripper falls well below the safe overhead band before leaving
        # Reach, send it back to the raise subgoal instead of rewarding a side approach.
        # Descending while horizontally centered over the cube is safe and desired.
        xy_reach = m.get('xy_distance', m['pregrasp_distance'])
        route_reset = in_reach & (self.reach_route == 1) & (tool_height < PREGRASP_CLEARANCE_M - .02) & (xy_reach >= .035)
        self.reach_route[:] = torch.where(route_reset, 0, self.reach_route)
        self.reach_route_dwell[:] = torch.where(route_reset, 0., self.reach_route_dwell)
        route_changed = route_switch | route_reset
        active = ~(self.success | self.failure)
        grasp = m['bilateral'] & m['enclosed'] & (m['jaw'] < 1.10)
        # Measure lift relative to the cube's height at the beginning of this
        # episode. In real training the first observation after reset captures
        # that height; tests may provide it explicitly as cube_start_height.
        initial_height=m.get('cube_start_height',m['height'])
        capture=~self.cube_start_height_valid
        self.cube_start_height[:]=torch.where(capture,initial_height,self.cube_start_height)
        self.cube_start_height_valid|=capture
        lift_rise=(m['height']-self.cube_start_height).clamp(min=0.)
        m['lift_rise']=lift_rise
        if self.reward_mode=='bounded':
            held=grasp & (lift_rise>=LIFT_RISE_MIN_M-1e-5) & (lift_rise<=LIFT_RISE_MAX_M+1e-5) & (m['speed']<.25)
            height_score=lift_height_quality(lift_rise)
            too_high=((lift_rise-LIFT_RISE_MAX_M)/.05).clamp(0,2)
        else:
            held=grasp & (lift_rise>=LIFT_RISE_MIN_M-1e-5) & (lift_rise<=LIFT_RISE_MAX_M+1e-5) & (m['speed']<.25)
            height_score=lift_height_quality(lift_rise)
            too_high=((lift_rise-LIFT_RISE_MAX_M)/.05).clamp(0,2)
        release_zone = m['inside_box'] & (m['jaw'] > 1.10)
        if self.reward_mode=='bounded':
            # Contact is naturally lost before slow jaws become fully open.
            release_zone |= (old==3) & m['inside_box'] & m.get('opening_for_release',torch.zeros_like(grasp))
        lost = (old >= 2) & ~grasp & ~release_zone
        self.lost[:] = torch.where(lost, self.lost + dt, 0.)
        if inspection:
            self.failure.zero_()  # Inspection may recover after an unsafe pose/lost grasp.
        else:
            self.failure |= m['unsafe'] | ((self.lost >= .2) & (old >= 2))
        # A direct valid grasp also proves the cube has been reached. Requiring
        # open jaws exclusively would strand a closed grasp in REACH.
        good = torch.where(old == 0, m['reach_ready'] | grasp,
               torch.where(old == 1, grasp,
               torch.where(old == 2, held, m['placed']))) & ~self.failure & ~m['unsafe']
        self.dwell[:] = torch.where(good, self.dwell + dt, 0.)
        durations = torch.tensor([.08, .2, .5, .4], device=old.device)[old]
        complete = (self.dwell >= durations) & active & ~self.failure
        # Progress only pays for a new episode/stage best. No reward on first measurement.
        score = torch.stack((
            torch.where(old == 0, (1-m['pregrasp_distance']/.30).clamp(0,1),
                torch.where(old == 1, m['enclosure'],
                torch.where(old == 2, torch.where(grasp, height_score, 0.),
                            (1-m['box_distance']/.30).clamp(0,1)))),
            torch.where(old == 0, m['alignment'],
                torch.where(old == 1, (self.dwell/.2).clamp(0,1),
                torch.where(old == 2, (self.dwell/.5).clamp(0,1), m['lowering']))),
        ), dim=-1)
        score[:, 0] = torch.where((old == 3) & ~grasp, self.best[:, 0], score[:, 0])
        score[:, 1] = torch.where((old == 3) & ~grasp, self.best[:, 1], score[:, 1])
        if self.reward_mode=='bounded':
            # Do not consume approach progress while carrying above the band.
            score[:,0]=torch.where((old==3) & (too_high>0),self.best[:,0],score[:,0])
        # Dwell starts at zero so the very first valid contact tick is counted.
        baseline = score.clone()
        baseline[:, 1] = torch.where((old == 1) | (old == 2), 0., baseline[:, 1])
        self.best[:] = torch.where(self.started[:, None], self.best, baseline)
        progress = (score-self.best).clamp(0,1)
        self.best[:] = torch.maximum(score, self.best)
        self.started[:] = True
        # Keep the independent family values for charts/logs. The three manager
        # slots aggregate corresponding columns; no active family is overwritten.
        self.group_raw.zero_()
        reach = (old <= 1)
        grasp_active = (old >= 0)
        # Stop positive grasp shaping once placement starts. Grasp validity is
        # still enforced by stage progression, lost-grasp failure, and placement.
        grasp_reward_active = grasp_active & (old < 3)
        # Reach follows two waypoints: first raise to cube clearance, then move
        # over the cube. The Grasp stage starts from that overhead pose and
        # rewards a downward, centered descent. This prevents a low diagonal
        # approach from scoring like a correct top-down approach.
        grasp_pose_distance=m.get('grasp_pose_distance',m['pregrasp_distance'])
        raise_distance=(PREGRASP_CLEARANCE_M-tool_height).abs()
        reach_distance=torch.where(self.reach_route==0,raise_distance,m['pregrasp_distance'])
        goal_distance=torch.where(old==0,reach_distance,grasp_pose_distance)
        # Reset the distance baseline at the raise->3D-center switch so changing
        # subgoals cannot create a fake reward spike.
        previous_distance=torch.where(first_in_stage | route_changed,goal_distance,self.previous[:,0])
        distance_progress=(previous_distance-goal_distance)/.05
        above=m.get('approach_from_above',torch.ones_like(m['pregrasp_distance']))
        grasp_approach=m.get('grasp_approach_quality',torch.ones_like(above))
        approach_quality=torch.where(old==0,
            torch.where(self.reach_route==0,torch.ones_like(above),above),grasp_approach)
        cosine=m.get('downward_cosine',torch.ones_like(above)).clamp(-1.,1.)
        wrist_quality=((cosine+1.)*.5).square()
        open_quality=gated_open_jaw_quality(cosine,above,m['jaw'])
        # Give a distinct, readable signal for wrist-down and open jaws in Reach.
        # Once Grasp begins, remove the jaw-open target so closing is encouraged.
        readiness=torch.where(old==0,.5*(wrist_quality+open_quality),wrist_quality)
        previous_readiness=torch.where(first_in_stage | route_changed,readiness,self.previous[:,1])
        readiness_progress=readiness-previous_readiness
        self.group_raw[:,0,0]=distance_progress*approach_quality*reach
        self.group_raw[:,0,2]=readiness_progress*reach
        self.previous[:,0]=torch.where(reach,goal_distance,self.previous[:,0])
        self.previous[:,1]=torch.where(reach,readiness,self.previous[:,1])
        self.group_raw[:,0,1] = ((.008-m['clearance'])/.008).clamp(0,1)*dt*reach
        # Grasp quality/contact shaping runs through grasp and lift, then stops
        # in placement so waiting while carrying cannot farm grasp reward.
        if self.reward_mode=='bounded':
            # Pay only new best enclosure and continuous contact, at most 4+8
            # points per episode. Releasing/regrasping never refills this budget.
            safe_grasp=grasp & ~m['unsafe'] & grasp_reward_active & active
            self.contact_dwell[:]=torch.where(safe_grasp,self.contact_dwell+dt,0.)
            quality=torch.stack((m['enclosure'].clamp(0,1),
                                 (self.contact_dwell/.2).clamp(0,1)),dim=-1)
            quality*= (grasp_reward_active & active & ~m['unsafe'])[:,None]
            self.group_raw[:,1,:2]=(quality-self.grasp_best).clamp(0,1)
            self.grasp_best[:]=torch.maximum(self.grasp_best,quality)
        else:
            self.group_raw[:,1,0] = m['enclosure'].clamp(0,1)*dt*grasp_reward_active
            self.group_raw[:,1,1] = grasp.float()*dt*grasp_reward_active
        self.group_raw[:,1,2] = (m['slide_speed']/.05).clamp(0,1)*dt*grasp_active
        self.group_raw[:,2,:2] = progress*(old==2)[:,None]
        # Signed height term: new-best lift credit, minus time-integrated
        # excess height. A previous maximum cannot hide later over-lifting.
        self.group_raw[:,2,0] -= too_high*dt*(old==2)
        self.group_raw[:,2,2] = (m['speed']/.25).clamp(0,1)*dt*(old==2)
        self.group_raw[:,3,:2] = progress*(old==3)[:,None]
        # Carrying excessively high must not earn box-approach credit;
        # lowering into the box remains allowed below the carry band.
        self.group_raw[:,3,0] *= (lift_rise<=LIFT_RISE_MAX_M)
        self.group_raw[:,3,1] -= too_high*dt*(old==3)
        self.group_raw[:,3,2] = (complete & (old==3)).float()
        if self.reward_style=='continuous':
            # Restore independent dense distance feedback, separate from pose.
            # Once safely overhead with open jaws, descend rather than hovering.
            xy=m.get('xy_distance',m['pregrasp_distance'])
            overhead=(m['pregrasp_distance']<.04)&(xy<.035)&(cosine>.75)&(tool_height>=PREGRASP_CLEARANCE_M-.02)&(m['jaw']>1.10)
            newly_overhead=overhead & ~self.descent_ready & (old<=1)
            self.descent_ready |= overhead & (old<=1)
            descent=self.descent_ready
            continuous_distance=torch.where(descent,grasp_pose_distance,m['pregrasp_distance'])
            # Count cumulative distance improvement and gated jaw opening.
            # Baselines advance only after a meaningful increment, so slow
            # progress is not lost by updating the baseline every tiny step.
            improved=continuous_distance < self.progress_best-.001
            pinch_xy=m.get('grasp_target_xy_distance',xy)
            pinch_height_error=m.get('grasp_target_height_error',tool_height)
            centered=torch.exp(-torch.square(pinch_xy/.025))*cosine.clamp(0.,1.)
            closing_gate=(pinch_xy<.02)&(pinch_height_error.abs()<.02)&(cosine>.8)&self.descent_ready&(old==1)
            closing_fraction=((1.2-m['jaw'])/.65).clamp(0.,1.)
            closing_progress=closing_gate & (closing_fraction>self.closing_best+.02)
            self.closing_best[:]=torch.where(closing_progress,closing_fraction,self.closing_best)
            opening_gate=(old<=1)&(m['pregrasp_distance']<.05)&(cosine>.75)&(xy<.035)&(tool_height>=PREGRASP_CLEARANCE_M-.02)
            open_fraction=(m['jaw']/1.10).clamp(0.,1.)
            opening_progress=opening_gate & (open_fraction>self.open_best+.01)
            self.open_best[:]=torch.where(opening_progress,open_fraction,self.open_best)
            reset_progress=first_in_stage | newly_overhead
            self.idle_time[:]=torch.where(improved|opening_progress|closing_progress|reset_progress,0.,self.idle_time+dt)
            self.progress_best[:]=torch.where(reset_progress|improved,continuous_distance,self.progress_best)
            opened_now=overhead & (old<=1) & ~self.open_bonus_paid & ~m['unsafe'] & active
            self.open_bonus_paid |= opened_now
            stalled=(self.idle_time-3.).clamp(min=0.)
            approach_decay=torch.exp(-stalled/2.)
            distance_scale=torch.where(self.descent_ready,torch.full_like(continuous_distance,.04),
                                       torch.full_like(continuous_distance,.10))
            distance_score=torch.exp(-continuous_distance/distance_scale)*approach_decay
            # At most +3 per episode for new best centered descent, never for
            # repeating an up/down motion. Stay open until within closing range.
            descent_quality=((PREGRASP_CLEARANCE_M-tool_height)/(PREGRASP_CLEARANCE_M-.0095)).clamp(0.,1.)*centered
            jaw_ready=(m['jaw']>1.10)|closing_gate
            descent_quality*=self.descent_ready*(old==1)*jaw_ready*~m['unsafe']*active
            descent_increment=(descent_quality-self.descent_best).clamp(min=0.)
            self.descent_best[:]=torch.maximum(self.descent_best,descent_quality)
            idle_penalty=.05*(stalled>0).float() if not inspection else torch.zeros_like(stalled)
            self.group_raw[:,0,0]=(distance_score-idle_penalty/self.weights[0][0])*dt*reach
            self.group_raw[:,0,0]+=3.*descent_increment/self.weights[0][0]
            self.group_raw[:,0,2]=0.
            self.group_raw[:,1,0]=m['enclosure'].clamp(0,1)*dt*grasp_reward_active
            # Dense closing feedback exists only at the near-cube pinch pose.
            self.group_raw[:,1,0]+=.20/self.weights[1][0]*closing_fraction*centered*closing_gate*dt
            self.group_raw[:,1,1]=grasp.float()*dt*grasp_reward_active
            self.group_raw[:,2,0]=(torch.where(grasp,height_score,0.)-too_high)*dt*(old==2)
            self.group_raw[:,2,1]=held.float()*dt*(old==2)
            over_box=m.get('over_box',m['inside_box'])
            valid_carry=grasp & (lift_rise>=LIFT_RISE_MIN_M)
            box_gate=(valid_carry | over_box) & (lift_rise<=LIFT_RISE_MAX_M)
            self.group_raw[:,3,0]=(1-m['box_distance']/.30).clamp(0,1)*dt*(old==3)*box_gate
            descent=m.get('descent',((LIFT_MIN-m['height'])/.10).clamp(0,1))
            self.group_raw[:,3,1]=(descent*over_box-too_high)*dt*(old==3)
        # Terminal outcome: +40 for successful Place, -0.1 for an
        # unsuccessful ending. Ordinary ongoing steps receive no outcome penalty.
        timed_out=m.get('timed_out',torch.zeros_like(self.failure))
        unsuccessful=(self.failure | timed_out) & ~(complete & (old==3)) & ~self.outcome_paid & active
        if inspection:unsuccessful=torch.zeros_like(unsuccessful)
        self.group_raw[:,3,2]=torch.where(unsuccessful,
            torch.full_like(self.group_raw[:,3,2],-.1/self.weights[3][2]),
            self.group_raw[:,3,2])
        self.outcome_paid |= unsuccessful | (complete & (old==3))
        if self.reward_mode=='bounded':
            # A new failure cannot collect positive task credit on its terminal tick.
            self.group_raw[:]=torch.where((self.failure | m["unsafe"])[:,None,None] & (self.group_raw>0),0.,self.group_raw)
        # Independent alignment score: center over the cube and point down.
        # No hand-height target or jaw-opening reward is part of this term.
        xy=m.get('xy_distance',m['pregrasp_distance'])
        alignment_quality=torch.exp(-torch.square(xy/.05))*cosine.clamp(0.,1.)
        if self.reward_style=="continuous":
            # Jaw opening is useful only at the aligned overhead pose; stop in Grasp.
            jaw_bonus=2.*gated_open_jaw_quality(cosine,above,m['jaw'])*(old<=1)*~self.descent_ready
            self.group_raw[:,0,2]=(alignment_quality+jaw_bonus)*approach_decay*dt*reach
            # +1 once when safe overhead jaws reach the unchanged opening gate.
            self.group_raw[:,0,2]+=opened_now.float()/self.weights[0][2]
        else:self.group_raw[:,0,2]=0.
        self.group_raw[:]=torch.where((self.failure | m['unsafe'])[:,None,None] & (self.group_raw>0),0.,self.group_raw)
        self.group_raw *= active[:,None,None]
        self.group_contribution[:] = self.group_raw * torch.tensor(self.weights,device=old.device)
        self.raw[:] = self.group_raw.sum(1)
        self.contribution[:] = self.group_contribution.sum(1)
        terminal_stage=2 if self.goal=="lift" else 3
        self.success |= complete & (old == terminal_stage)
        advance = complete & (old < terminal_stage)
        self.stage[advance] += 1
        self.dwell[advance] = 0
        self.started[advance] = False
        self.best[advance] = 0
        return self.contribution.sum(-1)


def machine(env):
    if not hasattr(env, '_four_stage'):
        env._four_stage = StageMachine(env.num_envs, env.device)
    return env._four_stage


def measurements(env):
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact_diagnostics, cube_in_box
    from isaaclab.utils.math import matrix_from_quat
    from isaac_vision_rl.utils.hand_table import hand_table_metrics
    from isaac_vision_rl.tasks.pickplace.mdp.terminations import gripper_outside_camera_view, object_outside_camera_view
    robot, cube, box = (env.scene[n] for n in ('robot','cube','box'))
    body = robot.find_bodies('gripper_frame_link')[0][0]
    jaw_id = robot.find_joints('gripper')[0][0]
    origin = env.scene.env_origins
    origin = origin.torch if hasattr(origin, 'torch') else origin
    cp = cube.data.root_pos_w.torch
    tool = robot.data.body_pos_w.torch[:,body]
    offset = cp-tool
    xy = torch.linalg.vector_norm(offset[:,:2],dim=-1)
    speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch,dim=-1)
    height = cp[:,2]-origin[:,2]
    clearance, force = hand_table_metrics(env)
    contact = bilateral_cube_contact_diagnostics(env)
    bilateral = contact['bilateral']
    jaw = robot.data.joint_pos.torch[:,jaw_id]
    distance = torch.linalg.vector_norm(offset,dim=-1)
    tool_rotation=matrix_from_quat(robot.data.body_quat_w.torch[:,body])
    fixed_body=robot.find_bodies('gripper_link')[0][0]
    fixed_rotation=matrix_from_quat(robot.data.body_quat_w.torch[:,fixed_body])
    grasp_target=ik_grasp_target(cp,fixed_rotation)
    target_error=tool-grasp_target
    target_distance=torch.linalg.vector_norm(target_error,dim=-1)
    target_xy=torch.linalg.vector_norm(target_error[:,:2],dim=-1)
    downward_cosine=(-tool_rotation[:,2,2]).clamp(-1.,1.)
    direction=downward_cosine.clamp(0.,1.)
    pre = pregrasp_target(cp)
    pd = torch.linalg.vector_norm(pre-tool,dim=-1)
    above = approach_from_above_quality(xy,tool[:,2]-cp[:,2])
    alignment = pregrasp_alignment(xy,direction,jaw,above)
    # During Grasp, encourage a centered downward descent toward the cube.
    grasp_alignment = torch.exp(-torch.square(xy/.035)-torch.square(offset[:,2]/.045))*direction
    enclosure = (1-xy/.03).clamp(0,1)*(1-offset[:,2].abs()/.035).clamp(0,1)*((1.2-jaw)/.25).clamp(0,1)*direction
    bp = box.data.root_pos_w.torch
    box_distance = torch.linalg.vector_norm((cp-bp)[:,:2],dim=-1)
    inside = cube_in_box(env)
    box_rotation=matrix_from_quat(box.data.root_quat_w.torch)
    rel=torch.bmm(box_rotation.transpose(1,2),(cp-bp).unsqueeze(-1)).squeeze(-1)
    cube_rotation=matrix_from_quat(cube.data.root_quat_w.torch)
    extent=torch.bmm(box_rotation.transpose(1,2),cube_rotation).abs().sum(-1)*.025
    whole_xy=(rel[:,0].abs()+extent[:,0]<.095)&(rel[:,1].abs()+extent[:,1]<.070)
    bottom=rel[:,2]-extent[:,2]
    near_floor=whole_xy & (bottom>=-.002) & (bottom<=.035)
    if machine(env).reward_mode=='overlap':
        near_floor=((cp[:,2]-bp[:,2])<.065)&((cp[:,2]-bp[:,2])>.020)&(box_distance<.035)
    command=env.action_manager.action[:,5]
    previous_command=env.action_manager.prev_action[:,5]
    opening=((command>previous_command+1e-5) | (command>.84)) & (.6*(command+1)>jaw+.005)
    # Lost-contact grace permits release only within the actual box, never on table.
    released_zone = inside | near_floor
    on_table=(height<.44) if machine(env).reward_mode=='overlap' else table_slide_mask(height,whole_xy,bottom)
    gripper_out=gripper_outside_camera_view(env,margin=.02)
    cube_out=object_outside_camera_view(env,margin=.01)
    return dict(bilateral=bilateral,
        fixed_contact_in_pad=contact['fixed_contact_in_pad'],
        moving_contact_in_pad=contact['moving_contact_in_pad'],
        fixed_contact_force_N=contact['fixed_force_N'],
        moving_contact_force_N=contact['moving_force_N'],
        fixed_contact_force_x_N=contact['fixed_force_vector_N'][:,0],
        fixed_contact_force_y_N=contact['fixed_force_vector_N'][:,1],
        fixed_contact_force_z_N=contact['fixed_force_vector_N'][:,2],
        moving_contact_force_x_N=contact['moving_force_vector_N'][:,0],
        moving_contact_force_y_N=contact['moving_force_vector_N'][:,1],
        moving_contact_force_z_N=contact['moving_force_vector_N'][:,2],
        opposing_contact_forces=contact['opposing_forces'],
        contact_force_cosine=contact['force_cosine'],
        jaw=jaw, height=height, speed=speed,
        gripper_out_of_view=gripper_out,cube_out_of_view=cube_out,
        robot_dropped=(robot.data.root_pos_w.torch[:,2]-origin[:,2]<.20),
        clearance=clearance, table_force=force,
        unsafe=(clearance<.003)|(force>.1)|(height<.35)
               |gripper_out|cube_out,
        enclosed=(xy<.03)&(offset[:,2].abs()<.035)&(direction>.8),
        reach_ready=(pd<.04)&(direction>.75)&(clearance>=.004),
        pregrasp_distance=pd, grasp_pose_distance=target_distance, xy_distance=xy,
        grasp_target_position=grasp_target,
        grasp_target_xy_distance=target_xy,grasp_target_height_error=target_error[:,2],
        tool_height_above_cube=tool[:,2]-cp[:,2],
        alignment=alignment, grasp_alignment=grasp_alignment, downward_cosine=downward_cosine,
        approach_from_above=above,
        grasp_approach_quality=torch.exp(-torch.square(xy/.05))*direction,
        enclosure=enclosure,
        on_table=on_table,
        slide_speed=torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch[:,:2],dim=-1)*on_table,
        box_distance=box_distance,
        lowering=torch.where(box_distance<.035,(1-(cp[:,2]-bp[:,2]-.029).abs()/.15).clamp(0,1),0.),
        over_box=whole_xy,descent=((LIFT_MIN-height)/(LIFT_MIN-(bp[:,2]-origin[:,2]+.029))).clamp(0,1),
        inside_box=released_zone,opening_for_release=opening,
        placed=inside & (jaw>1.10) & (distance>.065) & ~bilateral,
        cube_xy=cp[:,:2])


def tick(env):
    state = machine(env)
    if state.last != env.common_step_counter:
        m = measurements(env)
        m['timed_out'] = env.episode_length_buf >= env.max_episode_length - 1
        travel=torch.linalg.vector_norm(m['cube_xy']-state.previous_xy,dim=-1)
        state.slide_distance += torch.where(state.cube_seen & m['on_table'],travel,0.)
        state.previous_xy[:]=m['cube_xy'];state.cube_seen[:]=True
        m['slide_distance']=state.slide_distance
        state.update(m, env.step_dt,inspection=getattr(env.cfg,'manual_no_termination',False))
        state.last = env.common_step_counter
        state.measurements = m
        if hasattr(env,'extras'):
            log=env.extras.setdefault('log',{})
            for i,name in enumerate(NAMES):
                log['FourStage/'+name+'_fraction']=(state.stage==i).float().mean()
            log['FourStage/slide_distance_m']=state.slide_distance.mean()
            log['Grasp/ik_target_distance_cm']=m['grasp_pose_distance'].mean()*100
            log['Grasp/ik_target_xy_error_cm']=m['grasp_target_xy_distance'].mean()*100
            log['Grasp/ik_target_height_error_cm']=m['grasp_target_height_error'].abs().mean()*100
            log['Grasp/descent_unlocked_fraction']=state.descent_ready.float().mean()
            log['Grasp/jaws_open_fraction']=(m['jaw']>1.10).float().mean()
            for i,terms in enumerate(TERMS):
                for j,name in enumerate(terms):
                    log['FourStageReward/'+name]=state.group_contribution[:,i,j].mean()
        import os
        if os.environ.get('SO101_RECORD_SUCCESSES','0')=='1':
            from isaac_vision_rl.utils.success_videos import record_training_step
            record_training_step(env,state)
        if os.environ.get('SO101_LIVE_REWARD_GRAPHS','0')=='1' or os.environ.get('SO101_REWARD_WINDOW','0')=='1':
            from isaac_vision_rl.utils.training_reward_charts import update_training_charts
            update_training_charts(env,state)
    return state


def completed(env):
    return tick(env).success


def failed(env):
    return tick(env).failure


def phase_observation(env):
    return torch.nn.functional.one_hot(machine(env).stage,4).float()


def critic_observation(env):
    cube,box,robot = (env.scene[n] for n in ('cube','box','robot'))
    body=robot.find_bodies('gripper_frame_link')[0][0]
    origins=env.scene.env_origins
    origins=origins.torch if hasattr(origins,'torch') else origins
    s=machine(env)
    result=torch.cat((cube.data.root_pos_w.torch-origins,box.data.root_pos_w.torch-origins,
        robot.data.body_pos_w.torch[:,body]-origins,cube.data.root_lin_vel_w.torch,
        phase_observation(env),s.dwell[:,None],s.lost[:,None],s.best),dim=-1)
    if s.reward_mode=="bounded":
        result=torch.cat((result,s.grasp_best,s.contact_dwell.clamp(max=.2)[:,None]),dim=-1)
    return result


def configure(cfg):
    # Requested easier-simulation experiment: 10% lower gravity and cube friction.
    # Explicit assignments are idempotent if configuration is applied twice.
    cfg.sim.gravity = (0.0, 0.0, -8.829)
    cfg.scene.cube.spawn.physics_material.static_friction = 1.35
    cfg.scene.cube.spawn.physics_material.dynamic_friction = 1.08
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    from isaac_vision_rl.utils.four_stage_terms import StageReward
    @configclass
    class Rewards:
        slot_1=RewardTermCfg(func=StageReward,weight=1.,params={'slot':0})
        slot_2=RewardTermCfg(func=StageReward,weight=1.,params={'slot':1})
        slot_3=RewardTermCfg(func=StageReward,weight=1.,params={'slot':2})
    cfg.rewards=Rewards()
    cfg.terminations.placed=TerminationTermCfg(func=completed)
    cfg.terminations.stage_failure=TerminationTermCfg(func=failed)
    if hasattr(cfg.terminations,'lift_success'): cfg.terminations.lift_success=None
    cfg.observations.policy.phase=ObservationTermCfg(func=phase_observation)
    cfg.observations.critic.task_state=ObservationTermCfg(func=critic_observation)
    # Deterministic closed-jaw home at initial spawn and every episode reset.
    cfg.scene.robot.init_state.joint_pos["gripper"] = 0.0
    cfg.episode_length_s=45.
    cfg.actions.arm_action.max_target_speed=.5
    cfg.actions.arm_action.max_target_acceleration=1.
    cfg.actions.arm_action.wrist_roll_speed=.2
    cfg.actions.gripper_action.max_target_speed=.45
    cfg.actions.gripper_action.max_target_acceleration=.9
    cfg.events.reset_cube_pose.params['pose_range']={}
    cfg.events.reset_box_pose.params['pose_range']={}
    cfg.events.randomize_cube_mass=None
    cfg.events.randomize_cube_friction=None
