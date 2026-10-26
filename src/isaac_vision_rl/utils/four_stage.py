"""Versioned four-stage task. Three terms per reward family, with explicit overlapping activation."""
import math
import os
import torch


def practical_cube_contact(measurements):
    """Both filtered finger-to-cube forces, without pad-location/angle gates.

    Geometry, jaw closure, safety and acquisition dwell are checked separately.
    Legacy synthetic measurements can only use their existing strict predicate.
    """
    if all(key in measurements for key in ('fixed_contact_force_N','moving_contact_force_N')):
        return ((measurements['fixed_contact_force_N'] > .02)
                & (measurements['moving_contact_force_N'] > .02))
    return measurements['bilateral']

VERSION = ('four_stage_v38_later_stage_priority_overlap' if os.environ.get('SO101_GRASP_REWARD_MODE','bounded')=='overlap' and os.environ.get('SO101_REWARD_STYLE','progress')=='continuous'
           else 'four_stage_v37_high_overhead_approach_overlap_progress' if os.environ.get('SO101_GRASP_REWARD_MODE','bounded')=='overlap'
           else 'four_stage_v38_later_stage_priority' if os.environ.get('SO101_REWARD_STYLE','progress')=='continuous'
           else 'four_stage_v37_high_overhead_approach_progress')
# Cube-centre height in metres relative to environment origin, not lift distance.
# Box rim is about 0.491 m; 5 cm cube bottom clears it in this provisional band.
LIFT_MIN=.53
LIFT_MAX=.55
LIFT_RISE_MIN_M=float(os.environ.get('SO101_LIFT_RISE_MIN','0.10'))
LIFT_RISE_MAX_M=float(os.environ.get('SO101_LIFT_RISE_MAX','0.12'))
LIFT_HOLD_SECONDS=float(os.environ.get('SO101_LIFT_HOLD_SECONDS','0.5'))
GRASP_HOLD_SECONDS=float(os.environ.get('SO101_GRASP_HOLD_SECONDS','0.3'))
PREGRASP_CLEARANCE_M=.08
DESCENT_XY_TOLERANCE_M=.020
DESCENT_DOWNWARD_COSINE_MIN=.90
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
    """Score cube rise from its reset height, with a preferred 10-12 cm band."""
    rising=(lift_rise/LIFT_RISE_MIN_M).clamp(0,1)
    falling=(1-(lift_rise-LIFT_RISE_MAX_M).clamp(min=0)/.05).clamp(0,1)
    return rising*falling


def table_slide_mask(height,whole_box_xy,box_bottom):
    """Height proxy for source tabletop motion, excluding cube inside box floor region."""
    within_box=whole_box_xy & (box_bottom>=-.002) & (box_bottom<=.035)
    return (height<.44) & ~within_box


def top_descent_push(tool_height_above_cube,tool_down_speed,distance,jaw):
    """Flag a near-cube downward approach only after the jaws have started closing."""
    return ((tool_height_above_cube>.03) & (tool_down_speed>.02)
            & (distance<.10) & (jaw<1.10))


def pinch_reach_ready(between_fingers,target_distance,downward,clearance,jaw):
    """Reach ends at the open pinch pose, not at the overhead waypoint."""
    return between_fingers & (target_distance<.035) & (downward>.75) & (clearance>=.004) & (jaw>1.10)


def descent_alignment_ready(target_xy_distance,downward_cosine):
    """Centre on the asymmetric pinch target and point nearly straight down."""
    return ((target_xy_distance<DESCENT_XY_TOLERANCE_M)
            & (downward_cosine>DESCENT_DOWNWARD_COSINE_MIN))


NAMES = ('REACH', 'GRASP', 'LIFT', 'PLACE')
TERMS = (
    ('overhead_approach', 'table_clearance', 'wrist_down', 'jaws_open', 'centered_xy', 'camera_visibility'),
    ('static_finger_reach', 'moving_finger_reach', 'current_enclosure', 'bilateral_contact', 'table_slide_speed'),
    ('supported_lift_progress', 'held_lift_dwell', 'object_speed'),
    ('box_approach_progress', 'lowering_progress', 'settled_release'),
)
JAWS_OPEN_WEIGHT = float(os.environ.get("SO101_JAWS_OPEN_WEIGHT", "0.25"))
APPROACH_WEIGHT = float(os.environ.get("SO101_APPROACH_WEIGHT", "0.10"))
LIFT_HEIGHT_WEIGHT = float(os.environ.get("SO101_LIFT_HEIGHT_WEIGHT", "0.25"))
STABLE_LIFT_WEIGHT = float(os.environ.get("SO101_STABLE_LIFT_WEIGHT", "0.25"))
PLACE_CARRY_WEIGHT = float(os.environ.get("SO101_PLACE_CARRY_WEIGHT", "0.25"))
PLACE_LOWER_WEIGHT = float(os.environ.get("SO101_PLACE_LOWER_WEIGHT", "0.25"))
PLACE_SUCCESS_WEIGHT = float(os.environ.get("SO101_PLACE_SUCCESS_WEIGHT", "4.0"))
WEIGHTS = ((.5, -1., .5, JAWS_OPEN_WEIGHT, .1, -1.), (1.5, 1.5, 4., 12., -4.), (12., 8., -1.), (10., 6., 40.))
CONTINUOUS_WEIGHTS = ((APPROACH_WEIGHT, -1., .05, JAWS_OPEN_WEIGHT, .1, -1.), (.05, .05, .10, .30, -4.), (LIFT_HEIGHT_WEIGHT, STABLE_LIFT_WEIGHT, -1.), (PLACE_CARRY_WEIGHT, PLACE_LOWER_WEIGHT, PLACE_SUCCESS_WEIGHT))
CONDITIONS = (
    'Cube between open fingers at pinch height, wrist down and table safe, for 0.15 s',
    f'Cube enclosed, opposing two-finger contact and closed jaws for {GRASP_HOLD_SECONDS:g} s',
    f'Cube rises {LIFT_RISE_MIN_M:.2f}–{LIFT_RISE_MAX_M:.2f} m from reset, supported, speed <0.25 m/s, for {LIFT_HOLD_SECONDS:g} s',
    'Whole cube settled in box, released, tool withdrawn >6.5 cm, for 0.4 s',
)


def active_reward_groups(stage):
    """Zero-based stage -> active reward families (including their penalties)."""
    return (0, 1) if stage <= 1 else (1, 2) if stage == 2 else (1, 3)


class StageMachine:
    """Pure tensor state machine; simulator data enters only through measurements."""
    def __init__(self, count, device, reward_mode=None, goal=None, reward_style=None, lift_hold_seconds=None):
        self.reward_mode=reward_mode or os.environ.get("SO101_GRASP_REWARD_MODE","bounded")
        self.reward_style=reward_style or os.environ.get("SO101_REWARD_STYLE","progress")
        if self.reward_style not in ("continuous","progress"):raise ValueError("Invalid reward style")
        self.weights=CONTINUOUS_WEIGHTS if self.reward_style=="continuous" else WEIGHTS
        self.goal=goal or os.environ.get("SO101_TRAINING_GOAL","place")
        self.lift_hold_seconds=LIFT_HOLD_SECONDS if lift_hold_seconds is None else float(lift_hold_seconds)
        if not math.isfinite(self.lift_hold_seconds) or self.lift_hold_seconds<=0:
            raise ValueError('Lift hold duration must be finite and positive')
        if self.reward_mode not in ("bounded","overlap") or self.goal not in ("lift","place"):
            raise ValueError("Invalid grasp reward mode or training goal")
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.reward_stage = self.stage.clone()
        self.dwell = torch.zeros(count, device=device)
        self.lost = self.dwell.clone()
        self.best = torch.zeros(count, 2, device=device)
        # Previous distance, hand readiness, and enclosure for signed feedback.
        self.previous = torch.zeros(count, 3, device=device)
        self.previous_posture = torch.zeros(count, 3, device=device)
        # Reach has two ordered subgoals: raise to cube clearance, then center XY.
        self.reach_route = torch.zeros(count, dtype=torch.long, device=device)
        self.reach_route_dwell = torch.zeros(count, device=device)
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        self.cube_start_height = torch.zeros(count, device=device)
        self.cube_start_height_valid = torch.zeros(count, dtype=torch.bool, device=device)
        # Keep proof of a successful lift after the cube has been lowered into
        # the box. Placement should not require the cube to remain raised.
        self.lift_verified = torch.zeros(count, dtype=torch.bool, device=device)
        self.success = self.started.clone()
        self.failure = self.started.clone()
        self.outcome_paid = self.started.clone()
        self.descent_ready = self.started.clone()
        self.grasp_best = torch.zeros(count,2,device=device)
        self.contact_dwell = torch.zeros(count,device=device)
        self.raw = torch.zeros(count, 6, device=device)
        self.contribution = self.raw.clone()
        # Reward families keep independent graph columns; six is the largest stage.
        self.group_raw = torch.zeros(count, 4, 6, device=device)
        self.group_contribution = self.group_raw.clone()
        self.last = None
        self.slide_distance = self.dwell.clone()
        self.cube_seen = self.started.clone()
        self.previous_xy = torch.zeros(count,2,device=device)
        self.finger_sides_valid=torch.zeros(count,dtype=torch.bool,device=device)
        self.finger_axis=torch.zeros(count,dtype=torch.long,device=device)
        self.finger_sign=torch.ones(count,device=device)
        self.previous_finger_distance=torch.zeros(count,2,device=device)
        self.finger_distance_valid=torch.zeros(count,dtype=torch.bool,device=device)
        self.acquisition_dwell=torch.zeros(count,device=device)
        self.using_descent_target=torch.zeros(count,dtype=torch.bool,device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in (self.stage, self.reward_stage, self.dwell, self.lost, self.best,
                      self.previous, self.previous_posture, self.started, self.success, self.failure, self.raw, self.contribution,
                      self.group_raw, self.group_contribution, self.grasp_best, self.contact_dwell, self.outcome_paid, self.descent_ready):
            value[ids] = 0
        self.cube_start_height[ids] = 0
        self.cube_start_height_valid[ids] = False
        self.lift_verified[ids] = False
        self.reach_route[ids] = 0
        self.reach_route_dwell[ids] = 0
        self.slide_distance[ids]=0
        self.cube_seen[ids]=False
        self.finger_sides_valid[ids]=False
        self.finger_axis[ids]=0
        self.finger_sign[ids]=1.
        self.previous_finger_distance[ids]=0.
        self.finger_distance_valid[ids]=False
        self.acquisition_dwell[ids]=0.
        self.using_descent_target[ids]=False
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
        route_reset = in_reach & (self.reach_route == 1) & (tool_height < PREGRASP_CLEARANCE_M - .02) & (xy_reach >= .025)
        self.reach_route[:] = torch.where(route_reset, 0, self.reach_route)
        self.reach_route_dwell[:] = torch.where(route_reset, 0., self.reach_route_dwell)
        route_changed = route_switch | route_reset
        active = ~(self.success | self.failure)
        # Practical acquisition accepts imperfect contact angles/locations.
        # Both sensors are filtered to the cube; enclosure and dwell still apply.
        # Retention permits tilt after acquisition, but still needs both contacts.
        nearby=m.get('tool_cube_distance',torch.full_like(m['height'],float('inf'))) < .06
        retained=(old>=2) & nearby
        m['both_fingers_touching_cube']=practical_cube_contact(m)
        grasp = m['both_fingers_touching_cube'] & (m['enclosed'] | retained) & (m['jaw'] < 1.10)
        m['grasp_valid']=grasp
        acquiring=(old<=1) & grasp & ~m['unsafe'] & active
        self.acquisition_dwell[:]=torch.where(acquiring,self.acquisition_dwell+dt,0.)
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
        if self.reward_style=='continuous':
            # In the continuous experiment, make a 1 cm overshoot clearly
            # negative and cap the cost after 4 cm. This keeps the same lift
            # reward term and weight while teaching the arm to stop near 12 cm.
            too_high=((lift_rise-LIFT_RISE_MAX_M)/.01).clamp(0,4)
        release_zone = m['inside_box'] & (m['jaw'] > 1.10)
        # Reward mode chooses checkpoint/reward compatibility, not release
        # safety. Slow intentional opening over the actual box floor is valid
        # for a fine-tuned older actor as well as for a new bounded actor.
        release_zone |= (old==3) & m['inside_box'] & m.get('opening_for_release',torch.zeros_like(grasp))
        lost = (old >= 2) & ~grasp & ~release_zone
        self.lost[:] = torch.where(lost, self.lost + dt, 0.)
        if inspection:
            self.failure.zero_()  # Inspection may recover after an unsafe pose/lost grasp.
        else:
            self.failure |= m['unsafe'] | ((self.lost >= .2) & (old >= 2))
        # A direct valid grasp also proves the cube has been reached. Requiring
        # open jaws exclusively would strand a closed grasp in REACH.
        # Placement occurs after lowering; require a verified earlier lift,
        # not that the cube is still 10-12 cm above its starting height.
        placed_ok = m['placed'] & self.lift_verified
        good = torch.where(old == 0, m['reach_ready'] | grasp,
                   torch.where(old == 1, grasp,
                   torch.where(old == 2, held, placed_ok))) & ~self.failure & ~m['unsafe']
        self.dwell[:] = torch.where(good, self.dwell + dt, 0.)
        durations = torch.tensor([0.15, GRASP_HOLD_SECONDS, self.lift_hold_seconds, .4], device=old.device)[old]
        complete = (self.dwell >= durations) & active & ~self.failure
        self.lift_verified |= complete & (old == 2)
        # Progress only pays for a new episode/stage best. No reward on first measurement.
        score = torch.stack((
            torch.where(old == 0, (1-m['pregrasp_distance']/.30).clamp(0,1),
                torch.where(old == 1, m['enclosure'],
                torch.where(old == 2, torch.where(grasp, height_score, 0.),
                            (1-m['box_distance']/.30).clamp(0,1)))),
            torch.where(old == 0, m['alignment'],
                torch.where(old == 1, (self.dwell/.2).clamp(0,1),
                torch.where(old == 2, (self.dwell/self.lift_hold_seconds).clamp(0,1), m['lowering']))),
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
        above=m.get('approach_from_above',torch.ones_like(m['pregrasp_distance']))
        cosine=m.get('downward_cosine',torch.ones_like(above)).clamp(-1.,1.)
        xy=m.get('xy_distance',m['pregrasp_distance'])
        newly_overhead=torch.zeros_like(reach)
        aligned_for_descent=descent_alignment_ready(m.get('grasp_target_xy_distance',xy),cosine)
        if self.reward_style=='continuous':
            overhead=(m['pregrasp_distance']<.040)&aligned_for_descent&(tool_height>=PREGRASP_CLEARANCE_M-.005)&(m['jaw']>1.10)
            newly_overhead=overhead & ~self.descent_ready & reach
            self.descent_ready |= overhead & reach
        raise_distance=(PREGRASP_CLEARANCE_M-tool_height).clamp(min=0.)
        reach_distance=torch.where(self.reach_route==0,raise_distance,m['pregrasp_distance'])
        goal_distance=torch.where(old==0,reach_distance,grasp_pose_distance)
        target_changed=torch.zeros_like(reach)
        if self.reward_style=='continuous':
            # Open at the overhead waypoint before descending to the IK pinch
            # target. Keep the raise waypoint while Reach is still raising.
            overhead_distance=torch.where(old==0,reach_distance,m['pregrasp_distance'])
            # Strict alignment starts descent. A wider retention band prevents
            # target chatter from small tracking errors during a valid descent.
            xy_error=m.get('grasp_target_xy_distance',xy)
            jaws_fully_open=m['jaw']>1.10
            # Keep the approach open-jaw all the way through descent. If the
            # policy starts closing early, return its target to the overhead
            # waypoint until it opens again; closing belongs to Grasp.
            retain_descent=self.using_descent_target & (xy_error<.030) & (cosine>.85) & jaws_fully_open
            use_descent=self.descent_ready & (aligned_for_descent | retain_descent) & jaws_fully_open
            target_changed=use_descent!=self.using_descent_target
            self.using_descent_target[:]=use_descent
            goal_distance=torch.where(use_descent,grasp_pose_distance,overhead_distance)
        # Reset the distance baseline at the raise->3D-center switch so changing
        # subgoals cannot create a fake reward spike.
        reset_baseline=first_in_stage | route_changed | newly_overhead | target_changed
        previous_distance=torch.where(reset_baseline,goal_distance,self.previous[:,0])
        distance_progress=(previous_distance-goal_distance)/.05
        grasp_approach=m.get('grasp_approach_quality',torch.ones_like(above))
        approach_quality=torch.where(old==0,
            torch.where(self.reach_route==0,torch.ones_like(above),above),grasp_approach)
        # During Reach, only credit target progress while the gripper stays
        # fully open. This prevents a partly closed gripper from earning the
        # same approach reward while it presses on the cube.
        reach_jaws_open=(m['jaw']>1.10).to(approach_quality.dtype)
        approach_quality=torch.where(old==0,approach_quality*reach_jaws_open,approach_quality)
        # Separate gripper readiness components (all from measurements)
        wrist_down = m.get('wrist_down', m.get('downward_cosine', torch.ones_like(above)).clamp(0,1))
        jaws_open = m.get('jaws_open', gated_open_jaw_quality(cosine,above,m['jaw']))
        centered_xy = m.get('centered_xy', torch.exp(-m.get('xy_distance', m.get('pregrasp_distance', torch.zeros_like(above))) / 0.03))
        # Readiness combines downward wrist, open jaws, and overhead alignment
        readiness=torch.where(old==0,.5*(wrist_down+jaws_open),wrist_down)
        if self.reward_style=='continuous':
            # A bounded [0,1] posture score: centered and down, with hand height
            # and aligned open jaws. After descent unlocks, stop asking for open
            # jaws and measure height relative to the pinch pose instead.
            height_error=torch.where(self.descent_ready,
                m.get('grasp_target_height_error',m.get('tool_height_above_cube', torch.zeros_like(jaws_open))-.0095),
                m.get('tool_height_above_cube', torch.zeros_like(jaws_open))-PREGRASP_CLEARANCE_M)
            height_quality=torch.exp(-torch.square(height_error/.04))
            jaw_quality=jaws_open
            readiness=torch.exp(-torch.square(m.get('xy_distance', torch.zeros_like(jaws_open))/.05))*m.get('downward_cosine', torch.ones_like(jaws_open)).clamp(0.,1.)*(.5+.25*height_quality+.25*jaw_quality)
        # Readiness has the same goal across raise->center, so retain its
        # baseline there; only a new stage/pinch target changes its definition.
        previous_readiness=torch.where(first_in_stage | newly_overhead,readiness,self.previous[:,1])
        readiness_progress=readiness-previous_readiness
        self.group_raw[:,0,0]=distance_progress*approach_quality*reach
        self.group_raw[:,0,2]=readiness_progress*reach
        self.previous[:,0]=torch.where(reach,goal_distance,self.previous[:,0])
        self.previous[:,1]=torch.where(reach,readiness,self.previous[:,1])
        # Encourage a safe raise while still in Reach. Once the aligned descent
        # target is unlocked, stop paying a clearance cost for approaching the
        # table; hard table-contact safety termination remains active all the
        # way through grasp.
        clearance_shaping=( (.020-m['clearance'])/.020 ).clamp(0,1)
        clearance_shaping_active=(old==0) & ~self.descent_ready
        self.group_raw[:,0,1]=clearance_shaping*dt*clearance_shaping_active
        # A separate, gentle Reach penalty warns before the hard camera guard.
        # It is zero well inside the frame and grows smoothly near its edges.
        camera_cost=m.get('camera_visibility_cost',torch.zeros_like(m['height']))
        self.group_raw[:,0,5]=camera_cost.clamp(0,1)*dt*(old==0)
        # Grasp quality/contact shaping runs through grasp and lift, then stops
        # in placement so waiting while carrying cannot farm grasp reward.
        # GRASP has 5 terms: static_finger_reach, moving_finger_reach, current_enclosure, bilateral_contact, table_slide_speed
        if self.reward_mode=='bounded':
            # Pay only new best enclosure and continuous contact, at most 4+8
            # points per episode. Releasing/regrasping never refills this budget.
            safe_grasp=grasp & ~m['unsafe'] & grasp_reward_active & active
            self.contact_dwell[:]=torch.where(safe_grasp,self.contact_dwell+dt,0.)
            quality=torch.stack((m['enclosure'].clamp(0,1),
                                 (self.contact_dwell/.2).clamp(0,1)),dim=-1)
            quality*= (grasp_reward_active & active & ~m['unsafe'])[:,None]
            # Track best for enclosure (idx 2) and bilateral_contact (idx 3)
            self.group_raw[:,1,:2]=(quality-self.grasp_best).clamp(0,1)
            self.grasp_best[:]=torch.maximum(self.grasp_best,quality)
        else:
            self.group_raw[:,1,0] = m['enclosure'].clamp(0,1)*dt*grasp_reward_active
            self.group_raw[:,1,1] = grasp.float()*dt*grasp_reward_active
        # Route all Stage 2 costs through its negative-weight penalty term.
        # Subtracting from every raw term inverted the sign on this negative-
        # weight channel, accidentally paying the agent for a push event.
        push_cost=(.8*m['top_push'].float()+.5*m['top_approach_push'].float()
                   +.5*m['side_push'].float())
        self.group_raw[:,1,4] = ((m['slide_speed']/.05).clamp(0,1)+push_cost)*dt*grasp_active
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
            # Signed progress: closer is positive, stationary is zero, and
            # moving farther from the active target is negative. Do not clamp
            # at zero or add a dwell/hover penalty. At the instant the overhead
            # latch changes targets, reset the baseline without a fake jump.
            # Preserve the direction and open-jaw gates above in continuous
            # mode. Previously this assignment overwrote the gated value.
            # Gate positive approach credit by route/alignment/open-jaw quality,
            # but keep signed retreat negative even when the pose is incorrect.
            gated_approach=torch.where(distance_progress>0,
                distance_progress*approach_quality,distance_progress)
            self.group_raw[:,0,0]=gated_approach*reach
            # Keep each displayed term independent. No posture income for
            # waiting, and no combined readiness score labelled as wrist angle.
            posture=torch.stack((wrist_down.clamp(0,1),jaws_open.clamp(0,1),centered_xy.clamp(0,1)),dim=-1)
            baseline=torch.where(first_in_stage[:,None],posture,self.previous_posture)
            posture_delta=posture-baseline
            self.group_raw[:,0,2]=posture_delta[:,0]*reach
            self.group_raw[:,0,3]=posture_delta[:,1]*(old==0)
            center_gate=torch.where(old==0,
                torch.where(self.reach_route==0,torch.zeros_like(above),above*reach_jaws_open),
                torch.ones_like(above))
            self.group_raw[:,0,4]=posture_delta[:,2]*reach*center_gate
            self.previous_posture[:]=posture
            enclosure=m['enclosure'].clamp(0,1)
            previous_enclosure=torch.where(first_in_stage,enclosure,self.previous[:,2])
            # Partial enclosure without contact must not pay for hovering.
            # During Lift a real grasp can still receive the listed hold rates.
            # GRASP has 5 terms: static_finger_reach, moving_finger_reach, current_enclosure, bilateral_contact, table_slide_speed
            static_reach = m.get('static_finger_reach', torch.zeros_like(grasp))
            moving_reach = m.get('moving_finger_reach', torch.zeros_like(grasp))
            self.group_raw[:,1,0]=static_reach.clamp(0,1)*dt*grasp_reward_active
            self.group_raw[:,1,1]=moving_reach.clamp(0,1)*dt*grasp_reward_active
            self.group_raw[:,1,2]=torch.where(reach,enclosure-previous_enclosure,enclosure*dt)*grasp_reward_active
            self.previous[:,2]=enclosure
            self.group_raw[:,1,3]=grasp.float()*dt*grasp_reward_active
            self.group_raw[:,1,4]=((m['slide_speed']/.05).clamp(0,1)+push_cost)*dt*grasp_active
            self.group_raw[:,2,0]=(torch.where(grasp,height_score,0.)-too_high)*dt*(old==2)
            self.group_raw[:,2,1]=held.float()*dt*(old==2)
            over_box=m.get('over_box',m['inside_box'])
            valid_carry=grasp & (lift_rise>=LIFT_RISE_MIN_M)
            box_gate=(valid_carry | over_box) & (lift_rise<=LIFT_RISE_MAX_M)
            self.group_raw[:,3,0]=(1-m['box_distance']/.30).clamp(0,1)*dt*(old==3)*box_gate
            descent=m.get('descent',((LIFT_MIN-m['height'])/.10).clamp(0,1))
            self.group_raw[:,3,1]=(descent*over_box-too_high)*dt*(old==3)
        # Independent side-surface progress replaces centre-proximity income.
        # Reset the baseline on stage entry; never pay for initializing targets.
        if 'fixed_finger_distance' in m and 'moving_finger_distance' in m:
            distances=torch.stack((m['fixed_finger_distance'],m['moving_finger_distance']),dim=-1)
            baseline=torch.where((~self.finger_distance_valid | first_in_stage)[:,None],distances,self.previous_finger_distance)
            finger_progress=(baseline-distances)/.05
            dragging=m.get('on_table',torch.ones_like(grasp)) & (m['slide_speed']>.01)
            finger_progress=torch.where(dragging[:,None] & (finger_progress>0),0.,finger_progress)
            self.group_raw[:,1,:2]=finger_progress*((old==1) & ~grasp)[:,None]
            self.previous_finger_distance[:]=distances
            self.finger_distance_valid[:]=True
        # Do not encourage closing while the robot is still reaching/opening.
        self.group_raw[:,1,:4]*=(old>=1)[:,None]
        # Terminal outcome: configured one-time success reward, -0.1 for an
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
        if self.reward_style!="continuous":self.group_raw[:,0,2]=0.
        self.group_raw[:]=torch.where((self.failure | m['unsafe'])[:,None,None] & (self.group_raw>0),0.,self.group_raw)
        self.group_raw *= active[:,None,None]
        # Apply weights per stage (different term counts) - build padded tensor manually
        padded_weights = torch.zeros(4, 6, device=old.device)
        for i, w in enumerate(self.weights):
            padded_weights[i, :len(w)] = torch.tensor(w, device=old.device)
        # Pushing an ungrasped cube toward the hand is not approach/grasp
        # progress. Keep retreat penalties and table-dragging costs, and keep
        # baselines updated above so pushing cannot bank a later reward jump.
        ungrasped_drag=(m.get('on_table',torch.ones_like(grasp))
                        & (m['slide_speed']>.01) & ~grasp)
        no_push_credit=ungrasped_drag[:,None,None] & (padded_weights>0) & (self.group_raw>0)
        self.group_raw[:]=torch.where(no_push_credit,0.,self.group_raw)
        self.group_contribution[:] = self.group_raw * padded_weights
        self.raw[:] = self.group_raw.sum(1)
        self.contribution[:] = self.group_contribution.sum(1)
        terminal_stage=2 if self.goal=="lift" else 3
        self.success |= complete & (old == terminal_stage)
        advance = complete & (old < terminal_stage)
        self.stage[advance] += 1
        self.dwell[advance] = 0
        self.started[advance] = False
        # Preserve real contact dwell across Reach -> Grasp. A grasp already
        # confirmed during Reach must not wait for a second contact timer.
        acquired=(old<=1) & (self.acquisition_dwell>=GRASP_HOLD_SECONDS-1e-6) & active & ~self.failure
        self.stage[acquired]=2
        self.dwell[acquired]=0.
        self.started[acquired]=False
        self.best[advance] = 0
        return self.contribution.sum(-1)


def machine(env):
    if not hasattr(env, '_four_stage'):
        env._four_stage = StageMachine(env.num_envs, env.device)
    return env._four_stage


def measurements(env):
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact_diagnostics, cube_in_box, FINGER_CONTACT_REGIONS
    from isaaclab.utils.math import matrix_from_quat
    from isaac_vision_rl.utils.hand_table import hand_table_metrics
    from isaac_vision_rl.tasks.pickplace.mdp.terminations import gripping_area_outside_camera_view, object_outside_camera_view
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
    # Estimate the center of each inner finger pad from its calibrated local
    # contact patch. These positions let the stage gate check that the cube is
    # physically between the jaws, rather than merely close to the wrist.
    pad_centers=[]
    for _, body_name, lower, upper in FINGER_CONTACT_REGIONS:
        finger_id=robot.find_bodies(body_name)[0][0]
        local_center=torch.tensor(
            [(lo+hi)*.5 for lo,hi in zip(lower,upper,strict=True)],
            device=cp.device,dtype=cp.dtype,
        ).expand(env.num_envs,-1)
        finger_rotation=matrix_from_quat(robot.data.body_quat_w.torch[:,finger_id])
        pad_centers.append(
            robot.data.body_pos_w.torch[:,finger_id]
            + torch.bmm(finger_rotation,local_center.unsqueeze(-1)).squeeze(-1)
        )
    pinch_center=(pad_centers[0]+pad_centers[1])*.5
    jaw_axis=pad_centers[1]-pad_centers[0]
    jaw_gap=torch.linalg.vector_norm(jaw_axis,dim=-1).clamp(min=1e-6)
    jaw_axis=jaw_axis/jaw_gap[:,None]
    cube_from_pinch=cp-pinch_center
    cube_rotation=matrix_from_quat(cube.data.root_quat_w.torch)
    cube_upright=torch.max(cube_rotation[:,2,:].abs(),dim=-1).values
    cube_width=(.05*torch.bmm(
        cube_rotation.transpose(1,2),jaw_axis.unsqueeze(-1)
    ).squeeze(-1).abs().sum(-1))
    cube_between_fingers=(
        (torch.linalg.vector_norm(cube_from_pinch,dim=-1)<.035)
        & ((cube_from_pinch*jaw_axis).sum(-1).abs()+cube_width*.5<jaw_gap*.5)
    )
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import FingerSurfaces
    if not hasattr(env,'_opposite_finger_surfaces'):
        env._opposite_finger_surfaces=FingerSurfaces(robot.cfg.spawn.usd_path,cp.device,cp.dtype)
    finger_surfaces=env._opposite_finger_surfaces.measure(robot,cp,cube_rotation,machine(env))
    static_finger_reach = torch.clamp((0.12 - finger_surfaces['fixed_finger_distance']) / 0.12, min=0.0, max=1.0)
    moving_finger_reach = torch.clamp((0.12 - finger_surfaces['moving_finger_distance']) / 0.12, min=0.0, max=1.0)
    fixed_body=robot.find_bodies('gripper_link')[0][0]
    fixed_rotation=matrix_from_quat(robot.data.body_quat_w.torch[:,fixed_body])
    grasp_target=ik_grasp_target(cp,fixed_rotation)
    target_error=tool-grasp_target
    target_distance=torch.linalg.vector_norm(target_error,dim=-1)
    target_xy=torch.linalg.vector_norm(target_error[:,:2],dim=-1)
    downward_cosine=(-tool_rotation[:,2,2]).clamp(-1.,1.)
    direction=downward_cosine.clamp(0.,1.)
    # Both reward waypoints use the same calibrated XY offset. Otherwise the
    # policy is told to centre on one point and descend toward another.
    pre = grasp_target.clone()
    pre[:,2]=cp[:,2]+PREGRASP_CLEARANCE_M
    pd = torch.linalg.vector_norm(pre-tool,dim=-1)
    above = approach_from_above_quality(xy,tool[:,2]-cp[:,2])
    # Separate gripper_readiness components
    wrist_down = direction  # downward cosine clamped to [0,1]
    jaws_open = gated_open_jaw_quality(downward_cosine,above,jaw)
    centered_xy = torch.exp(-xy / 0.03)  # XY distance to cube
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
    command=env.action_manager.action[:,5]
    previous_command=env.action_manager.prev_action[:,5]
    opening=((command>previous_command+1e-5) | (command>.84)) & (.6*(command+1)>jaw+.005)
    # Lost-contact grace permits release only within the actual box, never on table.
    released_zone = inside | near_floor
    on_table=table_slide_mask(height,whole_xy,bottom)
    gripper_out=gripping_area_outside_camera_view(env,margin=.02)
    cube_out=object_outside_camera_view(env,margin=.01)
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import gripper_camera_margin_cost
    camera_cost=gripper_camera_margin_cost(env,safe_margin=.12)
    on_table_mask = table_slide_mask(height,whole_xy,bottom)
    slide_speed=torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch[:,:2],dim=-1)*on_table_mask
    # Stronger anti-top-push: penalize ANY downward approach from above cube (not from sides)
    # Triggers when: tool above cube, approaching from above, near cube (even before fingers close)
    top_push = ((tool[:,2] - cp[:,2]) > 0.03) & (distance < 0.08) & (jaw < 1.1)
    # Read the tool's own velocity. Cube velocity here would mislabel an object
    # being lifted as a downward robot approach, and could penalize safe descent.
    tool_down_speed=(-robot.data.body_lin_vel_w.torch[:,body,2]).clamp_min(0.)
    top_approach_push=top_descent_push(tool[:,2]-cp[:,2],tool_down_speed,distance,jaw)
    # Anti-side-push: penalize sideways cube motion when fingers near cube
    side_push = (slide_speed > 0.02) & (distance < 0.08) & (jaw < 1.1)

    return dict(**finger_surfaces,
        fixed_contact_in_pad=contact['fixed_contact_in_pad'],
        moving_contact_in_pad=contact['moving_contact_in_pad'],
        # Keep the canonical bilateral contact signal available to teacher,
        # evaluator, and reward panels. It was computed above but not returned.
        bilateral=bilateral,
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
        jaw=jaw, height=height, speed=speed,tool_cube_distance=distance,
        gripper_out_of_view=gripper_out,cube_out_of_view=cube_out,
        robot_dropped=(robot.data.root_pos_w.torch[:,2]-origin[:,2]<.20),
        clearance=clearance, table_force=force,
        camera_visibility_cost=camera_cost,
        unsafe=(clearance<.003)|(force>.1)|(height<.35)
               |gripper_out|cube_out,
        enclosed=(xy<.03)&(offset[:,2].abs()<.035)&(direction>.8),
        cube_between_fingers=cube_between_fingers,
        pinch_center_distance=torch.linalg.vector_norm(cube_from_pinch,dim=-1),
        finger_gap=jaw_gap,
        cube_width_on_jaw_axis=cube_width,
        reach_ready=pinch_reach_ready(cube_between_fingers,target_distance,direction,clearance,jaw),
        pregrasp_distance=pd, grasp_pose_distance=target_distance, xy_distance=xy,
        grasp_target_position=grasp_target,
        grasp_target_xy_distance=target_xy,grasp_target_height_error=target_error[:,2],
        tool_height_above_cube=tool[:,2]-cp[:,2],
        alignment=alignment, grasp_alignment=grasp_alignment, downward_cosine=downward_cosine,
        approach_from_above=above,
        grasp_approach_quality=torch.exp(-torch.square(xy/.05))*direction,
        enclosure=enclosure,
        static_finger_reach=static_finger_reach,
        moving_finger_reach=moving_finger_reach,
        wrist_down=wrist_down,
        jaws_open=jaws_open,
        centered_xy=centered_xy,
        on_table=on_table,
        slide_speed=slide_speed,
        top_push=top_push,
        side_push=side_push,
        tool_down_speed=tool_down_speed,
        box_distance=box_distance,
        lowering=torch.where(box_distance<.035,(1-(cp[:,2]-bp[:,2]-.029).abs()/.15).clamp(0,1),0.),
        over_box=whole_xy,descent=((LIFT_MIN-height)/(LIFT_MIN-(bp[:,2]-origin[:,2]+.029))).clamp(0,1),
        inside_box=released_zone,opening_for_release=opening,
        placed=inside & (jaw>1.10) & (distance>.065) & ~bilateral,
        cube_upright=cube_upright,
        cube_xy=cp[:,:2],
        top_approach_push=top_approach_push)


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
    # The visible gripping pads matter for RGB manipulation; a cropped wrist
    # origin alone must not terminate a grasp whose fingers and cube are visible.
    from isaac_vision_rl.tasks.pickplace.mdp.terminations import gripping_area_outside_camera_view
    cfg.terminations.gripper_out_of_view.func=gripping_area_outside_camera_view
    if hasattr(cfg.terminations,'lift_success'): cfg.terminations.lift_success=None
    cfg.observations.policy.phase=ObservationTermCfg(func=phase_observation)
    cfg.observations.critic.task_state=ObservationTermCfg(func=critic_observation)
    # Deterministic open-jaw home at initial spawn and every episode reset.
    cfg.scene.robot.init_state.joint_pos["gripper"] = 1.20
    cfg.scene.robot.init_state.joint_pos["wrist_roll"] = float(os.environ.get("SO101_INITIAL_WRIST_ROLL", "0.0"))
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
