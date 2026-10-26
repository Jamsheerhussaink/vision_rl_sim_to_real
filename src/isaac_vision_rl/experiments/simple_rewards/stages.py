"""Versioned four-stage task. Three terms per reward family, with explicit overlapping activation."""
import math
import os
import torch

VERSION = 'simple_rewards_v6_descend_until_cube_between_fingers'
# Cube-centre height in metres relative to environment origin, not lift distance.
# Box rim is about 0.491 m; 5 cm cube bottom clears it in this provisional band.
LIFT_MIN=.53
LIFT_MAX=.55
LIFT_RISE_MIN_M=float(os.environ.get('SO101_LIFT_RISE_MIN','0.10'))
LIFT_RISE_MAX_M=float(os.environ.get('SO101_LIFT_RISE_MAX','0.12'))
LIFT_HOLD_SECONDS=float(os.environ.get('SO101_LIFT_HOLD_SECONDS','0.0'))
PREGRASP_CLEARANCE_M=.045
OPEN_JAW_DOWN_COSINE_THRESHOLD=float(os.environ.get('SO101_OPEN_JAW_DOWN_COSINE','0.60'))
INITIAL_GRIPPER_POSITION=1.55
GRIPPER_ACTION_CLASS="isaac_vision_rl.tasks.pickplace.mdp.actions:PolicyGripperAction"


def pregrasp_target(cube_position):
    """Safe first waypoint centered 4.5 cm above the cube."""
    target=cube_position.clone()
    target[...,2]+=PREGRASP_CLEARANCE_M
    return target


def approach_from_above_quality(xy_distance, tool_height_above_cube):
    """Keep open-jaw alignment credit through descent, without favoring a hover."""
    # The previous ramp lost open-jaw credit as the gripper descended from the
    # overhead waypoint. That made the desired lowering motion reduce its score.
    height_quality=((tool_height_above_cube+.01)/.01).clamp(0,1)
    centered_quality=torch.exp(-torch.square(xy_distance/.10))
    return height_quality*centered_quality


OPEN_JAW_DOWN_COSINE_THRESHOLD = 0.60


def gated_open_jaw_quality(downward_cosine, approach_quality, jaw_position):
    """Reward open jaws when reasonably near, without requiring perfection."""
    downward = (downward_cosine - OPEN_JAW_DOWN_COSINE_THRESHOLD).clamp(0., 1.) / (1. - OPEN_JAW_DOWN_COSINE_THRESHOLD)
    open_fraction = ((jaw_position - .50) / (1.55 - .50)).clamp(0., 1.)
    return open_fraction * downward * approach_quality.clamp(0., 1.)

def lift_height_quality(lift_rise):
    """Score cube rise from its reset height, with a preferred 10-12 cm band."""
    rising=(lift_rise/LIFT_RISE_MIN_M).clamp(0,1)
    falling=(1-(lift_rise-LIFT_RISE_MAX_M).clamp(min=0)/.05).clamp(0,1)
    return rising*falling


def table_slide_mask(height,whole_box_xy,box_bottom):
    """Height proxy for source tabletop motion, excluding cube inside box floor region."""
    within_box=whole_box_xy & (box_bottom>=-.002) & (box_bottom<=.035)
    return (height<.44) & ~within_box


NAMES = ('REACH', 'GRASP', 'LIFT', 'PLACE')
TERMS = (
    ('reach_distance', 'point_down', 'open_jaws'),
    ('fixed_finger_near_side', 'moving_finger_near_opposite_side', 'table_dragging'),
    ('lift_height', 'cube_steadiness', 'lift_speed'),
    ('carry_to_box', 'lower_into_box', 'task_result'),
)
def _load_overrides():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../reward_overrides.json"))
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except Exception: pass
    return {}
_overrides = _load_overrides()
OPEN_JAWS_WEIGHT = float(os.environ.get("SO101_OPEN_JAWS_WEIGHT", _overrides.get("open_jaws", _overrides.get("jaws_open", 0.25))))
POINT_DOWN_WEIGHT = float(os.environ.get("SO101_POINT_DOWN_WEIGHT", _overrides.get("point_down", 0.25)))
REACH_DISTANCE_WEIGHT = float(os.environ.get("SO101_REACH_DISTANCE_WEIGHT", _overrides.get("reach_distance", 0.30)))
WEIGHTS = (
    (REACH_DISTANCE_WEIGHT, POINT_DOWN_WEIGHT, OPEN_JAWS_WEIGHT),
    (float(os.environ.get("SO101_GRASP_FINGER_WEIGHT", "0.35")), float(os.environ.get("SO101_GRASP_FINGER_WEIGHT", "0.35")), float(os.environ.get("SO101_GRASP_DRAG_WEIGHT", "-2.0"))),
    (.30, .20, -0.5),
    (.25, .25, 1.0)
)
CONTINUOUS_WEIGHTS = WEIGHTS
CONDITIONS = (
    'Lower with open jaws until cube is centered between the fingers at pinch height; keep safe table clearance',
    'Cube enclosed, opposing two-finger contact and closed jaws for 0.05 s',
    f'Grasped cube stays between {LIFT_RISE_MIN_M:.2f} and {LIFT_RISE_MAX_M:.2f} m rise for {LIFT_HOLD_SECONDS:.2f} s at speed <0.25 m/s',
    'Whole cube settled in box, released, tool withdrawn >6.5 cm, for 0.4 s',
)


def active_reward_groups(stage):
    """Only this stage's three terms are active; no overlapping reward families."""
    return (int(stage),)


def progress_delta(score, previous, valid, enabled):
    """No entry/gating bonus: only actual score changes receive credit."""
    return torch.where(valid & enabled, score-previous, torch.zeros_like(score))


class StageMachine:
    """Pure tensor state machine; simulator data enters only through measurements."""
    def __init__(self, count, device, reward_mode=None, goal=None, reward_style=None, lift_hold_seconds=None):
        self.reward_mode=reward_mode or os.environ.get("SO101_GRASP_REWARD_MODE","bounded")
        self.reward_style=reward_style or os.environ.get("SO101_REWARD_STYLE","progress")
        if self.reward_style != "continuous":raise ValueError("Simple reward experiment requires continuous style")
        self.weights=CONTINUOUS_WEIGHTS if self.reward_style=="continuous" else WEIGHTS
        self.goal=goal or os.environ.get("SO101_TRAINING_GOAL","place")
        self.lift_hold_seconds=LIFT_HOLD_SECONDS if lift_hold_seconds is None else float(lift_hold_seconds)
        if not math.isfinite(self.lift_hold_seconds) or self.lift_hold_seconds<0:
            raise ValueError('Lift hold duration must be finite and non-negative')
        if self.reward_mode not in ("bounded","overlap") or self.goal not in ("lift","place"):
            raise ValueError("Invalid grasp reward mode or training goal")
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.reward_stage = self.stage.clone()
        self.dwell = torch.zeros(count, device=device)
        self.lost = self.dwell.clone()
        self.best = torch.zeros(count, 2, device=device)
        # Previous simple reward scores prevent any free reward on reset.
        self.previous = torch.zeros(count, 3, device=device)
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        self.cube_start_height = torch.zeros(count, device=device)
        self.cube_start_height_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.cube_start_xy = torch.zeros(count, 2, device=device)
        self.cube_start_xy_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.success = self.started.clone()
        self.failure = self.started.clone()
        self.outcome_paid = self.started.clone()
        self.descent_ready = self.started.clone()
        self.grasp_best = torch.zeros(count,2,device=device)
        self.contact_dwell = torch.zeros(count,device=device)
        self.raw = torch.zeros(count, 3, device=device)
        self.contribution = self.raw.clone()
        self.group_raw = torch.zeros(count, 4, 3, device=device)
        self.group_contribution = self.group_raw.clone()
        self.previous_finger_jaw=torch.zeros(count,device=device)
        self.previous_finger_grasp=torch.zeros(count,dtype=torch.bool,device=device)
        self.finger_sides_valid=torch.zeros(count,dtype=torch.bool,device=device)
        self.finger_axis=torch.zeros(count,dtype=torch.long,device=device)
        self.finger_sign=torch.ones(count,device=device)
        self.simple_previous = torch.zeros(count,4,3,device=device)
        self.simple_valid = torch.zeros(count,4,3,dtype=torch.bool,device=device)
        self.last = None
        self.slide_distance = self.dwell.clone()
        self.cube_seen = self.started.clone()
        self.previous_xy = torch.zeros(count,2,device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in (self.stage, self.reward_stage, self.dwell, self.lost, self.best,
                      self.previous, self.started, self.success, self.failure, self.raw, self.contribution,
                      self.group_raw, self.group_contribution, self.grasp_best, self.contact_dwell, self.outcome_paid, self.descent_ready):
            value[ids] = 0
        self.previous_finger_jaw[ids]=0
        self.previous_finger_grasp[ids]=False
        self.finger_sides_valid[ids]=False
        self.finger_axis[ids]=0
        self.finger_sign[ids]=1
        self.simple_previous[ids] = 0
        self.simple_valid[ids] = False
        self.cube_start_height[ids] = 0
        self.cube_start_height_valid[ids] = False
        self.cube_start_xy[ids] = 0
        self.cube_start_xy_valid[ids] = False
        self.slide_distance[ids]=0
        self.cube_seen[ids]=False
        self.last = None

    def update(self, m, dt, inspection=False):
        old = self.stage.clone()
        self.reward_stage[:] = old
        first_in_stage = ~self.started
        in_reach = old == 0
        active = ~(self.success | self.failure)
        # Acquisition still requires enclosed opposing fingertip contact. After
        # its dwell has passed, real pad contact can retain a tilted cube: world
        # vertical alignment is an approach preference, not proof of retention.
        nearby=m.get('tool_cube_distance',torch.full_like(m['height'],float('inf'))) < .06
        retained=(old>=2) & nearby
        grasp = m['bilateral'] & (m['enclosed'] | retained) & (m['jaw'] < 1.35)
        # Measure lift relative to the cube's height at the beginning of this
        # episode. In real training the first observation after reset captures
        # that height; tests may provide it explicitly as cube_start_height.
        initial_height=m.get('cube_start_height',m['height'])
        capture=~self.cube_start_height_valid
        self.cube_start_height[:]=torch.where(capture,initial_height,self.cube_start_height)
        self.cube_start_height_valid|=capture
        initial_xy = m.get('cube_xy', torch.zeros(old.shape[0], 2, device=old.device))
        capture_xy = ~self.cube_start_xy_valid
        self.cube_start_xy[:] = torch.where(capture_xy[:, None], initial_xy, self.cube_start_xy)
        self.cube_start_xy_valid |= capture_xy
        lift_rise=(m['height']-self.cube_start_height).clamp(min=0.)
        m['lift_rise']=lift_rise
        # A lift only counts inside the requested height band. The stage dwell
        # below then requires this valid, slow, two-finger hold to last for the
        # configured duration; overshooting the band resets that dwell.
        held=(grasp & (lift_rise>=LIFT_RISE_MIN_M-1e-5)
              & (lift_rise<=LIFT_RISE_MAX_M+1e-5) & (m['speed']<.25))
        height_score=lift_height_quality(lift_rise)
        too_high=((lift_rise-LIFT_RISE_MAX_M)/.05).clamp(0,2)
        release_zone = m['inside_box'] & (m['jaw'] > 1.30)
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
        good = torch.where(old == 0, m['reach_ready'] | grasp,
               torch.where(old == 1, grasp,
               torch.where(old == 2, held, m['placed']))) & ~self.failure & ~m['unsafe']
        self.dwell[:] = torch.where(good, self.dwell + dt, 0.)
        durations = torch.tensor([0., float(os.environ.get("SO101_GRASP_DWELL", "0.20")), self.lift_hold_seconds, .4], device=old.device)[old]
        # Require the stage condition on this step as well. This matters when
        # the configured dwell is zero: an invalid grasp must never pass merely
        # because 0 seconds is greater than or equal to a 0-second timer.
        complete = good & (self.dwell >= durations) & active & ~self.failure
        self.started[:]=True
        self.descent_ready |= old>=1
        timed_out=m.get('timed_out',torch.zeros_like(self.failure))
        unsuccessful=(self.failure | timed_out) & ~(complete & (old==3)) & ~self.outcome_paid & active
        if inspection:unsuccessful=torch.zeros_like(unsuccessful)
        self.outcome_paid |= unsuccessful | (complete & (old==3))
        # Experimental rewards replace every legacy reward buffer. Stage checks above
        # remain unchanged. No legacy shaping or extra bonus reaches the optimizer.
        self.group_raw.zero_()
        down=m.get('downward_cosine',torch.ones_like(m['height'])).clamp(-1.,1.)
        overhead_distance=m.get('pinch_overhead_distance',m['pregrasp_distance'])
        # Align sideways first; once centered, continue down to the cube centre.
        pinch_pos = m.get('pinch_position', None)
        if pinch_pos is not None and 'cube_xy' in m:
            xy_dist_p = torch.linalg.vector_norm(pinch_pos[:, :2] - m['cube_xy'], dim=-1)
            pinch_target_z = m['height']
            z_dist_p = (pinch_pos[:, 2] - pinch_target_z).abs()
            xy_aligned_p = (1.0 - (xy_dist_p - 0.035).clamp(min=0.0) / 0.035).clamp(0.0, 1.0)
            reach_dist_score = -(xy_dist_p + z_dist_p * xy_aligned_p) / 0.05
        else:
            reach_dist_score = -overhead_distance / 0.05
        # Gate downward orientation reward so the arm cannot exploit pointing down near the base.
        # Within 12 cm of pregrasp target, orientation is rewarded independently.
        # Beyond 16 cm (near the base, ~19 cm), orientation reward drops to 0.
        near_overhead=(1.0-(overhead_distance-0.12).clamp(min=0.)/0.04).clamp(0.,1.)
        gated_down=down*near_overhead
        open_jaw_quality=gated_open_jaw_quality(
            down,m.get('approach_from_above',torch.zeros_like(down)),m['jaw'])
        # Gripper closing reward: active ONLY when the cube is properly enclosed between the fingers in grasp position.
        # Closing from 1.55 rad toward physical cube grip angle 0.78 rad yields positive progress without deducting opening credit.
        close_score = ((1.55 - m['jaw']) / (1.55 - 0.78)).clamp(0., 1.) * m.get('enclosed', torch.zeros_like(down)).float()
        # Guided release and withdrawal once settled inside box floor
        inside_box = m.get('inside_box', torch.zeros_like(down).bool())
        release_score = ((m['jaw'] - 0.78) / (1.50 - 0.78)).clamp(0., 1.) * inside_box.float()
        tool_dist = m.get('tool_cube_distance', torch.zeros_like(down))
        withdraw_score = ((tool_dist - 0.03) / 0.07).clamp(0., 1.) * (inside_box & (m['jaw'] > 1.25)).float()
        place_lowering = m.get('lowering', torch.zeros_like(down)) + 0.25 * release_score + 0.25 * withdraw_score
        scores=torch.stack((
            torch.stack((reach_dist_score,gated_down,open_jaw_quality),dim=-1),
            torch.stack((-m['fixed_finger_distance']/.05 + 0.5*close_score,-m['moving_finger_distance']/.05 + 0.5*close_score,torch.zeros_like(down)),dim=-1),
            torch.stack((lift_height_quality(lift_rise),torch.zeros_like(down),torch.zeros_like(down)),dim=-1),
            torch.stack((-m['box_distance']/.30,place_lowering,torch.zeros_like(down)),dim=-1),
        ),dim=1)
        enabled=torch.zeros_like(self.simple_valid)
        enabled[:,0,:]=(old==0)[:,None]
        enabled[:,1,:2]=((old==1)&~complete)[:,None]
        enabled[:,2,0]=((old==2)&grasp)
        enabled[:,3,0]=(old==3)&grasp
        enabled[:,3,1]=(old==3)&m.get('over_box',m['inside_box'])
        self.group_raw[:]=progress_delta(scores,self.simple_previous,self.simple_valid,enabled)
        # Brief contact still permits alignment feedback. Once already contacting,
        # deeper jaw closure must not create positive finger credit.
        extra_squeeze=grasp & self.previous_finger_grasp & (m['jaw']<self.previous_finger_jaw-1e-5)
        self.group_raw[:,1,:2]=torch.where(extra_squeeze[:,None],
            self.group_raw[:,1,:2].clamp(max=0),self.group_raw[:,1,:2])
        self.previous_finger_jaw[:]=m['jaw']
        self.previous_finger_grasp[:]=grasp
        self.simple_previous[:]=scores
        self.simple_valid[:]=enabled
        # Table dragging penalty with 3mm deadband to permit compliant grasping contact
        table_travel=m.get('table_step_distance',m['slide_speed']*dt)
        table_deadband=float(os.environ.get("SO101_TABLE_DRAG_DEADBAND", "0.0"))
        excess_slide=(table_travel - table_deadband).clamp(min=0.0)
        self.group_raw[:,1,2]=(excess_slide/(dt*.05)).clamp(0,1)*dt*(old==1)
        # Cube steadiness: per-step penalty, not progress delta, so shaking always costs
        self.group_raw[:,2,1]=-(m['speed']/.25).clamp(0,1)*dt*(old==2)
        self.group_raw[:,2,2]=((m['speed']-.25)/.25).clamp(0,1)*dt*(old==2)
        # Place outcome: +4.0 grand prize for verified placement success
        self.group_raw[:,3,2]=torch.where(unsuccessful & (old==3),-.1,((complete & (old==3)) & active).float() * 4.0)
        if self.reward_mode=='bounded':
            # A new failure cannot collect positive task credit on its terminal tick.
            self.group_raw[:]=torch.where((self.failure | m["unsafe"])[:,None,None] & (self.group_raw>0),0.,self.group_raw)
        if self.reward_style!="continuous":self.group_raw[:,0,2]=0.
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
        self.simple_valid[advance] = False
        # Reach stage safety penalties: enforce staying above table and not touching cube
        force = m.get('table_force', torch.zeros_like(down))
        clearance = m.get('clearance', torch.full_like(down, 0.05))
        xy_dist_meas = m.get('xy_distance', torch.full_like(down, 1.0))
        xy_near = (xy_dist_meas < 0.04)
        in_reach = (old == 0)
        # 1. Low fingertip penalty when not yet lined up sideways over cube (dipping < 1.5 cm)
        too_low_reach = (clearance < 0.015) & (~xy_near) & in_reach
        low_reach_penalty = -0.10 * ((0.015 - clearance) / 0.015).clamp(0, 1) * dt * too_low_reach.float()
        # 2. Table touch penalty during reach
        touch_reach_penalty = -0.15 * dt * ((clearance <= 0.002) | (force > 0.05)) * in_reach.float()
        # 3. Cube push penalty during reach (cube must not move in Stage 0)
        # Gate: only penalize when the gripper is actually close to the cube (<= 8 cm).
        # Without the gate the robot learns to avoid the cube entirely and wanders away.
        gripper_near_cube = (m.get('tool_cube_distance', torch.ones_like(down)) <= 0.08)
        cube_vel = m.get('speed', torch.zeros_like(down))
        drift = torch.linalg.vector_norm(m.get('cube_xy', self.cube_start_xy) - self.cube_start_xy, dim=-1)
        push_reach_penalty = -0.05 * (cube_vel / 0.03).clamp(0, 2) * dt * in_reach.float() * gripper_near_cube.float()
        drift_reach_penalty = -0.08 * ((drift - 0.005).clamp(min=0.0) / 0.02).clamp(0, 2) * dt * in_reach.float()
        reach_safety_penalty = low_reach_penalty + touch_reach_penalty + push_reach_penalty + drift_reach_penalty

        # Table pressing force penalty: protects physical servos against pushing into tabletop
        press_penalty = -0.05 * ((force - 0.3).clamp(min=0.0) / 2.0).clamp(0., 1.) * dt
        return self.contribution.sum(-1) + press_penalty + reach_safety_penalty


def machine(env):
    if not hasattr(env, '_four_stage'):
        env._four_stage = StageMachine(env.num_envs, env.device)
    return env._four_stage


def measurements(env):
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import bilateral_cube_contact_diagnostics, cube_in_box
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
    downward_cosine=(-tool_rotation[:,2,2]).clamp(-1.,1.)
    direction=downward_cosine.clamp(0.,1.)
    pre = pregrasp_target(cp)
    pd = torch.linalg.vector_norm(pre-tool,dim=-1)
    above = approach_from_above_quality(xy,tool[:,2]-cp[:,2])
    from isaac_vision_rl.tasks.pickplace.mdp.rewards import FINGER_CONTACT_REGIONS
    pads=[]
    for _,body_name,lower,upper in FINGER_CONTACT_REGIONS:
        index=robot.find_bodies(body_name)[0][0]
        local=(torch.tensor(lower,device=cp.device,dtype=cp.dtype)+torch.tensor(upper,device=cp.device,dtype=cp.dtype))*.5
        rotation=matrix_from_quat(robot.data.body_quat_w.torch[:,index])
        pads.append(robot.data.body_pos_w.torch[:,index]+torch.matmul(rotation,local))
    pinch=(pads[0]+pads[1])*.5
    pinch_delta=cp-pinch
    gap_vector=pads[1]-pads[0]
    gap=torch.linalg.vector_norm(gap_vector,dim=-1)
    axis=gap_vector/gap.clamp(min=1e-6)[:,None]
    cube_rot=matrix_from_quat(cube.data.root_quat_w.torch)
    cube_width=.05*torch.bmm(cube_rot.transpose(1,2),axis.unsqueeze(-1)).squeeze(-1).abs().sum(-1)
    centered=torch.linalg.vector_norm(pinch_delta,dim=-1)<.025
    projection=(pinch_delta*axis).sum(-1)
    between=projection.abs()<gap*.5
    pinch_overhead=cp.clone();pinch_overhead[:,2]+=PREGRASP_CLEARANCE_M
    bp = box.data.root_pos_w.torch
    box_distance = torch.linalg.vector_norm((cp-bp)[:,:2],dim=-1)
    inside = cube_in_box(env)
    box_rotation=matrix_from_quat(box.data.root_quat_w.torch)
    rel=torch.bmm(box_rotation.transpose(1,2),(cp-bp).unsqueeze(-1)).squeeze(-1)
    cube_rotation=matrix_from_quat(cube.data.root_quat_w.torch)
    cube_upright=torch.max(cube_rotation[:,2,:].abs(),dim=-1).values
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
    from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import FingerSurfaces
    if not hasattr(env,'_simple_finger_surfaces'):
        env._simple_finger_surfaces=FingerSurfaces(env.cfg.scene.robot.spawn.usd_path,cp.device,cp.dtype)
    finger_metrics=env._simple_finger_surfaces.measure(robot,cp,cube_rotation,machine(env))
    if os.environ.get('SO101_FINGER_GEOMETRY_AUDIT','0')=='1':
        # Check the selected mesh patch against actual reported fingertip contact points.
        from isaac_vision_rl.experiments.simple_rewards.finger_surfaces import closest_surface_point
        for label,(sensor,body_name,lower,upper) in zip(('fixed','moving'),FINGER_CONTACT_REGIONS,strict=True):
            index=robot.find_bodies(body_name)[0][0]
            points=env.scene[sensor].data.contact_pos_w.torch.reshape(env.num_envs,-1,3)
            rot=matrix_from_quat(robot.data.body_quat_w.torch[:,index])
            local=torch.bmm(points-robot.data.body_pos_w.torch[:,index,None,:],rot)
            valid=torch.isfinite(local).all(-1)&((local>=torch.tensor(lower,device=cp.device))&(local<=torch.tensor(upper,device=cp.device))).all(-1)
            selected=local[torch.arange(env.num_envs,device=cp.device),valid.long().argmax(-1)]
            err,_=closest_surface_point(torch.nan_to_num(selected),env._simple_finger_surfaces.triangles[body_name])
            finger_metrics[label+'_contact_surface_error']=torch.where(valid.any(-1),err,torch.full_like(err,-1.))
    table_clearance_limit = float(os.environ.get("SO101_TABLE_CLEARANCE_MIN", "0.0015"))
    table_force_limit = float(os.environ.get("SO101_TABLE_FORCE_MAX", "0.5"))
    table_unsafe = (
        ((clearance < table_clearance_limit) | (force > table_force_limit))
        if os.environ.get("SO101_ENABLE_TABLE_TERMINATION", "0") == "1"
        else torch.zeros_like(gripper_out)
    )
    return dict(**finger_metrics,pinch_position=pinch,fixed_pad_position=pads[0],moving_pad_position=pads[1],
        finger_gap=gap,cube_grip_width=cube_width,
        pinch_center_distance=torch.linalg.vector_norm(pinch_delta,dim=-1),
        pinch_overhead_distance=torch.linalg.vector_norm(pinch-pinch_overhead,dim=-1),
        jaw_gap_score=-(gap-(cube_width+.002)).abs()/.05,
        cube_between_fingers=between,
        bilateral=bilateral,
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
        jaw=jaw, height=height, speed=speed,tool_cube_distance=distance,
        gripper_out_of_view=gripper_out,cube_out_of_view=cube_out,
        robot_dropped=(robot.data.root_pos_w.torch[:,2]-origin[:,2]<.20),
        clearance=clearance, table_force=force,
        table_unsafe=table_unsafe,
        unsafe=table_unsafe|(height<.35)|gripper_out|cube_out,
        enclosed=centered & between & (direction>.75),
        # Advance as soon as the open, downward-facing fingers surround the
        # centered cube and maintain 4 mm table clearance. Grasp then closes.
        reach_ready=(
            centered & between & ((pinch[:,2]-cp[:,2]).abs()<=.0125)
            & (direction>.75) & (jaw>=1.50) & (clearance>=.004)
        ),
        pregrasp_distance=pd, xy_distance=xy,
        tool_height_above_cube=tool[:,2]-cp[:,2],
        downward_cosine=downward_cosine,
        approach_from_above=above,
        grasp_approach_quality=torch.exp(-torch.square(xy/.05))*direction,
        on_table=on_table,
        slide_speed=torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch[:,:2],dim=-1)*on_table,
        box_distance=box_distance,
        lowering=torch.where(box_distance<.035,(1-(cp[:,2]-bp[:,2]-.029).abs()/.15).clamp(0,1),0.),
        over_box=whole_xy,descent=((LIFT_MIN-height)/(LIFT_MIN-(bp[:,2]-origin[:,2]+.029))).clamp(0,1),
        inside_box=released_zone,opening_for_release=opening,
        placed=inside & (jaw>1.30) & (distance>.065) & ~bilateral,
        cube_upright=cube_upright,
        cube_xy=cp[:,:2])


def tick(env):
    state = machine(env)
    if state.last != env.common_step_counter:
        m = measurements(env)
        m['timed_out'] = env.episode_length_buf >= env.max_episode_length - 1
        travel=torch.linalg.vector_norm(m['cube_xy']-state.previous_xy,dim=-1)
        m['table_step_distance']=torch.where(state.cube_seen & m['on_table'],travel,0.)
        state.slide_distance += m['table_step_distance']
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
            from isaac_vision_rl.experiments.simple_rewards.charts import update_training_charts
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
    from isaac_vision_rl.experiments.simple_rewards.terms import StageReward
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
    # Keep the actor inputs identical in simulation and on the real robot:
    # RGB plus joint/controller readings. Stage remains available to the
    # training critic and reward logic, but is not given to the actor because
    # the real-camera adapter cannot yet estimate the stage reliably.
    cfg.observations.critic.task_state=ObservationTermCfg(func=critic_observation)
    # Start with open jaws and 90-degree rotated gripper; PPO learns when and how far to close them.
    cfg.scene.robot.init_state.joint_pos["gripper"] = INITIAL_GRIPPER_POSITION
    cfg.scene.robot.init_state.joint_pos["wrist_roll"] = float(os.environ.get("SO101_INITIAL_WRIST_ROLL", "0.0"))
    cfg.episode_length_s=45.
    cfg.actions.arm_action.max_target_speed=.5
    cfg.actions.arm_action.max_target_acceleration=1.
    cfg.actions.arm_action.wrist_roll_speed=.2
    cfg.actions.gripper_action.class_type=GRIPPER_ACTION_CLASS
    cfg.actions.gripper_action.max_target_speed=.45
    cfg.actions.gripper_action.max_target_acceleration=.9
    cfg.events.reset_cube_pose.params['pose_range']={}
    cfg.events.reset_box_pose.params['pose_range']={}
    cfg.events.randomize_cube_mass=None
    cfg.events.randomize_cube_friction=None
