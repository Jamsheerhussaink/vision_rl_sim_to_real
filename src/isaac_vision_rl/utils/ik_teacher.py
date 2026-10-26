"""Simulation IK demonstrator. Never commands physical hardware."""
import numpy as np
from isaac_vision_rl.utils.four_stage import PREGRASP_CLEARANCE_M

PHASES=('open','reach','descend','grasp','lift','carry','lower','release','withdraw','done')


class IKTeacher:
    def __init__(self,geometry,clearance,home,cube,box,dt):
        self.geometry=geometry;self.clearance=clearance;self.home=np.asarray(home)
        self.cube=np.asarray(cube).copy();self.box=np.asarray(box).copy()
        self.place_xy=self.box[:2]-np.array([.01,.005])
        self.dt=dt;self.phase=0;self.elapsed=0.;self.dwell=0.;self.target=self.home.copy()
        self.goal=self.home.copy();self.offset=np.array([0.,0.,.012]);self.enter=True
        self.grasp_bias=None
        self.failure=None
        self.previous_actual=None

    @property
    def phase_name(self):return PHASES[self.phase]

    def _goal(self,tool,cube):
        name=self.phase_name
        pos=None;jaw=1.2
        if name=='reach':pos=self.cube+[0,0,PREGRASP_CLEARANCE_M]
        elif name in ('descend','grasp'):
            pos=self.cube+[0,0,.0095]
            if self.grasp_bias is None:
                neutral=self.geometry.solve(pos,self.target[:5],position_weight=5.)
                fixed=self.clearance.body_poses(np.r_[neutral,1.2])['gripper_link']
                # The TCP is not centred between these asymmetric fingers.
                self.grasp_bias=fixed[:3,0]*.016
            pos=pos+self.grasp_bias
        elif name=='lift':pos=np.r_[self.cube[:2],.54]+self.offset
        elif name=='carry':pos=np.r_[self.place_xy,.54]+self.offset
        elif name in ('lower','release'):pos=np.r_[self.place_xy,self.box[2]+.029]+self.offset
        elif name=='withdraw':pos=np.r_[self.place_xy,.54]+self.offset
        if name in ('grasp','lift','carry','lower'):jaw=.55
        q=self.target[:5] if pos is None else self.geometry.solve(pos,self.target[:5],position_weight=5.)
        goal=np.r_[q,jaw]
        # Source wrist/finger meshes, sampled at both jaw extremes along the move.
        self.clearance.check_segment(self.target,goal,margin=.005,samples=61)
        return goal,pos

    def step(self,actual,tool,cube,m):
        actual=np.asarray(actual);tool=np.asarray(tool);cube=np.asarray(cube)
        joint_speed=(float('inf') if self.previous_actual is None else
                     np.max(np.abs(actual[:5]-self.previous_actual[:5]))/self.dt)
        self.previous_actual=actual.copy()
        if self.failure or self.phase_name=='done':return self.target.copy()
        if m['unsafe']:
            self.failure='unsafe measured state';self.target=actual.copy();return self.target.copy()
        if self.enter:
            self.goal,self.position=self._goal(tool,cube)
            self.enter=False
        # Slow reference motion; normal simulator speed/acceleration limits still apply.
        rates=np.array([.3,.3,.3,.3,.15,.4])
        self.target+=np.clip(self.goal-self.target,-rates*self.dt,rates*self.dt)
        name=self.phase_name
        self.elapsed+=self.dt
        ready=(np.max(np.abs(actual[:5]-self.goal[:5]))<.04)
        if self.position is not None:ready &= np.linalg.norm(tool-self.position)<.008
        duration=.2
        if name=='open':ready=abs(actual[5]-1.2)<.04
        elif name=='reach':
            # The teacher's overhead waypoint precedes its separate descent.
            # RL's reach_ready now means cube BETWEEN fingers at pinch height;
            # using it here would deadlock the teacher above the cube.
            ready &= (actual[5]>1.10 and m.get('clearance',0.)>=.005
                      and m.get('downward_cosine',0.)>.95)
        elif name=='descend':
            # Passing near the target during motion is not a settled grasp pose.
            # Keep jaws open until both reference and measured arm have arrived.
            ready=(np.max(np.abs(self.target[:5]-self.goal[:5]))<.001
                   # This model tracks a settled target with a small residual.
                   # Requiring 1 cm-rad precision caused safe descents to time out
                   # while the measured tool was already within about 4.5 mm.
                   and np.max(np.abs(actual[:5]-self.goal[:5]))<.02
                   and self.position is not None
                   # Allow the measured 4-5 mm settled IK residual while still
                   # requiring the gripper to be near the requested pinch pose.
                   and np.linalg.norm(tool-self.position)<.006
                   and joint_speed<.03)
            duration=.3
        elif name=='grasp':ready=bool(m.get('bilateral',False)) and bool(m.get('enclosed',False)) and actual[5]<1.1
        elif name=='lift':ready=bool(m.get('bilateral',False)) and .53<=m.get('height',-float('inf'))<=.55 and m.get('speed',float('inf'))<.25;duration=.6
        elif name=='carry':ready &= bool(m.get('bilateral',False)) and m.get('speed',float('inf'))<.15
        elif name=='release':ready=actual[5]>1.1 and not bool(m.get('bilateral',False));duration=.4
        elif name=='withdraw':ready &= bool(m.get('placed',False));duration=.5
        self.dwell=self.dwell+self.dt if ready else 0.
        if self.dwell>=duration:
            if name=='grasp':self.offset=tool-cube
            self.phase+=1;self.elapsed=0.;self.dwell=0.;self.enter=True
        elif self.elapsed>12.:
            self.failure='phase timeout: '+name;self.target=actual.copy()
        return self.target.copy()


def offline_check(cfg):
    """Nominal kinematic checks only. No contacts, dynamics, camera or box collisions."""
    from isaac_vision_rl.utils.grasp_geometry import GraspGeometry,ARM_NAMES
    from isaac_vision_rl.utils.table_clearance import HandTableClearance
    r=cfg.scene.robot
    g=GraspGeometry(r.spawn.usd_path,r.init_state.pos,r.init_state.rot)
    c=HandTableClearance(r.spawn.usd_path,r.init_state.pos,r.init_state.rot)
    home=np.array([r.init_state.joint_pos[n] for n in ARM_NAMES]+[0.])
    cube=np.array(cfg.scene.cube.init_state.pos);cube[2]=.425
    t=IKTeacher(g,c,home,cube,cfg.scene.box.init_state.pos,1/30)
    report=[]
    for phase in range(9):
        t.phase=phase
        goal,pos=t._goal(g.forward(t.target[:5])[:3,3],cube)
        report.append(dict(phase=t.phase_name,target=goal.tolist(),tool_position=g.forward(goal[:5])[:3,3].tolist(),
                           minimum_table_clearance_m=c.check_segment(t.target,goal,margin=.005,samples=121)))
        t.target=goal
    return dict(passed=True,checks='kinematics and sampled hand/table clearance only',waypoints=report)
