"""Graphical Newton reward dashboard and simulation-only joint jogging."""
from array import array

from isaac_vision_rl.experiments.simple_rewards.stages import (
    NAMES, TERMS, WEIGHTS, CONDITIONS, LIFT_RISE_MIN_M, LIFT_RISE_MAX_M,
    active_reward_groups,
)

DISPLAY_NAMES = {
 'reach_distance':'Lower Cube Between Fingers', 'point_down':'Point Fingers Down',
 'open_jaws':'Keep Jaws Open During Descent', 'table_clearance':'Too Close to Table', 'fixed_finger_near_side':'Fixed Finger Near Cube Side',
 'moving_finger_near_opposite_side':'Moving Finger Near Opposite Side', 'table_dragging':'Dragging Cube',
 'lift_height':f'Lift to {LIFT_RISE_MIN_M*100:.0f}-{LIFT_RISE_MAX_M*100:.0f} cm', 'cube_steadiness':'Keep Cube Steady',
 'lift_speed':'Moving Cube Too Fast', 'carry_to_box':'Move Cube Toward Box',
 'lower_into_box':'Lower Cube Into Box', 'task_result':'Task Success or Failure',
}
def display_name(name): return DISPLAY_NAMES.get(name,name.replace('_',' ').title())
def visible_reward_terms(snapshot):
    """Display only the current stage, including the frame where it advances."""
    stage=int(snapshot.get('stage',snapshot.get('reward_stage',0)))
    return stage,TERMS[stage]


CUMULATIVE_TERMS={name:(name+'_total',name+'_running_total')
 for name in ('reach_distance','point_down','fixed_finger_near_side','moving_finger_near_opposite_side',
              'open_jaws','lift_height','cube_steadiness','carry_to_box','lower_into_box')}


class RewardPanel:
    def __init__(self,names):
        self.names=names;self.velocity=[0.]*6;self.speed=.25
        self.paused=True;self.reset_requested=False;self.stop_requested=False;self.locked=False
        self.reason='Ready: press Run, then hold a joint +/- button to jog.'
        self.snapshot=None;self.total=0.;self.joints=[0.]*6;self.draw_count=0
        self.history=[0.]*100
        self.term_history={name:[] for terms in TERMS for name in terms}
        self.full_history={name:array('f') for name in self.term_history}
        self.show_full_history=False
        self.live_window=150
        self.sample_count=0;self.smoothing=.6;self.follow=True;self.active_only=True
        self.chart_note='Each chart is one reward or penalty. No mixed-stage curves.'
        self.pose_draft=None
        self.pending_pose=None
        self.joint_limits=[(-3.14,3.14)]*5+[(0.,1.2)]
        self.hold_gripper=False
        self.gripper_hold_target=None
        self.approach_running_total=0.
        self.open_jaws_running_total=0.
        self.enclosure_running_total=0.
        for _,attribute in CUMULATIVE_TERMS.values():setattr(self,attribute,0.)

    def record(self,snapshot):
        self.sample_count+=1
        stage=snapshot['reward_stage']
        active=({name:snapshot['group_contribution'][i][j]
                 for i,terms in enumerate(TERMS) for j,name in enumerate(terms)}
                if 'group_contribution' in snapshot else dict(zip(TERMS[stage],snapshot['contribution'])))
        for name,history in self.term_history.items():
            # Pregrasp charts display exact per-episode reward totals, so
            # no movement means a flat line, rather than dropping back to zero.
            if name in CUMULATIVE_TERMS:
                key,attribute=CUMULATIVE_TERMS[name]
                if key in snapshot:
                    value=float(snapshot[key])
                else:
                    value=getattr(self,attribute)+float(active.get(name,0.))
                setattr(self,attribute,value)
            else:
                value=active.get(name,0.)
            history.append(float(value))
            self.full_history[name].append(history[-1])
            del history[:-3000]

    def clear_history(self):
        self.history=[0.]*100
        self.approach_running_total=0.
        self.open_jaws_running_total=0.
        self.enclosure_running_total=0.
        for _,attribute in CUMULATIVE_TERMS.values():setattr(self,attribute,0.)
        for name in self.term_history:
            self.term_history[name]=[]
            self.full_history[name]=array("f")
        self.sample_count=0

    def draw_joints(self,ui):
        ui.separator_text('JOINT JOG | Hold +/-; release to hold')
        _,self.speed=ui.slider_float('Speed (rad/s)',self.speed,.05,.5)
        for i,name in enumerate(self.names):
            ui.text(f'{name}: {self.joints[i]:+.3f} rad');ui.same_line()
            ui.button(f'-##joint{i}')
            if ui.is_item_active() and not self.locked:self.velocity[i]=-self.speed
            ui.same_line();ui.button(f'+##joint{i}')
            if ui.is_item_active() and not self.locked:self.velocity[i]=self.speed

    def draw_coordinated_pose(self,ui):
        ui.separator_text('COORDINATED POSE | SIMULATION ONLY')
        ui.text_wrapped('Copy the current pose, edit several targets, then Apply to move them together. Run enables motion.')
        if ui.button('Copy current pose'):
            self.pose_draft=list(self.joints)
        if self.pose_draft is not None:
            for i,name in enumerate(self.names):
                lo,hi=self.joint_limits[i]
                _,self.pose_draft[i]=ui.slider_float('Target '+name,self.pose_draft[i],lo,hi)
            if ui.button('Apply all joint targets') and not self.locked:
                self.pending_pose=list(self.pose_draft)
        if ui.button('Release gripper hold' if self.hold_gripper else 'Hold current gripper target'):
            self.hold_gripper=not self.hold_gripper
            self.gripper_hold_target=None  # Runner captures its commanded target.
        ui.text('Gripper target: HELD' if self.hold_gripper else 'Gripper target: editable')
        ui.text_wrapped('Holding a jaw target does not attach the cube or guarantee a grasp. STOP cancels pending motion and gripper hold.')

    def chart_samples(self,name):
        """Default to a moving live window; retain compact session history."""
        import numpy as np
        if self.show_full_history:
            values=np.frombuffer(self.full_history[name],dtype=np.float32)
            start=0
        else:
            values=np.asarray(self.term_history[name][-self.live_window:],dtype=np.float32)
            start=max(0,self.sample_count-len(values))
        if len(values)<2:
            values=np.array([0.,float(values[-1]) if len(values) else 0.])
        indices=np.arange(len(values))
        # Full-session display uses min/max buckets to preserve penalty spikes
        # without drawing or smoothing hundreds of thousands of points per frame.
        if len(values)>5000:
            width=(len(values)+2499)//2500
            selected={0,len(values)-1}
            for offset in range(0,len(values),width):
                block=values[offset:offset+width]
                selected.add(offset+int(block.argmin()))
                selected.add(offset+int(block.argmax()))
            indices=np.array(sorted(selected))
        return (indices+start).astype(np.float64),values[indices].astype(np.float64)

    def plot_term(self,ui,name,weight):
        import numpy as np
        from imgui_bundle import implot
        xs,history=self.chart_samples(name)
        smooth=history.copy()
        for i in range(1,len(smooth)):
            smooth[i]=self.smoothing*smooth[i-1]+(1-self.smoothing)*history[i]
        signed=name in ('supported_lift_progress','lowering_progress')
        color=(.25,.8,.45,1.) if weight>=0 else (.95,.3,.3,1.)
        ui.text_colored(ui.ImVec4(*color),display_name(name))
        distance_key={'fixed_finger_near_side':'fixed_finger_distance','moving_finger_near_opposite_side':'moving_finger_distance'}.get(name)
        measurements=self.snapshot.get('measurements',{}) if self.snapshot else {}
        if distance_key in measurements:ui.text(f"Distance to target area: {measurements[distance_key]*1000:.1f} mm")
        ui.text(f'Last: {history[-1]:+.5f} | weight {weight:+.3f} | '+('Reward / too-high penalty' if signed else 'Reward' if weight>=0 else 'Penalty'))
        if implot.begin_plot('##chart_'+name,ui.ImVec2(-1.,170.)):
            y_label='Cumulative weighted reward' if name in CUMULATIVE_TERMS else 'Weighted reward'
            implot.setup_axes('Simulation step',y_label,0,implot.AxisFlags_.auto_fit if self.follow else 0)
            implot.setup_axis_limits(implot.ImAxis_.x1,float(xs[0]),float(max(xs[-1],xs[0]+1)),
                                     ui.Cond_.always if self.follow else ui.Cond_.once)
            implot.plot_line('Raw',xs,history,implot.Spec(line_color=ui.ImVec4(*color[:3],.25),line_weight=1.))
            implot.plot_line('Smoothed',xs,smooth,implot.Spec(line_color=ui.ImVec4(*color),line_weight=2.))
            implot.end_plot()

    def draw_charts(self,ui,standalone=False):
        from imgui_bundle import implot
        if implot.get_current_context() is None:
            self._implot_context=implot.create_context()
            implot.set_imgui_context(ui.get_current_context())
        viewport=ui.get_main_viewport()
        if standalone:
            ui.set_next_window_pos(viewport.pos,ui.Cond_.always)
            ui.set_next_window_size(viewport.size,ui.Cond_.always)
            flags=ui.WindowFlags_.no_title_bar|ui.WindowFlags_.no_move|ui.WindowFlags_.no_resize|ui.WindowFlags_.no_collapse
        else:
            ui.set_next_window_pos((max(650.,viewport.size.x-710.),30.),ui.Cond_.first_use_ever)
            ui.set_next_window_size((690.,min(920.,viewport.size.y-60.)),ui.Cond_.first_use_ever)
            flags=0
        opened,_=ui.begin('Reward charts | TensorBoard style',flags=flags)
        if opened:
            if standalone:
                ui.text_colored(ui.ImVec4(.25,.86,.92,1.),'SO-101   /   LEARNING OBSERVATORY')
                ui.same_line(max(400.,ui.get_window_width()-110.))
                if ui.small_button('Close charts'):
                    from imgui_bundle import hello_imgui
                    hello_imgui.get_runner_params().app_shall_exit=True
                ui.text_disabled('LIVE TELEMETRY   /   ENVIRONMENT 00   /   READ ONLY')
                ui.separator()
                if self.snapshot:
                    current=self.snapshot['stage']
                    for i,name in enumerate(NAMES):
                        if i:ui.same_line()
                        color=(.25,.86,.92,1.) if i==current else (.32,.70,.52,1.) if i<current else (.43,.48,.57,1.)
                        ui.text_colored(ui.ImVec4(*color),f'{i+1:02d}  {name}'+('  /  ACTIVE' if i==current else ''))
                ui.spacing()
            else:ui.text_wrapped(self.chart_note)
            if self.snapshot and self.snapshot.get('reward_style')=='continuous':
                ui.text_disabled('One score per graph. Better: up. Worse: down. Still: flat. New episode: zero.')
                ui.text_wrapped('Progress charts show reward earned this episode. Stationary: flat. Penalties show their current cost.')
            if self.snapshot and 'grasp_bonus_used' in self.snapshot:
                ui.text_disabled('Lift target: cube centre 53–55 cm above scene origin. Higher is penalized. Then lower into box.')
                budget=self.snapshot['grasp_bonus_used']
                ui.text_disabled(f'Grasp bonus earned: position {budget[0]*100:.0f}% | touch {budget[1]*100:.0f}%. Holding longer earns no extra bonus.')
            _,self.smoothing=ui.slider_float('Smoothing',self.smoothing,0.,.99)
            _,self.show_full_history=ui.checkbox('Show full history',self.show_full_history)
            if not self.show_full_history:
                _,window=ui.slider_float('Live window (samples)',float(self.live_window),30.,600.)
                self.live_window=int(window)
            ui.text_disabled('Full history: since this chart opened; large histories preserve min/max peaks.')
            _,self.follow=ui.checkbox('Follow latest steps',self.follow)
            stage,terms=visible_reward_terms(self.snapshot)
            ui.text_disabled(f'Showing only the current stage: Stage {stage} ({NAMES[stage]}).')
            ui.text_wrapped('Uncheck Follow to pan/zoom. Scroll to zoom, drag to pan. Raw is faint; smoothed is bold.')
            ui.separator_text(NAMES[stage])
            for i,name in enumerate(terms):
                self.plot_term(ui,name,self.snapshot.get("reward_weights",WEIGHTS)[stage][i])
        ui.end()

    @staticmethod
    def bar(ui,label,fraction,detail,color=(.2,.65,.9,1.)):
        ui.text(label)
        ui.push_style_color(ui.Col_.plot_histogram,ui.ImVec4(*color))
        ui.progress_bar(max(0.,min(1.,float(fraction))),(-1.,18.),detail)
        ui.pop_style_color()

    def draw(self,ui):
        self.draw_count+=1
        ui.set_next_window_size((570.,850.),ui.Cond_.first_use_ever)
        opened,_=ui.begin('SO-101 manual control and live rewards')
        self.velocity=[0.]*6
        if opened:
            ui.text_colored(ui.ImVec4(.3,.85,.95,1.),'SIMULATION ONLY  |  No policy learning')
            ui.text_wrapped(self.reason)
            if ui.button('STOP / HOLD'):
                self.stop_requested=True;self.paused=True
                self.pending_pose=None;self.hold_gripper=False;self.gripper_hold_target=None
            ui.same_line()
            if ui.button('Pause' if not self.paused else 'Run') and not self.locked:self.paused=not self.paused
            ui.same_line()
            if ui.button('Reset to HOME'):
                self.reset_requested=True;self.pending_pose=None;self.pose_draft=None
                self.hold_gripper=False;self.gripper_hold_target=None
            self.draw_joints(ui)
            self.draw_coordinated_pose(ui)
            if self.snapshot:
                s=self.snapshot;stage=s['stage'];rs=s['reward_stage'];m=s['measurements']
                ui.separator_text('TASK PROGRESS')
                if ui.begin_table('stage_track',4):
                    for i,name in enumerate(NAMES):
                        ui.table_next_column()
                        self.bar(ui,f'{i+1} {name}',1. if i<stage else .5 if i==stage else 0.,
                            'DONE' if i<stage else 'ACTIVE' if i==stage else 'WAIT',
                            (.2,.75,.45,1.) if i<stage else (.2,.6,.95,1.))
                    ui.end_table()
                ui.text_wrapped(CONDITIONS[stage])
                ui.text('Checks: cube between open fingers '+str(bool(m.get('reach_ready',False)))
                    +' | enclosed '+str(bool(m.get('enclosed',False)))
                    +' | bilateral '+str(bool(m.get('bilateral',False)))
                    +' | unsafe '+str(bool(m.get('unsafe',False))))
                if m.get('unsafe',False):
                    ui.text_wrapped('Stage advance blocked: clearance, table force, camera boundary or dropped cube check is unsafe.')
                self.bar(ui,'Required continuous dwell',s.get('dwell_fraction',0.),
                         f'{s.get("dwell_fraction",0.)*100:.0f}%')
                ui.separator_text('ACTIVE REWARD FAMILIES')
                groups=active_reward_groups(stage)
                all_values=s.get('group_contribution')
                scale=max(.001,max(abs(x) for row in all_values for x in row)) if all_values else max(.001,max(abs(x) for x in s['contribution']))
                for group in groups:
                    ui.text(NAMES[group])
                    for i,name in enumerate(TERMS[group]):
                        value=all_values[group][i] if all_values else (s['contribution'][i] if group==rs else 0.)
                        color=(.25,.8,.45,1.) if value>=0 else (.95,.3,.3,1.)
                        self.bar(ui,display_name(name),abs(value)/scale,
                            f'weighted reward {value:+.4f}',color)
                ui.text('Bars show relative contribution magnitude; values are actual reward.')
                ui.text(f'STEP {s["step_reward"]:+.4f}       EPISODE RETURN {self.total:+.3f}')
                import numpy as np
                ui.plot_lines('Step reward',np.array(self.history,dtype=np.float32),graph_size=(-1.,45.))
                ui.separator_text('CONTACT AND CLEARANCE')
                if ui.begin_table('contact_cards',2):
                    for label,key,target,unit in (
                        ('Table clearance','clearance',.008,'m'),
                        ('Cube height','height',.50,'m'),
                        ('Table force','table_force',.1,'N'),
                        ('Cube sliding','slide_distance',.01,'m')):
                        ui.table_next_column();value=m.get(key,0.)
                        danger=(value<.003 if key=='clearance' else value>target if key in ('table_force','slide_distance') else False)
                        self.bar(ui,label,value/target,f'{value:.4f} {unit}',
                            (.95,.3,.3,1.) if danger else (.25,.7,.6,1.))
                    ui.end_table()
                for key in ('fixed_finger_contact_N','moving_finger_contact_N'):
                    self.bar(ui,key.replace('_',' '),m.get(key,0.)/4.,f'{m.get(key,0.):.3f} N')
                self.bar(ui,'Opposing two-finger contact',m.get('bilateral',0.),
                         'CONTACT' if m.get('bilateral',0.) else 'NO BILATERAL CONTACT')
            if self.snapshot and ui.collapsing_header('All numeric diagnostics'):
                for key,value in self.snapshot['measurements'].items():ui.text(f'{key}: {value}')
        ui.end()
        if self.snapshot:self.draw_charts(ui)
