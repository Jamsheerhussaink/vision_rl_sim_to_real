"""Graphical Newton reward dashboard and simulation-only joint jogging."""
from array import array

from isaac_vision_rl.utils.four_stage import (
    NAMES, TERMS, WEIGHTS, CONDITIONS, LIFT_RISE_MIN_M, LIFT_RISE_MAX_M,
    active_reward_groups,
)

DISPLAY_NAMES = {
    'overhead_approach': 'Approach distance',
    'table_clearance': 'Table clearance',
    'wrist_down': 'Wrist downward',
    'jaws_open': 'Jaws open',
    'centered_xy': 'Centered XY',
    'camera_visibility': 'Camera margin',
    'static_finger_reach': 'Fixed Finger Near Cube Side',
    'moving_finger_reach': 'Moving Finger Near Opposite Side',
    'current_enclosure': 'Cube enclosure',
    'bilateral_contact': 'Valid two-finger grasp',
    'table_slide_speed': 'Table dragging',
    'top_push': 'Pushing down on cube top',
    'top_approach_push': 'Downward approach on cube top',
    'side_push': 'Pushing cube sideways',
    'supported_lift_progress': 'Lift reward',
    'cube_rise_cm': 'Cube height gained (cm)',
    'held_lift_dwell': 'Stable held lift',
    'object_speed': 'Cube speed during Lift',
    'box_approach_progress': 'Carry toward box',
    'lowering_progress': 'Lower into box',
    'settled_release': 'Successful placement',
}


def display_name(name): return DISPLAY_NAMES.get(name, name.replace('_', ' ').title())


CUMULATIVE_TERMS={
    'overhead_approach':('approach_progress_total','approach_running_total'),
    'gripper_readiness':('readiness_progress_total','readiness_running_total'),
    'descend_to_pinch_progress':('descend_progress_total','descend_running_total'),
    'current_enclosure':('enclosure_progress_total','enclosure_running_total'),
    'static_finger_reach':('fixed_finger_progress_total','fixed_finger_running_total'),
    'moving_finger_reach':('moving_finger_progress_total','moving_finger_running_total'),
}


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
        self.episode_total_view=True
        self.step_history={name:[] for name in self.term_history}
        self.step_full_history={name:array('f') for name in self.term_history}
        self.reward_totals={name:0. for name in self.term_history}
        self._record_episode_step=None
        self.reset_samples=[]
        self.term_history['cube_rise_cm']=[]
        self.full_history['cube_rise_cm']=array('f')
        self.live_window=150
        self.sample_count=0;self.smoothing=.6;self.follow=True;self.active_only=True
        self.chart_note='Each chart is one reward or penalty. No mixed-stage curves.'
        self.pose_draft=None
        self.pending_pose=None
        self.joint_limits=[(-3.14,3.14)]*5+[(0.,1.2)]
        self.hold_gripper=False
        self.gripper_hold_target=None
        self.approach_running_total=0.
        self.descend_running_total=0.
        self.readiness_running_total=0.  # retained for the original reward profile
        self.enclosure_running_total=0.
        self.grasp_target_running_total=0.
        self._chart_layout_initialized=False
        self.fixed_finger_running_total=0.
        self.moving_finger_running_total=0.

    def record(self,snapshot):
        self.sample_count+=1
        stage=snapshot['reward_stage']
        active=({name:snapshot['group_contribution'][i][j]
                 for i,terms in enumerate(TERMS) for j,name in enumerate(terms)}
                if 'group_contribution' in snapshot else dict(zip(TERMS[stage],snapshot['contribution'])))
        age=snapshot.get('episode_step')
        if age is not None and self._record_episode_step is not None and (age < self._record_episode_step or age == 0):
            self.reward_totals=dict.fromkeys(self.reward_totals,0.)
            self.reset_samples.append(self.sample_count-1)
        self._record_episode_step=age
        for name,history in self.term_history.items():
            if name=='cube_rise_cm':
                history.append(float(snapshot.get('cube_rise_cm',0.)))
                self.full_history[name].append(history[-1])
                del history[:-3000]
                continue
            step=float(active.get(name,0.))
            self.step_history[name].append(step)
            self.step_full_history[name].append(step)
            del self.step_history[name][:-3000]
            self.reward_totals[name]+=step
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
                value=self.reward_totals[name]
            history.append(float(value))
            self.full_history[name].append(history[-1])
            del history[:-3000]

    def clear_history(self):
        self.history=[0.]*100
        self.approach_running_total=0.
        self.descend_running_total=0.
        self.readiness_running_total=0.
        self.enclosure_running_total=0.
        self.grasp_target_running_total=0.
        self.fixed_finger_running_total=0.
        self.moving_finger_running_total=0.
        for name in self.term_history:
            self.term_history[name]=[]
            self.full_history[name]=array("f")
        self.sample_count=0
        self.reset_samples=[]
        self.reward_totals=dict.fromkeys(self.reward_totals,0.)
        self._record_episode_step=None
        for name in self.step_history:
            self.step_history[name]=[]
            self.step_full_history[name]=array('f')

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
        totals=self.episode_total_view or name=='cube_rise_cm'
        full=self.full_history if totals else self.step_full_history
        recent=self.term_history if totals else self.step_history
        if self.show_full_history:
            values=np.frombuffer(full[name],dtype=np.float32)
            start=0
        else:
            values=np.asarray(recent[name][-self.live_window:],dtype=np.float32)
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

    def plot_term(self,ui,name,weight,compact=False,plot_height=None):
        import numpy as np
        from imgui_bundle import implot
        xs,history=self.chart_samples(name)
        smooth=history.copy()
        for i in range(1,len(smooth)):
            smooth[i]=self.smoothing*smooth[i-1]+(1-self.smoothing)*history[i]
        signed=name in ('wrist_down','jaws_open','centered_xy','static_finger_reach','moving_finger_reach','overhead_approach','descend_to_pinch_progress',
                        'grasp_target_progress','supported_lift_progress','lowering_progress')
        color=(.25,.8,.45,1.) if weight>=0 else (.95,.3,.3,1.)
        ui.text_colored(ui.ImVec4(*color),display_name(name))
        value_label='Height (cm)' if name=='cube_rise_cm' else 'Episode total' if self.episode_total_view else 'Latest step'
        if name=='cube_rise_cm':
            ui.text_wrapped(f'Height: {history[-1]:.1f} cm | target {LIFT_RISE_MIN_M*100:g}-{LIFT_RISE_MAX_M*100:g} cm | Measurement only')
        else:
            ui.text_wrapped(f'{value_label}: {history[-1]:+.5f} | weight {weight:+.3f} | '+('Signed progress' if signed else 'Reward' if weight>=0 else 'Penalty'))
        # Two-column stage cards keep all terms visible together in the live
        # monitor. The grid is a presentation change only; samples and rewards
        # are unchanged.
        if plot_height is None:
            plot_height=170. if compact else 190.
        if implot.begin_plot('##chart_'+name,ui.ImVec2(-1.,plot_height)):
            y_label='Height (cm)' if name=='cube_rise_cm' else 'Episode total' if self.episode_total_view else 'Step reward'
            implot.setup_axes('Simulation step',y_label,0,implot.AxisFlags_.auto_fit if self.follow else 0)
            implot.setup_axis_format(implot.ImAxis_.y1,'%.3g')
            implot.setup_axis_limits(implot.ImAxis_.x1,float(xs[0]),float(max(xs[-1],xs[0]+1)),
                                     ui.Cond_.always if self.follow else ui.Cond_.once)
            if name=='cube_rise_cm':
                for label,level in (('Target minimum',LIFT_RISE_MIN_M*100),('Target maximum',LIFT_RISE_MAX_M*100)):
                    implot.plot_line(label,np.array([xs[0],xs[-1]]),np.array([level,level]))
            # Mark actual attempt boundaries; do not interpret reset drops as
            # negative policy feedback. This is display-only.
            for reset_x in self.reset_samples:
                if xs[0]<=reset_x<=xs[-1]:
                    lo,hi=float(history.min()),float(history.max())
                    if hi==lo:hi=lo+1e-6
                    implot.plot_line('New attempt',np.array([float(reset_x),float(reset_x)]),np.array([lo,hi]))
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
            width=min(viewport.size.x-40.,max(1000.,viewport.size.x*.65))
            if not self._chart_layout_initialized:
                ui.set_next_window_pos((viewport.pos.x+viewport.size.x-width-20.,viewport.pos.y+20.),ui.Cond_.always)
                ui.set_next_window_size((width,viewport.size.y-40.),ui.Cond_.always)
                self._chart_layout_initialized=True
            flags=0
        opened,_=ui.begin('Live stage rewards | Grid',flags=flags)
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
            if self.snapshot and 'episode_step' in self.snapshot:
                status=('SUCCESS' if self.snapshot.get('success') else
                        'FAILED - reset pending' if self.snapshot.get('failed') else 'Attempt running')
                grip='confirmed' if self.snapshot.get('grasp_confirmed') else 'not confirmed'
                ui.text_wrapped(f'Episode step: {self.snapshot["episode_step"]} | Grasp: {grip} | {status}')
            if self.snapshot and 'grasp_checks' in self.snapshot:
                checks=self.snapshot['grasp_checks']
                labels=(('fixed_touch','Fixed finger touches cube'),('moving_touch','Moving finger touches cube'),
                        ('enclosed','Cube enclosed'),
                        ('jaw_ready','Jaw angle'),('safe','Table/camera safe'))
                ui.text_wrapped(' | '.join(f'{label}: {"OK" if checks[key] else "WAIT"}' for key,label in labels))
                ui.text_wrapped(f'Diagnostics only: fixed pad {checks["fixed_pad"]} | moving pad {checks["moving_pad"]} | opposite forces {checks["opposing"]}')
                ui.text_wrapped(f'Hold: {checks["held_seconds"]:.2f}/{checks["required_seconds"]:.2f} s | '
                                f'Finger forces: {checks["fixed_force_N"]:.3f}, {checks["moving_force_N"]:.3f} N | '
                                f'Jaw: {checks["jaw_rad"]:.3f} rad')
            if self.snapshot and 'grasp_checks' in self.snapshot:
                ui.text_wrapped('Lift entry: both fingers touching cube + enclosure + closed jaws + safe hold. Opposite forces/pad locations are diagnostics only.')
            if self.snapshot and 'approach_checks' in self.snapshot:
                checks=self.snapshot['approach_checks']
                ui.text_wrapped(f'Descent: {"ALLOWED" if checks["descent_allowed"] else "ALIGN ABOVE CUBE FIRST"} | '
                                f'Sideways error: {checks["xy_error_cm"]:.2f} cm (need <{checks.get("xy_tolerance_cm",3.0):.1f}) | '
                                f'Wrist down: {checks["downward_cosine"]:.2f} (need >{checks.get("downward_cosine_min",0.85):.2f}) | '
                                f'Table gap: {checks["clearance_mm"]:.1f} mm')
            if self.snapshot and self.snapshot.get('reward_style')=='continuous':
                ui.text_disabled('Distance charts are episode totals: closer = up, farther = down, still = flat.')
                ui.text_wrapped('Each episode starts at zero. Grasp, lift and place feedback stays in its own stage.')
            if self.snapshot and 'grasp_gate_checks' in self.snapshot:
                checks=self.snapshot['grasp_gate_checks']
                if 'descended_to_pinch' in checks:
                    labels=(
                        ('descended_to_pinch','Descended to pinch'),
                        ('cube_between_fingers','Cube between fingers'),
                        ('wrist_down','Wrist down'),
                        ('jaws_open','Jaws open'),
                        ('safe_clearance','Table safe'),
                    )
                else:
                    labels=(
                        ('safe_overhead','Safe overhead first'),
                        ('pinch_close','Pinch'),
                        ('cube_between_fingers','Cube between fingers'),
                        ('wrist_down','Wrist down'),
                        ('jaws_open','Jaws open'),
                        ('safe_clearance','Table safe'),
                    )
                status='  |  '.join(
                    f'{label}: {"OK" if checks.get(key,False) else "waiting"}'
                    for key,label in labels
                )
                ui.text_wrapped('REACH -> GRASP checks: '+status)
                ui.text(f'Hold all checks for 0.08 s: {self.snapshot.get("grasp_gate_dwell_fraction",0.)*100:.0f}%')
            if self.snapshot and 'grasp_bonus_used' in self.snapshot:
                ui.text_disabled(f'Lift target: {LIFT_RISE_MIN_M*100:g}-{LIFT_RISE_MAX_M*100:g} cm above starting height; hold steadily, then place.')
                budget=self.snapshot['grasp_bonus_used']
                ui.text_disabled(f'Grasp bonus earned: position {budget[0]*100:.0f}% | touch {budget[1]*100:.0f}%. Holding longer earns no extra bonus.')
            _,self.smoothing=ui.slider_float('Smoothing',self.smoothing,0.,.99)
            _,self.episode_total_view=ui.checkbox('Episode totals (uncheck for latest step rewards)',self.episode_total_view)
            ui.text_wrapped('Latest step shows the actual reward now; episode totals add rewards within each attempt. This switch does not change training.')
            _,self.show_full_history=ui.checkbox('Show full history',self.show_full_history)
            if not self.show_full_history:
                _,window=ui.slider_float('Live window (samples)',float(self.live_window),30.,600.)
                self.live_window=int(window)
            ui.text_disabled('Full history: since this chart opened; large histories preserve min/max peaks.')
            _,self.follow=ui.checkbox('Follow latest steps',self.follow)
            ui.same_line();_,self.active_only=ui.checkbox('Only active rewards',self.active_only)
            ui.text_wrapped('Uncheck Follow to pan/zoom. Scroll to zoom, drag to pan. Raw is faint; smoothed is bold. Inactive stages record zero.')
            # Show ONLY current stage rewards
            # Display the current task phase immediately. reward_stage belongs
            # to the action just scored and can lag a transition by one step.
            current_stage = self.snapshot.get('stage', self.snapshot.get('reward_stage', 0))
            if isinstance(current_stage, list):
                current_stage = current_stage[0] if current_stage else 0
            stages = [current_stage] if self.active_only else range(4)
            for stage in stages:
                ui.separator_text(f'{stage+1} {NAMES[stage]} {"- CURRENT ATTEMPT" if stage == current_stage else ""}')
                weights=self.snapshot.get("reward_weights",WEIGHTS)[stage]
                cards=list(TERMS[stage])+(['cube_rise_cm'] if stage==2 else [])
                rows=(len(cards)+1)//2
                available_height=ui.get_content_region_avail().y
                height=max(140.,min(260.,available_height/rows-62.)) if self.active_only else 170.
                if ui.begin_table(f'reward_grid_{stage}',2):
                    for i,name in enumerate(cards):
                        ui.table_next_column()
                        self.plot_term(ui,name,weights[i] if i<len(weights) else 0.,compact=True,plot_height=height)
                    ui.end_table()
                else:
                    # Fallback for an ImGui backend without table support.
                    for i,name in enumerate(cards):
                        self.plot_term(ui,name,weights[i] if i<len(weights) else 0.,compact=True)
        ui.end()

    @staticmethod
    def bar(ui,label,fraction,detail,color=(.2,.65,.9,1.)):
        ui.text(label)
        ui.push_style_color(ui.Col_.plot_histogram,ui.ImVec4(*color))
        ui.progress_bar(max(0.,min(1.,float(fraction))),(-1.,18.),detail)
        ui.pop_style_color()

    def draw(self,ui):
        self.draw_count+=1
        ui.set_next_window_size((1400.,1800.),ui.Cond_.first_use_ever)
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
            # STAGE INDICATOR - shows all 4 stages, highlights current
            if self.snapshot:
                current = self.snapshot.get('stage', self.snapshot.get('reward_stage', 0))
                if isinstance(current, list): current = current[0] if current else 0
                ui.separator_text('STAGE PROGRESS')
                for i, name in enumerate(NAMES):
                    is_active = (i == current)
                    color = ui.ImVec4(0.2, 1.0, 0.2, 1.0) if is_active else ui.ImVec4(0.6, 0.6, 0.6, 1.0)
                    prefix = "▶ " if is_active else "  "
                    ui.text_colored(color, f'{prefix}{i+1}. {name}')
                ui.spacing()
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
                ui.text('Checks: open pregrasp '+str(bool(m.get('reach_ready',False)))
                    +' | enclosed '+str(bool(m.get('enclosed',False)))
                    +' | bilateral '+str(bool(m.get('bilateral',False)))
                    +' | unsafe '+str(bool(m.get('unsafe',False))))
                if m.get('unsafe',False):
                    ui.text_wrapped('Stage advance blocked: clearance, table force, camera boundary or dropped cube check is unsafe.')
                self.bar(ui,'Required continuous dwell',s.get('dwell_fraction',0.),
                         f'{s.get("dwell_fraction",0.)*100:.0f}%')
                ui.separator_text('ACTIVE REWARD FAMILIES')
                groups=active_reward_groups(rs)
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
