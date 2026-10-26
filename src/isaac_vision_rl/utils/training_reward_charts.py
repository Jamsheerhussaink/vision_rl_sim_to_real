"""Read-only Newton reward charts for environment zero of a PPO run."""
from isaac_vision_rl.utils.reward_panel import RewardPanel


def grasp_check_snapshot(measurements,hold_seconds,required_seconds):
    """Expose the actual grasp blockers; these are diagnostics, not rewards."""
    m=measurements
    needed=('fixed_contact_in_pad','moving_contact_in_pad','fixed_contact_force_N',
            'moving_contact_force_N','opposing_contact_forces','enclosed','jaw','unsafe')
    if not all(k in m for k in needed):return None
    return dict(
        fixed_touch=float(m['fixed_contact_force_N'][0])>.02,
        moving_touch=float(m['moving_contact_force_N'][0])>.02,
        fixed_pad=bool(m['fixed_contact_in_pad'][0]) and float(m['fixed_contact_force_N'][0])>.02,
        moving_pad=bool(m['moving_contact_in_pad'][0]) and float(m['moving_contact_force_N'][0])>.02,
        opposing=bool(m['opposing_contact_forces'][0]),
        enclosed=bool(m['enclosed'][0]),
        jaw_ready=float(m['jaw'][0])<1.10,
        safe=not bool(m['unsafe'][0]),
        fixed_force_N=float(m['fixed_contact_force_N'][0]),
        moving_force_N=float(m['moving_contact_force_N'][0]),
        jaw_rad=float(m['jaw'][0]),
        held_seconds=float(hold_seconds),required_seconds=float(required_seconds),
    )


def reset_training_charts(env,env_ids=None):
    """Publish environment zero's reset before Isaac Lab renders its home pose."""
    panel=getattr(env,'_training_reward_panel',None)
    if panel is None:return
    if isinstance(env_ids,slice):
        if 0 not in range(len(env.episode_length_buf))[env_ids]:return
    elif env_ids is not None and not any(int(i)==0 for i in env_ids):return
    counter=getattr(env,'common_step_counter',0)
    if getattr(panel,'last_reset_step',None)==counter:return
    panel.last_reset_step=counter
    panel.last_episode_step=0
    panel.total=0.
    for attribute in ('approach_running_total','descend_running_total',
                      'readiness_running_total','enclosure_running_total',
                      'fixed_finger_running_total','moving_finger_running_total'):
        setattr(panel,attribute,0.)
    if panel.snapshot:
        sample=dict(panel.snapshot)
        sample.update(stage=0,reward_stage=0,episode_step=0,simulation_step=counter,
                      failed=False,success=False,grasp_confirmed=False,cube_rise_cm=0.,
                      group_contribution=[[0. for _ in row] for row in sample['group_contribution']],
                      contribution=[0. for _ in sample['contribution']],step_reward=0.)
        for key in ('approach_progress_total','descend_progress_total',
                    'readiness_progress_total','enclosure_progress_total',
                    'fixed_finger_progress_total','moving_finger_progress_total'):
            sample[key]=0.
        sample.pop('grasp_gate_checks',None)
        sample.pop('grasp_checks',None)
        sample.pop('approach_checks',None)
        panel.snapshot=sample
        panel.record(sample)
        if hasattr(panel,'reward_window'):panel.reward_window.send(sample)


def update_training_charts(env,state):
    import os
    standalone=os.environ.get('SO101_REWARD_WINDOW','0')=='1'
    if not hasattr(env,'_training_reward_panel'):
        viewer=next((getattr(v,'_viewer',None) for v in env.sim.visualizers
                     if getattr(v,'_viewer',None) is not None),None)
        if viewer is None and not standalone:return
        panel=RewardPanel([])
        panel.chart_note='LIVE PPO TRAINING | Environment 0 (the visible robot), not batch average'
        panel.last_episode_step=-1
        if standalone:
            from isaac_vision_rl.utils.reward_window import RewardWindow
            panel.reward_window=RewardWindow()
        elif viewer is not None:
            viewer.register_ui_callback(panel.draw_charts,position='free')
        env._training_reward_panel=panel
    panel=env._training_reward_panel
    age=int(env.episode_length_buf[0])
    if age<panel.last_episode_step:
        panel.total=0.
        panel.approach_running_total=0.
        panel.descend_running_total=0.
        panel.readiness_running_total=0.
        panel.enclosure_running_total=0.
        panel.fixed_finger_running_total=0.
        panel.moving_finger_running_total=0.
    panel.last_episode_step=age
    contribution=state.contribution[0].detach().cpu().tolist()
    groups=state.group_contribution[0].detach().cpu().tolist()
    approach_step=groups[0][0]
    descend_step=groups[0][2]
    panel.approach_running_total+=approach_step
    is_grasp_finetune=hasattr(state,"grasp_gate_checks")
    if is_grasp_finetune:
        panel.descend_running_total+=descend_step
    else:
        panel.readiness_running_total+=descend_step
    panel.enclosure_running_total+=groups[1][2]
    panel.fixed_finger_running_total+=groups[1][0]
    panel.moving_finger_running_total+=groups[1][1]
    panel.total+=sum(contribution)
    m=getattr(state,'measurements',{})
    grasp_confirmed=bool(m['grasp_valid'][0]) if 'grasp_valid' in m else False
    panel.snapshot=dict(stage=int(state.stage[0]),reward_stage=int(state.reward_stage[0]),
        episode_step=age,simulation_step=int(env.common_step_counter),
        failed=bool(state.failure[0]),success=bool(state.success[0]),
        grasp_confirmed=grasp_confirmed,
        cube_rise_cm=float(m["lift_rise"][0])*100 if "lift_rise" in m else 0.,
        group_contribution=groups,raw=state.raw[0].detach().cpu().tolist(),contribution=contribution,
        step_reward=sum(contribution),
        approach_step_reward=approach_step,
        approach_progress_total=panel.approach_running_total,
        descend_step_reward=descend_step,
        descend_progress_total=panel.descend_running_total,
        readiness_progress_total=panel.readiness_running_total,
        enclosure_progress_total=panel.enclosure_running_total,
        fixed_finger_progress_total=panel.fixed_finger_running_total,
        moving_finger_progress_total=panel.moving_finger_running_total)
    if int(state.stage[0])==1:
        from isaac_vision_rl.utils.four_stage import GRASP_HOLD_SECONDS
        checks=grasp_check_snapshot(m,state.acquisition_dwell[0],GRASP_HOLD_SECONDS)
        if checks is not None:panel.snapshot['grasp_checks']=checks
    elif int(state.stage[0])==0 and all(k in m for k in ('grasp_target_xy_distance','downward_cosine','clearance')):
        from isaac_vision_rl.utils.four_stage import DESCENT_DOWNWARD_COSINE_MIN,DESCENT_XY_TOLERANCE_M
        panel.snapshot['approach_checks']=dict(
            xy_error_cm=float(m['grasp_target_xy_distance'][0])*100,
            xy_tolerance_cm=DESCENT_XY_TOLERANCE_M*100,
            downward_cosine=float(m['downward_cosine'][0]),
            downward_cosine_min=DESCENT_DOWNWARD_COSINE_MIN,
            clearance_mm=float(m['clearance'][0])*1000,
            descent_allowed=bool(state.using_descent_target[0]),
        )
    if int(state.reward_stage[0]) == 0 and is_grasp_finetune:
        panel.snapshot["grasp_gate_checks"] = {
            name: bool(value[0].detach().cpu())
            for name, value in state.grasp_gate_checks.items()
        }
        panel.snapshot["grasp_gate_dwell_fraction"] = min(
            1.0, float(state.dwell[0].detach().cpu()) / 0.08
        )
    if hasattr(state,"weights"):
        panel.snapshot["reward_weights"]=state.weights
        panel.snapshot["reward_style"]=state.reward_style
    if getattr(state,"reward_mode",None)=="bounded" and getattr(state,"reward_style","progress")=="progress":
        panel.snapshot["grasp_bonus_used"]=state.grasp_best[0].detach().cpu().tolist()
    panel.record(panel.snapshot)

    if hasattr(panel,"reward_window"):panel.reward_window.send(panel.snapshot)
