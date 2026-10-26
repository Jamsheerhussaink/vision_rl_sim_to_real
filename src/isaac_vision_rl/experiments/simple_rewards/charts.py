"""Read-only Newton reward charts for environment zero of a PPO run."""
from isaac_vision_rl.experiments.simple_rewards.panel import RewardPanel, CUMULATIVE_TERMS


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
            from isaac_vision_rl.experiments.simple_rewards.window import RewardWindow
            panel.reward_window=RewardWindow()
        elif viewer is not None:
            viewer.register_ui_callback(panel.draw_charts,position='free')
        env._training_reward_panel=panel
    panel=env._training_reward_panel
    age=int(env.episode_length_buf[0])
    if age<panel.last_episode_step:
        panel.total=0.
        panel.approach_running_total=0.
        panel.open_jaws_running_total=0.
        panel.enclosure_running_total=0.
    panel.last_episode_step=age
    contribution=state.contribution[0].detach().cpu().tolist()
    groups=state.group_contribution[0].detach().cpu().tolist()
    approach_step=groups[0][0]
    panel.approach_running_total+=approach_step
    panel.open_jaws_running_total+=groups[0][2]
    panel.enclosure_running_total+=groups[1][0]
    panel.total+=sum(contribution)
    panel.snapshot=dict(stage=int(state.stage[0]),reward_stage=int(state.reward_stage[0]),
        group_contribution=groups,raw=state.raw[0].detach().cpu().tolist(),contribution=contribution,
        step_reward=sum(contribution),
        approach_step_reward=approach_step,
        approach_progress_total=panel.approach_running_total,
        open_jaws_total=panel.open_jaws_running_total,
        enclosure_progress_total=panel.enclosure_running_total)
    for i,terms in enumerate(__import__('isaac_vision_rl.experiments.simple_rewards.stages',fromlist=['TERMS']).TERMS):
        for j,name in enumerate(terms):
            if name in CUMULATIVE_TERMS:
                key,attribute=CUMULATIVE_TERMS[name]
                if age<getattr(panel,'simple_last_age',-1):setattr(panel,attribute,0.)
                setattr(panel,attribute,getattr(panel,attribute)+groups[i][j])
                panel.snapshot[key]=getattr(panel,attribute)
    panel.simple_last_age=age
    panel.snapshot['measurements']={key:float(state.measurements[key][0]) for key in
        ('fixed_finger_distance','moving_finger_distance') if key in state.measurements}
    if hasattr(state,"weights"):
        panel.snapshot["reward_weights"]=state.weights
        panel.snapshot["reward_style"]=state.reward_style
    if getattr(state,"reward_mode",None)=="bounded" and getattr(state,"reward_style","progress")=="progress":
        panel.snapshot["grasp_bonus_used"]=state.grasp_best[0].detach().cpu().tolist()
    panel.record(panel.snapshot)

    if hasattr(panel,"reward_window"):panel.reward_window.send(panel.snapshot)
