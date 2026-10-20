"""Read-only Newton reward charts for environment zero of a PPO run."""
from isaac_vision_rl.utils.reward_panel import RewardPanel


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
    if age<panel.last_episode_step:panel.total=0.
    panel.last_episode_step=age
    contribution=state.contribution[0].detach().cpu().tolist()
    panel.total+=sum(contribution)
    panel.snapshot=dict(stage=int(state.stage[0]),reward_stage=int(state.reward_stage[0]),
        group_contribution=state.group_contribution[0].detach().cpu().tolist(),raw=state.raw[0].detach().cpu().tolist(),contribution=contribution,
        step_reward=sum(contribution))
    if hasattr(state,"weights"):
        panel.snapshot["reward_weights"]=state.weights
        panel.snapshot["reward_style"]=state.reward_style
    if getattr(state,"reward_mode",None)=="bounded" and getattr(state,"reward_style","progress")=="progress":
        panel.snapshot["grasp_bonus_used"]=state.grasp_best[0].detach().cpu().tolist()
    panel.record(panel.snapshot)

    if hasattr(panel,"reward_window"):panel.reward_window.send(panel.snapshot)
