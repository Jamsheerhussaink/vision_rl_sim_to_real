"""Three manager slots include every independently graphed reward column."""
from isaaclab.managers import ManagerTermBase
from isaac_vision_rl.utils.four_stage import machine,tick

class StageReward(ManagerTermBase):
    def reset(self,env_ids=None):
        machine(self._env).reset(env_ids)
        if hasattr(self._env,'_training_reward_panel'):
            from isaac_vision_rl.utils.training_reward_charts import reset_training_charts
            reset_training_charts(self._env,env_ids)
        if hasattr(self._env,"_success_videos"):
            self._env._success_videos.reset(env_ids)

    def __call__(self,env,slot):
        # Manager multiplies by dt: undo it for explicitly integrated/event rewards.
        # Families now have up to six columns. Indexing only columns 0,1,2
        # silently dropped columns 3,4 (grasp credit and dragging penalties).
        # Stride them across the existing three adapter slots; graphs retain
        # all independent terms while their total exactly equals PPO's reward.
        return tick(env).contribution[:,slot::3].sum(-1)/env.step_dt
