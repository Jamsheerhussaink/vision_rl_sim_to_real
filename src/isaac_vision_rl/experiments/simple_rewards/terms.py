"""Reward-manager adapter; each slot sums a column of the active reward families."""
from isaaclab.managers import ManagerTermBase
from isaac_vision_rl.experiments.simple_rewards.stages import machine,tick

class StageReward(ManagerTermBase):
    def reset(self,env_ids=None):
        machine(self._env).reset(env_ids)
        if hasattr(self._env,"_success_videos"):
            self._env._success_videos.reset(env_ids)

    def __call__(self,env,slot):
        # Manager multiplies by dt: undo it for explicitly integrated/event rewards.
        return tick(env).contribution[:,slot]/env.step_dt
