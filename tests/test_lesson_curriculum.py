"""Run isolated subprocesses because lesson activation deliberately patches imports."""
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('lesson,terminal', [('approach',0),('grasp',1),('lift',2),('place',3)])
def test_lesson_stops_at_its_boundary_without_reward_bonus(lesson, terminal):
    code = f'''
import torch
from test_simple_reward_experiment import sample
from isaac_vision_rl.experiments.simple_rewards.lesson_curriculum import activate
activate({lesson!r})
from isaac_vision_rl.experiments.simple_rewards import stages
s=stages.StageMachine(1,'cpu',reward_style='continuous',lift_hold_seconds=0)
s.stage[:]={terminal}
s.cube_start_height[:]=.425
s.cube_start_height_valid[:]=True
for _ in range(20):
    s.update(sample(reach_ready=True,bilateral=True,enclosed=True,jaw=.8,
                    height=.425+stages.LIFT_RISE_MAX_M,speed=.0,placed=True),1/30)
assert s.success.item()
assert s.stage.item()=={terminal}
s.reset()
assert not s.success.item() and s.stage.item()==0
s.stage[:]={terminal}
for _ in range(20): s.update(sample(unsafe=True),1/30)
assert not s.success.item()
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=os.environ.copy())
