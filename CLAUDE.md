# vision_rl
Vision-based PPO for SO-101 pick-and-place on MuJoCo-MJX (JAX/Flax/Optax).

- Layout: `vision_rl/{envs,ppo,rewards,models,rendering}`, `scripts/`, `tests/`, `runs/` (git-ignored).
- Tests: `python -m pytest -q`. After reward/env edits run `/reward-audit`.
- Design decisions (obs, camera, frames, resize): see the `project-conventions` skill.
- Only hand-edit `envs/assets/robot.xml`; never `runs/`, `*.egg-info/`, or `envs/assets/meshes/`.
- Hooks in `.claude/hooks/` auto-format .py with ruff.
