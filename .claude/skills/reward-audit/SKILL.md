---
name: reward-audit
description: Run reward-hacking audit, task feasibility check and related tests after editing rewards/ or envs/
disable-model-invocation: true
---
Run in order and stop to report on the first failure:
1. `python -m pytest -q tests/test_reward_hacking.py tests/test_grasp_controls.py`
2. `python scripts/check_reward.py` (add `--only <config>` or `--terms <strategy>` to drill in)
3. `python scripts/check_pick.py --no-physics`, then `--physics-only` if reward/control changed.

Summarise: which scripted strategy scored highest, whether any exploit (reward without a real grasp/lift) beats the honest strategy, and which reward term drives it. Propose a fix but don't apply it without asking.
