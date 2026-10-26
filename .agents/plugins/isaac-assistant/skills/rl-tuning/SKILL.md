---
name: rl-tuning
description: Expert runbook and procedures for Isaac Vision RL policy training, reward weight tuning, curriculum lessons, and live diagnosis on the SO-101 robotic arm.
---

# SO-101 Isaac Vision RL Tuning Guide

Use this skill whenever the user wants to train RL policies, diagnose training bottlenecks, tune reward weights, or evaluate checkpoints for the SO-101 robot arm.

## 1. Task Architecture & Four Stages
1. **Approach (`REACH`)**: Gripper descends from home to pregrasp waypoint (centered above cube `< 2.0 cm` XY error, `wrist_down > 0.90`, `jaws > 1.10 rad`).
2. **Grasp (`GRASP`)**: Descends from pregrasp waypoint to cube height, fingers enclose cube, bilateral force contact registered on both finger pads.
3. **Lift (`LIFT`)**: Raises cube vertically off the table (`> 0.10 m`), penalizes dragging or table sliding.
4. **Place (`PLACE`)**: Transports cube to target box and opens fingers.

## 2. Standard Training Commands
- **Curriculum Lesson (Recommended for stage bottlenecks)**:
  ```bash
  python scripts/train_curriculum.py --lesson <approach|grasp|lift|place> \
    --task place --curriculum place \
    --reward-profile four-stage --reward-style continuous \
    --envs 16 --iterations 500 --device cuda:0 \
    --viz newton --max_visible_envs 1 --reward-window
  ```
- **Full Four-Stage Training**:
  ```bash
  python scripts/train.py \
    --task place --policy rgb --curriculum place \
    --reward-profile four-stage --reward-style continuous \
    --grasp-reward-mode bounded --training-goal lift \
    --lr-schedule adaptive --cube-jitter 0 --box-jitter 0 \
    --envs 16 --iterations 500 --device cuda:0 \
    --viz newton --max_visible_envs 1 --reward-window
  ```

## 3. Diagnosis & Reward Tuning Workflow
1. Run `diagnose_live_training` to inspect live progress and stage completion rates.
2. If the robot reaches but fails to close fingers:
   - Check if sideways error is `> 2.0 cm` or `wrist_down < 0.90` (Gate locked).
   - If gate is locked, train `--lesson approach`.
   - If gate is open but fingers won't close, call `set_reward_weights` to increase `grasp_cube` and `grasp_approach`.
3. If table collisions occur: increase `table_collision` penalty.
4. If cube slips during lift: increase `pinch_force` bonus and `lift_cube`.

