# Grasp fine-tuning experiment

This experiment is isolated behind `scripts/train_grasp_finetune.py`. It keeps
the original training script and checkpoints intact. Each stage has only three
reward terms. Stage 1 now shows three easy-to-read lines: move to the overhead
target 4.5 cm above the cube, stay clear of the table, and descend toward the
pinch point. The first and third lines are signed running totals: improving the
target pose raises the line, moving away lowers it, and staying still keeps it
flat. The descent line gives gradual feedback for opening and aligning over
the cube, then descending. Opening can help only in Stage 1, when the hand is
centered and the wrist points down; unsafe table clearance gives no credit.
Stage 2 begins only when the cube fits between the fingers at the pinch point
and the jaws are open enough to fit it. The live chart lists the actual
Reach-to-Grasp checks separately.

Once the overhead check unlocks descent, the fine-tuning run keeps the robot
on the descent route. The descent score rises as the hand moves down toward
the pinch height, then levels off; upward movement gets a negative signal. The
shared task logic used to send a low, off-center hand back to the raise target,
which could cause a visible upward motion. This experiment now suppresses that
route reset after descent has unlocked. Table-clearance safety checks still
apply.

Stage 2 still has three terms: move closer to the calibrated pinch target
(`+0.10`), close the jaws near a downward pinch pose (`+0.50`, raised from
`+0.10`), and avoid dragging the cube (`-2.00`). The jaw reward pays only when
the gripper reaches a new, more-closed position near the cube. Opening or
holding still gives zero jaw reward; reopening cannot restore already-earned
credit. This prevents repeated open-close cycles from farming reward. A real
opposing-finger grasp still controls the Grasp-to-Lift transition.

The Reach-to-Grasp transition requires the arm to descend near the pinch point,
the cube to fit between the calibrated finger-pad centers, the wrist to point
down, the jaws to remain open, and safe table clearance for 0.08 seconds. It no
longer also requires a separate earlier overhead latch; that extra gate was
redundant and prevented the jaw-closing reward from becoming active. The
cube-fit geometry already checks the rotated cube width against the measured
finger gap. A second `gap > width + 5 mm` test was redundant and could reject
valid fits; it has been removed. This does not relax contact, lift, placement,
or safety checks.

## Evidence so far

- The old `model_1550.pt` actor completed all four stages in a one-episode
  check, with 131 Reach steps, 15 Grasp steps, 92 Lift steps, and 1,111 Place
  steps. It lifted about 19.6 cm and slid the cube about 1.9 cm, then timed out
  in Place. It is a useful fine-tuning start, not a successful place model.
- Across three fixed-scene episodes with the patched reward/stage logic, all
  trials reached the requested 15–20 cm lift band (15.9–20.0 cm), with average
  cube slide about 2.7 cm. Placement success was still 0/3; all timed out.
- The same actor under the previous stage logic also scored 0/3, with average
  lift 14.9 cm and slide 7.0 cm. These small samples suggest improvement in
  lift/slide behavior, but do not prove it was caused only by the reward change.
- A behavior-cloning candidate fit the demonstrations closely (validation loss
  0.00004), but failed the table-clearance safety check in 3/3 trials. Do not
  use that candidate as the starting policy.
- Focused tests passed: 81. They test reward signs, stationary behavior,
  stage gating, current-stage graphs, and evaluator stage reporting. They do
  not prove a learned policy will place the cube.
- The later `model_350.pt` check did not leave Reach: the gripper came within
  about 5.2 cm of the cube, but it did not descend into the pinch gate, lifted
  0 cm, and timed out. Use the earlier `model_1550.pt` actor as a starting
  point for this experiment; it at least reached Grasp and Lift in the recorded
  trial. Those results are small evaluation samples, not proof of reliable
  grasping or placement.
- The Oct 22 `grasp_close_v2` run was stopped after checkpoint 200. At update
  201, descent had never unlocked and every environment remained in Reach. The
  old code paid the descent graph only after a strict overhead latch, including
  a nearly fully open jaw, while the stage transition also required the pinch
  geometry. The updated experiment gives signed progress toward that good pose
  before the latch and keeps the real descended-and-between-fingers transition.
  This is a code-level fix; it still needs a new training run and evaluation.
- A one-episode deterministic check of `model_1550.pt` with the revised gate
  still did not pass Reach: it timed out after 1,350 steps, the cube slid about
  34.3 cm, and it lifted only 1.35 cm. The trace showed close-jaw commands for
  1,342 of 1,350 steps. This confirms the actor needs to learn to open at the
  centered, wrist-down approach before it can close around the cube. One episode
  is a diagnosis, not a reliable success-rate estimate.

Recorded evidence:

- `outputs/evaluation/grasp_reward_audit_model1550_20261022.json/.mp4`
- `outputs/evaluation/grasp_finetune_gate_model1550_20261022.json/.mp4`
- `outputs/evaluation/grasp_finetune_stagecheck_model1550_20261022.json`
- `outputs/evaluation/grasp_reward_audit_bc_candidate_20261022.json/.mp4`
- `outputs/evaluation/grasp_finetune_gate_model1550_after_fix_20261022.json/.mp4`

## Fine-tuning command

This starts a new PPO run from the old RGB actor. The old checkpoint remains
untouched; the critic, optimizer, and update counter start fresh because the
reward changed. The headless version below uses 8 environments to reduce RAM
use and opens the live chart in its own desktop window. The chart shows one
environment, even though training uses 8. You start it yourself; do not add
`--live-reward-graphs` to the headless command.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/train_grasp_finetune.py \
  --task place --policy rgb --curriculum place \
  --reward-profile four-stage --reward-style continuous \
  --grasp-reward-mode overlap --training-goal place \
  --lift-rise-min 0.15 --lift-rise-max 0.20 \
  --lr-schedule adaptive --cube-jitter 0 \
  --envs 8 --iterations 2000 --device cuda:0 \
  --checkpoint logs/rsl_rl/so101_pickplace_rgb_four_stage_v3_overlap/2026-10-17_18-50-28/model_1550.pt \
  --actor-warmstart \
  --headless --reward-window
```

To see the robot in Newton with the same live charts, use this instead of
`--headless --reward-window` (do not combine both display modes):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/train_grasp_finetune.py \
  --task place --policy rgb --curriculum place \
  --reward-profile four-stage --reward-style continuous \
  --grasp-reward-mode overlap --training-goal place \
  --lift-rise-min 0.15 --lift-rise-max 0.20 \
  --lr-schedule adaptive --cube-jitter 0 \
  --envs 8 --iterations 2000 --device cuda:0 \
  --checkpoint logs/rsl_rl/so101_pickplace_rgb_four_stage_v3_overlap/2026-10-17_18-50-28/model_1550.pt \
  --actor-warmstart \
  --viz newton --max_visible_envs 1 --live-reward-graphs
```

To evaluate a saved checkpoint with this experiment's stage rules:

```bash
python scripts/eval_grasp_finetune.py \
  --task place --checkpoint PATH_TO_CHECKPOINT.pt \
  --episodes 10 --seed 123 --device cuda:0 --headless --record-video \
  --output outputs/evaluation/grasp_finetune_candidate.json
```

The best checkpoint should be selected by recorded contact, stable lift,
placement, table-slide, and camera-visibility results—not by training reward
alone. No real-robot readiness is claimed.
