# Comparing smooth lift control with the baseline

## Current experiment: wrist-only control

The completed seed-101 comparison gave baseline 46/50 successes and full smooth
control 34/50. Full smooth control reduced rotation but changed approach and
closure timing too much for the existing policy. The previous baseline run was
42/50, so repeatability across seeds remains to be checked.

Use `--wrist-only-control` for the next isolated experiment. It preserves the
baseline arm limit (1.5 rad/s), EMA (0.2), immediate closure, 0.6-second capture
pause and reward settings. It holds the measured wrist-roll angle from camera
capture until release/reset. This is a camera-capture latch, not proof of a secure
physical grasp. Other joints still move, so full tool orientation is not locked.
Do not combine this option with `--smooth-control`.

Headless evaluation, after permission to begin the evaluation stage:

```bash
python scripts/eval_pipeline.py --task lift --curriculum lift --policy vision \
  --checkpoint logs/rsl_rl/so101_picklift_vision/2026-10-10_09-44-58/model_3898.pt \
  --episodes 50 --seed 101 --cube-jitter 0.01 --headless --record-video \
  --wrist-only-control --output outputs/evaluations/lift_3898_wrist_only_seed101.json
```

Visible evaluation: replace `--headless` with `--viz kit`. Use the same seed and
jitter; run one evaluation at a time. Compare against the recorded baseline's
strict success, supported tool rotation speed, sliding, force peaks and timeouts.
Do not fine-tune until the isolated comparison has been reviewed.

Four motion/hold tests and three capture tests run without simulation. They test
reference bounds, capture/release, partial resets, joint limits and diagnostic
conditions, but do not establish physical smoothness or grasp performance.

Baseline: `model_3898.pt` from `2026-10-10_09-44-58`, evaluated at
42/50 strict successes with seed 101 and 0.01 m cube jitter. Preserve that
checkpoint and `outputs/evaluations/lift_3898_seed101.*`.

`--smooth-control` is optional and currently applies only to the lift task.
Without it the existing controller parameters remain in use.

Experimental settings:

- Arm position-reference speed cap: 0.75 rad/s, instead of 1.5.
- Position-reference acceleration cap: 2 rad/s² during ordinary policy motion.
- Wrist-roll reference speed cap: 0.25 rad/s.
- Wrist-roll target retained after camera capture until reset/release.
- Gripper target ramps at 1.2 rad/s instead of jumping directly to zero.
- Capture pause extended from 0.6 to 1.2 seconds to allow ramped closure.

The reference limiter brakes near targets. Hard joint boundaries and the existing
capture pause take priority over acceleration limiting. These are command limits,
not guarantees on actual joint speed, acceleration, force, or tool orientation.
Holding wrist roll does not hold full tool orientation because other joints move.
Gripper effort stays at 0.20 N·m until measured contact/slip evidence supports a change.
Capture tolerances, rewards, policy architecture and observation dimensions are unchanged.

New `Grasp/capture_gate_*` training logs identify visibility, confidence, downward
orientation, XY alignment, Z alignment and finite input. Recorded evaluation traces
include those conditions plus actual wrist speed, tool quaternion/angular velocity,
and each finger's cube-specific normal force. Normal force is not total shear force
or a real-world damage threshold. Episode reports include peak wrist speed,
peak tool angular speed while both fingers contact the cube, and peak normal force.

## Before more training

1. Verify `nvidia-smi` works. Do not launch a simulator while the driver is unavailable.
2. Re-record the baseline with the new diagnostics, without `--smooth-control`,
   to a new filename (`lift_3898_baseline_diagnostics.json`). The old video has no
   wrist or force measurements.
3. Evaluate the same checkpoint and seed with `--smooth-control`. Do not overwrite
   the original evaluation.
4. Compare strict success, sliding, supported tool rotation and contact force.
   Slower motion can hurt performance before fine-tuning; six-second timeouts remain.
5. Inspect failures using gate traces before changing thresholds. Occluded capture
   should not automatically close or reopen jaws without reliable evidence.
6. Only then fine-tune a separate run and repeat randomized evaluation with seeds
   101, 202 and 303. Training completion is not a promotion criterion.

Example comparison (headless):

```bash
python scripts/eval_pipeline.py --task lift --curriculum lift --policy vision \
  --checkpoint logs/rsl_rl/so101_picklift_vision/2026-10-10_09-44-58/model_3898.pt \
  --episodes 50 --seed 101 --cube-jitter 0.01 --headless --record-video \
  --output outputs/evaluations/lift_3898_baseline_diagnostics.json

python scripts/eval_pipeline.py --task lift --curriculum lift --policy vision \
  --checkpoint logs/rsl_rl/so101_picklift_vision/2026-10-10_09-44-58/model_3898.pt \
  --episodes 50 --seed 101 --cube-jitter 0.01 --headless --record-video \
  --smooth-control --output outputs/evaluations/lift_3898_smooth_seed101.json
```

For visible evaluation replace `--headless` with `--viz kit` and use a separate
output filename. For visible playback use `scripts/play.py --task lift
--curriculum lift --policy vision --smooth-control --checkpoint <checkpoint>
--num_envs 1 --cube-jitter 0.01 --viz kit`.

Grasp recovery is not implemented by this experiment: an unseen cube alone is
not proof of a failed grasp, and an automatic reopen during lift could drop it.
Use the new measurements to design and verify recovery next.
