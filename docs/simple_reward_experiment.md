# Separate simple reward experiment: simple pure-PPO rewards

The original training launcher, original rewards/UI, physics, camera, actor inputs,
and existing checkpoints are unchanged. Only the separate experiment is modified.
This experiment now uses `simple_rewards_v6_descend_until_cube_between_fingers`. Its logs are separate from
the original training. The policy controls the jaws directly; there is no
stage-based jaw override in this version.

## Rewards

Exactly three terms per stage. No overlapping reward families.

| Stage | Term 1 | Term 2 | Term 3 |
|---|---|---|---|
| Reach | Lower Cube Between Fingers (+0.30) | Point Fingers Down (+0.10) | Keep Jaws Open During Descent (+0.25) |
| Grasp | Fixed Finger Near Cube Side (+0.35) | Moving Finger Near Opposite Side (+0.35) | Dragging Cube (-2) |
| Lift | Lift to 10–12 cm (+0.30) | Keep Cube Steady (+0.20) | Moving Cube Too Fast (-0.5; above 0.25 m/s) |
| Place | Move Cube Toward Box (+0.25) | Lower Cube Into Box (+0.25) | Success +4 / failed Place -0.1 once |

Reach first aligns the gripper sideways, then its distance reward continues
down toward the cube centre. It changes to Grasp as soon as the cube is centered
between the open, downward-facing fingers, the pinch centre is within 1.25 cm
vertically of the cube centre, and table clearance is safe. Grasp then closes
the jaws. The Reach transition no longer waits at an overhead-only waypoint or
adds a dwell delay. The table safety checks remain active.

The open-jaw reward remains tied to alignment and downward orientation, but it
no longer loses score simply because the gripper descends. This avoids making
the desired lowering motion look like a mistake to PPO.

The two finger rewards replace the old Stage 2 center-distance and closing terms.
Each measures one geometric distance: from the actual inner finger mesh patch to
a small central region on its assigned cube side. The region is 12 x 12 mm, sampled at nine fixed surface points. Mesh patches come from the configured USD collision
mesh, filtered by the existing contact regions and inward face normals. Contact
boxes are selection masks, never substitute surfaces or bounding-box centers.

At Stage 2 entry, the fixed inward surface normal chooses the closest horizontal
cube-face axis. Fixed and moving fingers receive opposite signs on that axis.
The assignments remain locked until reset; face targets follow cube position and
rotation. Choosing by the fixed surface normal avoids bias from the moving jaw's
open angle. Penetration samples from the real patch prevent sampled intrusion
from receiving perfect credit, even if another patch point crosses the target.
This check is sampled geometry, not an exact volume-collision certificate.

For each finger, reward = 0.15 * (previous_distance - distance) / 0.05.
Distances are metres; the graph also shows millimetres. First sample earns zero.
Closer is positive, farther negative, stationary zero additional income. Each
chart accumulates independently within the episode, including below zero. Both
finger rewards remain available until the 0.2-second grasp check passes. Positive
credit from further jaw closure after initial contact is suppressed; alignment
feedback remains available. Stationary contact gives no extra income.
Real opposing contact and the existing dwell still decide advancement to Lift.
Cube distance alone is not proof of contact, grasp, held lift, or placement.

To advance from Lift to Place, the cube must still be grasped, move below
0.25 m/s, and reach at least 12 cm above its reset height. It advances as soon
as those conditions are true; there is no extra 0.5-second hold. The lift-height
reward target remains unchanged.

The live UI shows only the current stage's three terms. Stage 2 has the two
finger graphs and dragging, with live/full-history controls retained. Audit-only
contact-to-mesh checks run in the dedicated manual/evaluation launcher, not on
every normal training step. Actor inputs remain RGB and existing policy readings;
object coordinates are reward-only simulator information.

## Verification and limits

100 tests passed, including 24 focused experiment tests: independent approach and
retreat, stationary/no free initialization credit, wrong same-side placement,
rotation, above/behind/penetrating cases, fixed assignments until reset, contact
stopping squeeze credit, reset, three terms, graphs, and unchanged original files.

A bounded old-policy evaluation reached all four stages but did not verify final
placement within its 300-step budget. It is an unchanged actor, not a model trained
with the new rewards. Earlier repeat behavior also included camera-exit failures.
An IK approach test showed fixed distance 31.2 -> 6.1 mm and moving distance
80.8 -> 20.2 mm, with correctly signed independent reward changes. This exposes
remaining off-center contact. In the recorded policy check, median contact-to-mesh
errors were about 0.75 mm fixed and 0.38 mm moving; maximum about 3.44 and 2.51 mm.
The source mesh and PhysX cooked collision/contact margins are not identical.
These are bounded checks in one layout, not broad grasp certification.

The older IK routine's 11 cm lift disagreed with the task's 15–20 cm requirement.
The separate teacher now requests a relative lift, but the strict wrist-down IK
solver can reject that pose. It reports the blockage cleanly. This is not proof
that the learned policy cannot lift; the actor can tilt its wrist while holding.
No stage or safety checks were relaxed to fake success.

Evidence in `outputs/simple_rewards/`:
- `finger_faces_evaluation_report.json`
- `finger_faces_final_case02.json/.mp4` (unchanged learned actor)
- `finger_faces_approach_check.json` (IK approach diagnostic)
- `finger_faces_stage2_graphs.png` (read-only replay of recorded samples)

## Manual test

Simulation only; no physical motor commands. One environment and no manual episode
terminations. Do not run another simulator alongside an active training process.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/manual_simple_rewards.py --mode manual --device cuda:0 \
  --output outputs/simple_rewards/manual_finger_faces.json
```

## Training commands

Run only one command at a time. These import the older actor but reset critic,
optimizer and learning counter because the rewards changed. They do not modify
its checkpoint. Smoke-test checkpoints are not validated policies. For a genuine
resume from a saved v2 finger-face run, use that checkpoint and omit actor-warmstart.

Start fresh for v5 because the jaw controller and rewards changed. Do not pass a
checkpoint or teacher/IK action data. Existing checkpoints remain untouched.

Headless with graphs:

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/train_simple_rewards.py \
  --task place --policy rgb --curriculum place \
  --reward-profile four-stage --reward-style continuous \
  --grasp-reward-mode overlap --training-goal place \
  --lift-rise-min 0.10 --lift-rise-max 0.12 --lift-hold-seconds 0.1 \
  --lr-schedule adaptive --cube-jitter 0 \
  --envs 16 --iterations 4000 --device cuda:0 \
  --record-successes --headless --reward-window
```

Newton, one visible environment and graphs (16 environments still train):

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/train_simple_rewards.py \
  --task place --policy rgb --curriculum place \
  --reward-profile four-stage --reward-style continuous \
  --grasp-reward-mode overlap --training-goal place \
  --lift-rise-min 0.10 --lift-rise-max 0.12 --lift-hold-seconds 0.1 \
  --lr-schedule adaptive --cube-jitter 0 \
  --envs 16 --iterations 4000 --device cuda:0 \
  --record-successes --viz newton --max_visible_envs 1 --live-reward-graphs
```

Focused verification:

```bash
PYTHONPATH=src:.:tests /home/ai/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_simple_reward_experiment.py -q
```

A final two-environment/two-update CUDA training check passed and wrote only new
v2 checkpoints under `2026-10-21_16-21-01`. No long training was launched.

## Verified tuning: v3

On the SAME recorded controlled approach (121 Grasp samples, 2.421 mm of cube
translation), the old instantaneous-velocity dragging penalty was -2.625. Using
actual cube XY displacement between control steps gives -0.09687 with the SAME
-2 weight. Total Stage 2 reward changes from -2.369 to +0.1387. This corrects a
measurement problem; it is not evidence of better learned manipulation.

Finger weights remain +0.15 each. The central side region has no air-gap tolerance:
its nine targets lie exactly on the cube surface. Penetration protection and real
opposing-contact requirements remain. Region/center distances are both recorded
for audit. A 16 mm region was checked offline but not selected: it did not meet
the strict 2 mm goal for both fingers, so there was no reason to broaden the area
further. With the 12 mm region, an offline search found a safe near-grasp pose
about 0.6 mm / 3.3 mm from the targets, with 10.7 mm table clearance and fingers
down. This is not a perfect target-calibration or physical grasp certificate.
Distance does NOT have to reach zero to advance: real grasp/contact checks govern
stage advancement, so residual mesh/contact error does not trap the task here.

The unchanged old actor still reached Place but failed with camera exit and
excess height. No improved policy has been claimed. No long training was launched.

Evidence: `outputs/simple_rewards/tuning_20261021/reward_tuning_report.json`,
`tuned_approach.json/.mp4`, `tuned_old_actor.json/.mp4`, `tuned_stage2_graphs.png`,
and `target_reachability.json`. The prior experiment source is preserved in
`v2_source_before_tuning.zip`. Original training sources/checkpoints are untouched.

### Previous guided-jaws trial (v4, historical)

`simple_rewards_v4_guided_jaws` kept every reward unchanged. Its separate
`GuidedGripperAction` opens the jaws during Reach and blocks premature closing
in Grasp until the cube center is within 4 cm of the pad midpoint, lies between
the pad planes, and downward alignment exceeds 0.75. Existing speed and
acceleration limits still apply; home still starts closed. Lift and Place retain
policy jaw control so holding and release are possible. This is simulator-state
controller assistance, not an RGB-only autonomous closing decision. Object
coordinates are not added to actor inputs. A real deployment would need an
RGB-derived replacement for this guidance.

That version is retained for history only. Its jaw override is not used by v5.
The v5 model starts with open jaws and learns jaw movement from its RGB policy.
