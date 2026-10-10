# Camera-triggered grasp assistance and recorded evaluation

The available checkpoint `2026-10-08_11-25-59/model_0.pt` was recorded for three
exploratory episodes. It is an early saved checkpoint, not a completed training run.
The recording shows poorly timed closure/reopening, withdrawal without a grasp,
and cube sliding. Measured episode slide was approximately 8–15 cm and held-lift
success was 0/3. The annotated global-camera recording and gripper trace are:

- `outputs/evaluations/before_grasp_assist.mp4`
- `outputs/evaluations/before_grasp_assist.trace.json`
- `outputs/evaluations/before_grasp_assist.json`

The reference image `grasp_closed.png` is not a learned-policy result. Its actual
gripper angle near 1.04 rad does not mean the actuator ignored the close command:
the cube blocks further closure. The subsequent reference lift measured cube
centre above 0.47 m. A static camera image alone cannot prove a held grasp.

The lift task now uses a hybrid controller:

```text
PPO controls the five arm joints
  -> global RGB-D sees a centred cube in the tool frame
  -> confirm visibility, confidence and downward tool orientation twice
  -> command gripper closed
  -> hold arm joints at the measured capture pose for about 0.6 seconds
  -> keep jaws closed; PPO controls the arm's subsequent lift motion
  -> reset clears the latch
```

Capture needs camera-estimated lateral offset at most 12 mm, vertical offset at
most 15 mm, geometry confidence at least 0.75 and a downward-facing tool. These
are engineering thresholds, not proof that every grasp will succeed. Invalid or
occluded detections cannot start capture. After capture, occlusion does not reopen
the jaws. Actor and critic observe the latch and closure-time state, alongside
processed targets, so the controller memory is visible to learning.

While assistance is enabled, the policy's sixth gripper action is overridden:
open before capture, closed after capture. This is **assisted RL**, not evidence
that PPO independently learned when to close. The lift-only assist is disabled in
the placement environment so it cannot prevent release; placement remains gated
on learned lift mastery and needs separate evaluation.

The easy near-grasp stage also initializes the arm's action mean toward the
physically verified grasp pose, rather than the old hover pose. This initialization
is a fixed workcell prior. Live cube estimates still come from the camera.
Near-grasp exploration now starts at standard deviation 0.15 (previously 0.4),
with entropy targeting 0.15. Later stages retain their usual broader initialization.
Actor numerical inputs are now 34; critic inputs are 55. Start fresh: old
checkpoints are incompatible with `camera_grasp_assist_v7`.

The recorded assisted reference lift uses camera estimates for automatic closure
and camera-guided IK for arm movement. It measured both cube contacts during the
final hold and cube centre at about 0.494 m:

- `outputs/evaluations/camera_capture_assist.mp4`
- `outputs/evaluations/camera_capture_assist.json`

The new two-update assisted-policy smoke checkpoint also completed 3/3 held lifts
in deterministic evaluation on seed 202 with ±1 cm reset jitter. Its inspected recording is
`outputs/evaluations/assisted_policy_smoke.mp4` and measurements are
`outputs/evaluations/assisted_policy_smoke.json`. Cube slide was 2.0–2.3 cm,
exceeding the 1 cm mastery limit; this short sample is not a passed mastery gate.

The same two-update policy with 0.4 exploratory noise achieved 0/3 sustained lifts;
its recording is `outputs/evaluations/assisted_policy_exploration.mp4`. This is why
local-stage exploration was reduced. Deterministic and exploratory results are
reported separately; neither short sample proves mastery.

The reference recording verifies the close trigger and physical lift reference. Simulator cube height is used for scoring/video labels, not capture control.

Before new training, activate `env_isaaclab`, change to the project directory and
run preflight. Run only one simulator command at a time:

```bash
python scripts/train_pipeline.py --stage preflight --headless
```

To record and inspect the assisted reference, choose one:

```bash
python scripts/camera_pick_lift.py --headless --grasp-assist --record-video \
  --output outputs/evaluations/camera_capture_assist.json

python scripts/camera_pick_lift.py --viz newton_gl --grasp-assist --record-video \
  --output outputs/evaluations/camera_capture_assist.json
```

After preflight passes, choose one fresh training command:

```bash
python scripts/train_pipeline.py --stage grasp --policy state --cube-jitter 0.01 \
  --headless --envs 32 --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard

python scripts/train_pipeline.py --stage grasp --policy state --cube-jitter 0.01 \
  --viz newton_gl --envs 32 --max_visible_envs 32 \
  --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard
```

Monitor `Grasp/assist_latched_fraction`, bilateral cube contact, camera visibility,
and completed held lifts. Latch activation is a close command, not proof of a grip.
The success criteria still require both contacts and a stable elevated hold.

To record an actual new policy, supply its exact checkpoint:

```bash
python scripts/eval_pipeline.py --task lift --curriculum grasp --policy state \
  --cube-jitter 0.01 --checkpoint EXACT_NEW_CHECKPOINT --episodes 3 \
  --record-video --headless --output outputs/evaluations/learned_policy.json

python scripts/eval_pipeline.py --task lift --curriculum grasp --policy state \
  --cube-jitter 0.01 --checkpoint EXACT_NEW_CHECKPOINT --episodes 3 \
  --record-video --viz kit --output outputs/evaluations/learned_policy.json
```

Add `--sample-actions` to reproduce exploratory actions; omit it for deterministic
policy evaluation. Video annotations show the current applied gripper target and
actual jaw position; the trace also records the next requested raw action and
camera features. Retain the JSON, MP4 and trace together when diagnosing a failure.

The reduced-noise two-update exploratory smoke also achieved 0/3 sustained lifts,
with 13–27 cm accumulated table slide. Its video is
`outputs/evaluations/assisted_policy_local_exploration.mp4` and JSON is beside it.
This confirms automatic closure is not sufficient to make an untrained arm policy
stable. Do not interpret noise reduction or the deterministic 3/3 sample as mastery.
