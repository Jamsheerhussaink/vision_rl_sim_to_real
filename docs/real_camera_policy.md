# Observe a trained policy reacting to real RGB and encoder feedback

This is different from `mirror_evaluation.py`: no simulator launches and no
simulated camera or joint state is used. `scripts/hardware/real_camera_policy.py`
loads the trained actor directly and feeds it RealSense RGB and physical encoders.

Default shadow mode writes no motor target or torque registers. It displays and
records policy predictions, so objects can be repositioned by hand while watching
the action overlay. It is not a physical grasp test. Previous-action and filter
memory are hypothetical controller history because the motors do not execute it.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/hardware/real_camera_policy.py \
  --checkpoint logs/rsl_rl/so101_picklift_rgb/2026-10-15_18-49-25/model_1999.pt \
  --port /dev/ttyACM0 --seconds 6
```

Close other camera and serial programs first. Add `--headless` to record without
opening the preview window. Press Q or Ctrl+C to stop. The default duration is
one six-second checkpoint episode; restart for another episode.

Each run creates a unique folder under `outputs/real_policy/` with:

- `real.mp4`: actual colour camera video, host wall-clock playback at 30 fps.
- `policy_reaction.mp4`: same video with predicted actions and joint targets.
- `actions.jsonl`: inputs, outputs, camera timestamps and elapsed host times.
- `report.json`: mode, checkpoint, camera intrinsics and errors.

The script requests 424x240 RGB at the loaded checkpoint control rate (60 Hz for
the older checkpoint above). Current training defaults differ; camera/profile
matching remains unresolved. Images use letterbox128 preprocessing. State order is six relative positions, six velocities, six previous
clipped actions, six filtered targets and one normalized time-remaining value.
Relative positions use the saved training default joint angles, not guessed zero
angles. Velocities are finite differences of real encoder samples. The saved
EMA coefficient and velocity limit are used; critic simulator state is unnecessary.

Shadow mode labels poses outside calibration rather than treating them as safe.
The displayed target radians remain unmodified policy/controller proposals;
logged tick conversion is clipped for reference only. No ticks are sent.

An explicit `--execute` path exists. A 0.5-second physical control-path test has
passed; manipulation and longer operation remain unverified.
It requires the arm within 25 ticks per joint of checkpoint home, or 50 ticks
with `--align-home`, checked startup and a bounded episode. Out-of-range policy
targets are currently clipped to calibrated limits and counted in the report;
controller consistency remains unresolved. The motor worker enforces feedback,
speed and stale-data checks. It leaves torque
at the last targets on exit and does not perform an automatic home return.
Do not bypass failed checks or treat this option as a certified deployment.

## Current evidence

A two-second read-only test on 2026-10-16 completed 120 predictions at approximately
60 Hz using real RGB and encoders. All real joint positions remained unchanged
in its log. There were 43 out-of-calibration target proposals; motor execution
was not enabled. Recording:
`outputs/real_policy/1792127629087734247/policy_reaction.mp4`.

A separate five-step simulator-only parity test compared the standalone adapter
against the actual simulation observation, loaded actor and action processing.
Maximum difference was approximately 2.4e-7. Evidence:
`outputs/real_policy/adapter_parity.json`.
This verifies the software adapter for that checkpoint/controller configuration,
not sim-to-real task performance or correctness of physical calibration.

## First bounded physical policy trial

On 2026-10-16, with explicit user instruction to start real-policy control, the
first 0.5-second trial completed 30 steps at about 59.8 Hz. It used actual RealSense
RGB and physical encoders, not simulation mirroring. All 30 proposed poses passed
calibration checks. The shoulder was initially 27 ticks from the checkpoint home;
`--align-home` allowed a bounded nearby alignment first. This option refuses any
starting pose more than 50 ticks per joint from home, then uses the existing
speed and settling checks; the current gate without it is 25 ticks.

Recording and report: `outputs/real_policy/1792127848968835792/`.
A read-only postcheck confirmed all six motors holding with torque enabled and
fault registers zero. The arm stayed at the final commanded pose; it did not
return home. A repeat must pass the starting-pose gate again. This trial does not
prove grasping, lift, placement, or a generally safe workspace trajectory.

## Startup and shutdown repair (2026-10-16)

Startup refusal and shadow cleanup close serial without issuing a home trajectory
or changing torque. Real-camera execution now preserves powered hold on completion,
interruption and worker faults. It does not automatically park or release torque.
Powered hold is a requested motor state, not proof of support after power loss,
a servo fault or a communication failure. Reports distinguish confirmed serial
closure from unconfirmed shutdown; cleanup never races a worker that may still run.

Shared worker fault cleanup also overrides requested torque release. The legacy
`park_on_exit` option skips its unverified route and preserves torque. Explicit
release without parking on a clean worker exit remains supported for existing
callers; this alone does not verify a supported resting pose.

No physical recovery route was validated by this repair. Wrist/gripper calibration,
controller limits, camera matching and feedback-confirmed home recovery remain
required before another deployment trial.

Mock-only verification (no hardware access):

```bash
PYTHONPATH=src:tests conda run -n env_isaaclab python -m pytest -q -p no:cacheprovider \
  tests/test_real_policy_shutdown.py tests/test_evaluation_mirror.py \
  tests/test_bridge_startup.py tests/test_hardware_bridge.py -k 'not udp_loopback'
```
