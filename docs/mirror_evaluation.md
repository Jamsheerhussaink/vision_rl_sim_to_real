# Run the simulated policy and mirror it on SO-101

Cleanup note (2026-10-17): historical videos, per-frame traces, and diagnostic
images referenced below were removed. Retained JSON summaries remain evidence;
new runs can create fresh recordings. See [the cleanup guide](project_cleanup.md).

`scripts/hardware/mirror_evaluation.py` loads an explicit checkpoint and runs its
deterministic actor on the simulation RGB camera and simulated joint/controller
readings. Real scene changes do not feed back into the policy. A real camera is
unnecessary for this workflow.

The five arm references copy simulated measured joint positions. The gripper
reference uses the processed policy jaw target from the preceding applied action,
not the contact-constrained simulated jaw angle. For example, a simulated cube
can keep the actual jaw near 0.8 rad even while the policy requests 0 rad (close).
Sending 0.8 rad to hardware reproduced that partial opening rather than the
closure command. The initial frame uses the initial measured jaw angle; it does
not copy the action term's unrelated reset target. No action filters are run twice.

## Commands

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
CHECKPOINT=logs/rsl_rl/so101_picklift_rgb/2026-10-15_13-11-21/model_1900.pt
```

Test with mock motors first (no serial port opens without `--execute`):

```bash
python scripts/hardware/mirror_evaluation.py --checkpoint "$CHECKPOINT" \
  --task lift --curriculum lift --seed 101 --cube-jitter 0.01 \
  --viz newton --alignment-tolerance 12
```

For physical movement, add `--execute --port /dev/ttyACM0`. Use the actual port.
The following command matches the recorded model1900 seed101 evaluation settings
and mirrors one episode, then returns home and holds:

```bash
python scripts/hardware/mirror_evaluation.py --checkpoint "$CHECKPOINT" \
  --task lift --curriculum lift --seed 101 --cube-jitter 0.01 \
  --viz newton --alignment-tolerance 12 --execute --port /dev/ttyACM0
```

Replace `--viz newton` with `--viz kit` for the Kit viewer or `--headless` for no
viewer. Add `--max-steps 20` for a bounded trial; this still includes startup
alignment and the return home. Remove it for a whole episode.

Only this script should own serial: close other motor programs and do not launch
the UDP bridge alongside it. The Isaac Python environment needs `scservo_sdk`.
Verify physical calibration, joint signs, zeros and gripper mapping before
whole-arm copying. Opening the bus does not establish correct physical mapping.

## Home and shutdown behavior

Home means the **initial simulation pose captured before the episode**, using the
existing physical calibration. It is not the last position reached or a newly
inferred robot pose. The gripper returns to its initial angle as part of home.
The return is bounded joint interpolation, not collision-aware path planning.
Keep the return path clear. There is no automatic object placement/release step.

- Normal episode end or `--max-steps`: gradually return home, verify encoder error
  stays within the alignment tolerance for 0.2 seconds, then close serial while
  leaving torque enabled at the home targets.
- Ctrl+C or a runtime error: stop the worker and skip home movement. By default,
  close serial without disabling existing torque or replacing the last targets.
- `--release-on-exit`: explicitly disables torque when closing; support the arm
  before using this option. Other tools calling the motor bus's default `close()`
  retain their existing torque-disabling behavior.
- Partial startup failures can still disable torque through the existing startup
  rollback. Hardware faults, power loss and emergency power cutoff can also remove
  holding torque. Software cannot guarantee the arm stays supported in those cases.

After a normal hold, the motors remain powered. To run again on that stationary
arm, add `--allow-holding`; it adopts measured holding targets before approaching
home. Without that option, an already powered arm is refused. Support the arm
before manually releasing torque or disconnecting motor power.

## Timing and checks

The serial worker owns all bus access and is independent of inference. Arm speed
is bounded by `--speed 400` ticks/second per joint. Jaw speed has a separate
`--gripper-speed 600` ticks/second limit. Full calibrated gripper travel is 1516
ticks, so it needs approximately 2.53 seconds at that limit, plus motor lag.
Speed limits can make hardware lag the simulated motion.

Startup default tolerance is 5 ticks per joint. Recorded physical alignment errors
were 2, 2, 2, 10, 4 and 6 ticks, so `--alignment-tolerance 12` accommodates those
observed offsets. Twelve ticks is approximately 1.05 degrees for an arm joint;
the gripper has a separate calibration mapping. Both startup and home return
require command arrival plus 0.2 seconds inside tolerance, with a 30-second timeout.

The worker retains calibrated position bounds, motor fault checks, checked reads,
write failure checks and `--tracking-limit 200` ticks against the previous sent
command. A grasped object can prevent the jaw from reaching its closure target:
that does not justify bypassing feedback checks. Commanded closure does not prove
a physical grasp. Diagnose any tracking stop using recorded feedback.

The `--watchdog 0.25` timeout stops stale policy targets. During home return the
worker owns a fixed destination with its own timeout, so it does not require
simulation updates. The simulation is paced using its actual `step_dt`, nominally
60 Hz. Python scheduling is not hard real-time; a physical power cutoff remains
necessary for blocked serial calls or process/OS failures.

Episode reset poses are never sent to hardware. The script checks `dones` after
stepping, exits policy copying, and starts its explicit bounded home return.

## Reports and validation

Reports have unique paths under `outputs/mirror/`, or use `--output PATH`.
They record simulator angles, hardware references, processed jaw targets, latest
asynchronous commanded/measured ticks, last feedback, returned-home status,
shutdown torque state and errors. Motor feedback can refer to the previous sim
frame. `motor_shutdown_state: torque_left_enabled` records the software shutdown
choice, not a guarantee against subsequent hardware power loss.

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_evaluation_mirror.py tests/test_hardware_bridge.py tests/test_bridge_startup.py
```

The UDP loopback test needs permission to bind a local socket. Tests use fake
motors. The gripper/home mock integration report is
`outputs/mirror/gripper_home_mock.json`: model1900, lift, seed101, 0.01m jitter,
92 steps, full open/closed gripper command range, verified return home and
serial-only shutdown with torque preserved. Physical execution of these changes
has not been tested. Simulation grasp metrics remain in `eval_pipeline.py` reports.

The original 1600-tick/s jaw default caused an opening tracking stop in real
report `outputs/mirror/1792068680376408549.json`. It was reduced to 600 ticks/s;
the 200-tick tracking guard remains unchanged. This lower speed still requires
physical verification and may increase lag relative to the simulated gripper.

## Record real RGB and simulation together

Add `--record-video` to the mirror command. It starts a RealSense RGB recording
before opening the motor bus; failure to start the camera or an encoder aborts
before enabling motors. Close camera preview apps before recording. The real
camera is for reference only: the policy still consumes simulated RGB.

Alongside the report, a uniquely named `<report_stem>_videos/` folder contains:

- `sim.mp4`: simulated RGB, 424x240.
- `real.mp4`: RealSense colour, 640x480 at requested 30 fps, no depth.
- `comparison.mp4`: side-by-side simulation and real camera, with phase labels.
- `frames.jsonl`: frame times on the shared host clock.
- `recording.json`: actual camera profile, frame count and recording errors.

All three movies use a common 30-fps wall-clock timeline. Slow simulation frames
are repeated so the recording retains physical elapsed time. This is approximate
host-time alignment, not camera hardware synchronization. The recording includes
startup and return home. During return, the simulated frame is frozen and labelled
`return_home (sim frame held)`; do not interpret it as simulated home motion.
Raw physical RGB is resized with letterboxing in the comparison video only.

Example (real motors move; omit `--execute --port ... --allow-holding` for mocks):

```bash
python scripts/hardware/mirror_evaluation.py \
  --checkpoint logs/rsl_rl/so101_picklift_rgb/2026-10-15_18-49-25/model_1999.pt \
  --task lift --curriculum fixed --seed 101 --cube-jitter 0 --viz newton \
  --execute --port /dev/ttyACM0 --allow-holding --alignment-tolerance 12 \
  --gripper-speed 600 --record-video
```

The existing tracking guard remains active. A recording shows observed movement;
it does not certify physical grasp or placement success.
