# Hardware bridge startup checks

Startup no longer enables motors as a side effect of opening serial. It checks
all motors are torque-disabled, in position mode, error-free and inside recorded
calibration limits. It then reads actual positions, writes those exact holding
targets and verifies target readback, position stability and torque still off.
Only after these checks does it enable torque and verify feedback.

It refuses takeover of an already-powered arm. It does not change PID gains,
operating modes or torque limits to make startup pass. Failed reads do not produce
invented midpoint positions. Partial enable failures attempt to restore disabled
torque and report failures to confirm that state.

These two checks do not open the UDP motion receiver:

```bash
conda activate lerobot
python scripts/hardware/hardware_bridge.py --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031249-if00 --prepare-only
python scripts/hardware/hardware_bridge.py --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031249-if00 --hold-check
```

Preparation seeds and verifies targets without intentionally enabling torque.
The holding check enables torque at newly measured targets and leaves it enabled
on success. Both require the arm to start torque-disabled; do not disable a
load-bearing arm without supporting it. If the arm is already holding, refusal
is expected and does not mean a communication failure.

Validation: eight fake-register startup tests passed, covering ordering, bad
reads, bad targets, motor faults, partial enable failure, powered-arm takeover
and unexpected power during preparation. Five existing calibration/UDP tests
also passed.

Real target preparation passed on 2026-10-13 at approximately 16:01 IST, using
ticks [1992, 2043, 2044, 2053, 1138, 2044]. The later holding-only startup refused
because motors were already powered. Read-only postcheck confirmed all six
motors holding these targets with no reported errors. The user reported no
intervening control action. A separate pan-only check did not reproduce
automatic torque enable on a goal write; the state transition remains
unexplained. Therefore fresh off-to-on startup is not certified on real hardware.

This change does not calibrate simulated angles to physical angles, resolve
shoulder-lift tracking, or authorize the whole-arm mirror demo. The existing
normal bridge shutdown still disables motor torque.
