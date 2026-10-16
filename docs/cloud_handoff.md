# SO-101 project handoff for Codex Cloud

This is project context, not a full chat-history import.

## Repository and scope

- GitHub: Jamsheerhussaink/vision_rl_sim_to_real, branch isaaclab.
- Local project: /home/ai/jamshi/isaac_vision_rl.
- Current aim: RGB-only SO-101 pick-and-lift, then randomized lifting, then placement.
- Teach step by step; distinguish reward from actual grasp/held-lift success.
- Preserve checkpoints. User performs hardware commands on a separate computer.
- Latest source changes are local and must be reviewed/committed before cloud uses them.

## Active simulation

Isaac Lab/PhysX with RSL-RL PPO. RGB 424x240, 60 Hz control/rendering,
240 Hz physics. Actor image is aspect-preserving letterboxed 128x128.
Actor gets RGB and joint/controller state, no depth geometry or capture latch.
Simulator object state is training critic/reward/evaluation information only.
Camera uses nominal D455 pinhole optics; user-supplied lens parameters remain
unverified for resolution/stream and are not applied. Latest camera pitch is
48 degrees downward and aim yaw 11 degrees from +Y toward +X. Camera height
and horizontal base distance are approximately 30 cm. Home gripper is closed.
Latest scene image: outputs/rgb_scene_reference.png (outputs are Git-ignored).

## Evidence and Completed Milestones

- **Hardware Deployment Bridge Operational**: Implemented 60 Hz UDP bridge (`scripts/hardware/hardware_bridge.py`) connecting Isaac Sim / Isaac Lab to Feetech STS3215 servo bus on `/dev/ttyACM0`. Includes 80-tick/step slew rate limiting, watchdog timeout (150 ms), and safe torque shutdown.
- **Sim-to-Real Kinematic Parity Verified**: Built diagnostic suite (`scripts/hardware/mirror_evaluation.py` and `scripts/check_sim_real_mirroring.py`). All 6 joints verified with sub-degree mean tracking error in simultaneous sim/hardware sweeps (`outputs/hardware_mapping/sim_real_mirror_report_20261015_171424.json`).
- **Wrist Roll Zero Alignment Resolved**: Fixed 90° gripper tilt and 360° rollover spin. Measured physical neutral at tick 3096 (~272°), updated `FollowerArmCalibration` in `src/isaac_vision_rl/utils/calibration.py` to support `zero_ticks`, and updated `my_awesome_follower_arm.json` line 36 with `"zero_ticks": 3096`.
- **Gripper Trajectory Streaming Fixed**: In `scripts/demo_mirror_motion.py`, resolved gripper not closing by streaming pure `target_q` reference trajectories directly at 60 Hz rather than lagging simulated PhysX actuator states (`robot.data.joint_pos`), ensuring decisive clamping at 0.0 rad (2029 ticks) and clean release at 1.2 rad (3545 ticks).

## Real robot setup and verified reads

- Host computer: `/home/ai/jamshi/isaac_vision_rl`, conda environments `env_isaaclab` (Isaac Lab, PyTorch) and `lerobot` (Feetech SCServo SDK).
- Arm: SO-101 6-DoF follower arm on `/dev/ttyACM0` at 1,000,000 baud (Feetech STS3215 servos, IDs 1–6).
- External Power: Servos require 7.4V–12V DC power brick; 5.0V USB bus power causes undervoltage under load.
- Motor Limits and Offsets (`my_awesome_follower_arm.json`):
  - `shoulder_pan` (ID 1): [766, 3354], zero: 2048
  - `shoulder_lift` (ID 2): [0, 4093], zero: 2048
  - `elbow_flex` (ID 3): [4, 4093], zero: 2048
  - `wrist_flex` (ID 4): [33, 3000], zero: 2048
  - `wrist_roll` (ID 5): [0, 4095], zero: 3096 (calibrated physical neutral)
  - `gripper` (ID 6): [2029, 3545], 2029 = closed (0.0 rad), 3545 = open (1.2 rad)

## Next work

1. Execute full motion mirroring demo (`scripts/demo_mirror_motion.py --viz newton`) with physical 5 cm cube to verify real grasp friction and lift stability.
2. Integrate RealSense D455 physical camera stream via `scripts/hardware/realsense_rgb_preview.py` to benchmark physical image distribution against the synthetic 128x128 letterbox observation space.
3. Deploy trained RGB RL policy (`scripts/hardware/stream_to_robot.py`) in closed-loop inference.

