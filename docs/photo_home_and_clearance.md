# Measured photo home and table clearance — 2026-10-15

The user confirmed the real arm was in the attached reference-photo pose. Three read-only encoder samples were identical. No motor target or torque register was written.

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

Captured ticks: `[2012, 904, 3096, 2797, 3089, 2037]`.

The shared definition is `src/isaac_vision_rl/utils/home_pose.py`. Simulation uses the existing calibration conversion, including wrist-roll zero 3096 and the gripper's dedicated linear mapping. This replaces visual guesses of the reset angles. The simulation shoulder-lift lower limit was adjusted from -100.000046 degrees to -101 degrees to accommodate the measured -100.546875-degree home. Real calibration limits, motor gains, camera, object layout, and checkpoints were not changed.

## How table clearance works

The mirrored demo first unfolds the elbow by 0.30 rad and reduces wrist flex by 0.15 rad, raising the hand before moving toward the cube. Return reverses that route: retract above the cube, return to the raised home waypoint, then fold into measured home.

The tool frame is not the lowest fingertip. Consequently the demo now targets 1.5 cm above cube centre during grasp instead of placing the tool origin at cube centre. This avoids the collision mesh passing through the tabletop. It changes the scripted demonstration, not PPO reward or actor inputs. Grasp success was not certified by this clearance test.

The IK solver intersects simulator limits with the existing real calibrated joint ranges; it does not silently clip the streamed trajectory. Every outgoing reference is checked against calibration limits and wrist/finger collision-mesh vertices, requiring at least 5 mm above the flat z=0.40 m table. At each 240 Hz physics step, measured link poses are also checked, stopping below 3 mm. These checks cover the wrist, gripper and moving jaw against the table, not self-collision, arbitrary obstacles, or guaranteed real-world clearance.

A complete simulation-only round trip passed. Minimum commanded clearance: 7.738 mm. Minimum measured PhysX clearance: 6.986 mm. Both departure and return stayed above the table in this test. The real travel path has not been executed or certified. Sub-centimetre simulation clearance is sensitive to real calibration and tabletop differences.

The bridge no longer blindly parks on STOP/shutdown. It closes according to its existing torque policy, without automatic motion. The standalone parking helper and bridge's direct parking helper refuse to move from poses farther than five ticks from their home target; they are not general return-path planners. The mirrored demo's checked route is the supported return for its own complete cycles. Interrupted runs stop rather than trying an unverified return.

## Verification

Run jobs sequentially. Neither command below opens a hardware connection:

```bash
conda run --no-capture-output -n env_isaaclab python scripts/demo_mirror_motion.py --sim-only --viz none --cycles 1
conda run --no-capture-output -n env_isaaclab python scripts/demo_mirror_motion.py --sim-only --viz newton --cycles 1
```

The first is headless, the second is GUI; choose one.

```bash
conda run --no-capture-output -n env_isaaclab python -m unittest tests.test_home_pose tests.test_hardware_bridge tests.test_bridge_startup tests.test_evaluation_mirror -q
```

All 40 tests passed. Tests use mock buses and local UDP, not real motors.

Evidence: `outputs/hardware_mapping/photo_home_20261015.json` and `outputs/hardware_mapping/home_clearance_check.json`. Current home preview: `outputs/training/scene_preview.png`.
