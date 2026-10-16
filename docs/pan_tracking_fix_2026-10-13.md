# Historical script locations

`test_pan_motion.py` is now archived. Restore it using
[archive/README.md](../archive/README.md) before running its commands below.

# SO-101 simulation pan tracking correction

The RGB reset pose placed the hand in contact with the table. The shoulder-pan
target reached the actuator buffer, but the physical joint barely moved. Pan,
elbow and wrist reported actuator saturation in the normal scene.

Controlled +2 degree pan tests (three-second ramp, one-second hold, return):

| Configuration | Measured endpoint change |
| --- | ---: |
| Original reset, normal collisions | 0.016557 degrees |
| Original reset, robot collisions disabled for diagnosis | 1.999088 degrees |
| Original reset, table collisions disabled for diagnosis | 1.999088 degrees |
| Original reset, cube collisions disabled for diagnosis | 0.007696 degrees |
| Raised wrist reset, normal collisions | 1.999222 degrees |

The production RGB reset wrist angle in `src/isaac_vision_rl/utils/pipeline.py`
changed from 0.85 to 0.45 radians. Offline forward kinematics predicts the tool
frame rises from z=0.426094 m to z=0.466214 m, while the table surface is z=0.4 m.
The other home angles, collision settings, actuator limits, rewards and success
criteria were retained. The new angle is a collision-clear simulation reset,
not a measured real-hardware wrist calibration.

The production verification report is
`outputs/hardware_mapping/pan_sim_20261013_153047.json`.
It records normal collisions, no diagnostic overrides, a 1.999222 degree endpoint
and -0.000746 degree return error. Earlier test reports remain preserved.

This verifies a small pan movement, not grasp feasibility or learned-policy
mastery. Full real-to-sim joint zero/sign mapping and real tracking calibration
remain unverified. The earlier real +2.021978 degree test reached +1.494505
degrees and had +0.439560 degree return error. Different starting poses prevent
claiming full dynamic equivalence.

## Pipeline verification

After the fix, both RGB configuration tests passed. A fresh smoke run completed
two PPO updates with four environments, saved `model_1.pt`, and evaluated two
complete episodes with video recording. Evaluation was 0/2 held lifts; this is
pipeline execution evidence, not grasp mastery. Results are in
`outputs/training/live_20261013_153133_531438/evaluation.json`.
No real motor commands were sent during this simulation investigation.

## Reproduce

Run sequentially from the project directory with `env_isaaclab` active.

Headless production pan check:

```bash
python scripts/test_pan_motion.py --mode sim --headless
```

GUI production pan check:

```bash
python scripts/test_pan_motion.py --mode sim
```

Small training/evaluation pipeline smoke test, creating a new run:

```bash
python scripts/train_live_session.py --envs 4 --iterations 2 --episodes 2 --headless
```

Omit `--headless` for simulator windows. These commands do not send real motor
commands. Collision isolation flags on `test_pan_motion.py` are diagnostic only;
do not use disabled collisions as evidence of physical grasp readiness.
