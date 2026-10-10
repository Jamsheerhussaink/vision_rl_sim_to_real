# Evidence-gated SO-101 training pipeline

This pipeline advances only after measured success. Training completion, reward,
critic loss and a passing unit test are not grasp mastery. Full PPO runs are
started by the user; preflight and short startup checks can run first.

## Sequence and commands

Activate the existing environment and stay in the project directory:

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/train_pipeline.py --stage preflight
python scripts/train_pipeline.py --stage status
```

Preflight runs regressions, table/box support and wall-impact tests, repeated cube resets, and a controlled physical grasp. It solves reachable
poses from the USD joint chain, commands open jaws, moves near the cube, closes,
raises the arm, and requires the unassisted dynamic cube to stay above 0.47 m
near the gripper for the last 0.5 seconds. The diagnostic bypasses the learned
policy, not collisions or gravity. No cube attachment or teleport is used.
Its forward-kinematics prediction must agree with simulation within 5 mm.
Failure blocks training. Evidence and camera snapshots are in `outputs/pipeline/`.

The cube rests at 0.425 m on the table and 0.430 m on the box floor. The physical
box has a 4 mm floor in its rigid-body frame; its floor and walls collide in all
stages. Lift training no longer uses a ghost box that objects can pass through.

The stages are:

| Stage | Start and variability | Prerequisite |
| --- | --- | --- |
| `grasp` | Verified near-cube robot pose, open jaws, fixed cube and physics | Passing physical preflight |
| `fixed` | Reference home pose, direct approach to fixed cube | Grasp mastery |
| `lift` | Home pose; randomized cube position/yaw, friction and mass | Fixed-position mastery |
| `place` | Full pick, lift, transport, release and withdrawal | Randomized lift mastery |

The fresh `grasp` actor's mean actions initialize to the verified pose, keeping
the arm near the cube instead of immediately moving toward joint midpoints.
Exploration and PPO updates remain active. Later stages load the learned weights
with exactly the same action mapping; no scripted controller drives training.

Run one stage at a time:

```bash
python scripts/train_pipeline.py --stage grasp --headless --envs 64 --dashboard
python scripts/train_pipeline.py --stage fixed --headless --envs 64
python scripts/train_pipeline.py --stage lift --headless --envs 64
python scripts/train_pipeline.py --stage place --headless --envs 64
```

Use `--viz kit --envs 4` instead of `--headless --envs 64` to train with a window.
The GUI selects the actual task camera; its physical housing is hidden.
The near-cube initialization is an easy starting condition, not a required
above-cube approach trajectory. The later stages retain direct approach guidance.

For a lighter Newton OpenGL viewer, train two environments and display one:

```bash
python scripts/train_pipeline.py --stage grasp --viz newton_gl --envs 2 --max_visible_envs 1
```

This uses the verified PhysX physics and RTX policy camera with a Newton GL GUI.
Newton's camera panel displays the actual policy RGB for environment zero.
The initial 3D view focuses on environment zero with its grid-origin offset;
zoom out manually to see the other visible environments. In Streaming View,
choose Open for a close camera image independently of the overall grid view.
Two environments favor interactive viewing but collect fewer samples per PPO
iteration than the 64-environment headless run; 2,000 iterations is the same
update count, not the same amount of training experience. Evaluation still runs
headless afterward and the same mastery gates apply.

Every stage trains for 2,000 iterations by default, then evaluates its own final
checkpoint on 50 complete deterministic episodes for each seed 101, 202 and 303.
Each seed must achieve at least 90% success and mean tabletop slide <= 1 cm.
The early fixed stages measure repeatability at a fixed position, not generalization.
Only the randomized lift stage measures randomized lift mastery.
Stage transitions use the exact passing checkpoint, not a globally newest run.

The automated full sequence is available when the user wants to run all stages:

```bash
python scripts/train_pipeline.py --stage all --headless --envs 64
```

It stops on the first failed preflight, training process or evaluation gate. It
never starts placement just because lift training reached its iteration limit.

## Failed gates and resuming

Inspect `outputs/pipeline/<stage>_training.log`, the three evaluation reports,
and `<stage>_gate.json`. Watch the candidate using the same curriculum:

```bash
python scripts/play.py --task lift --curriculum grasp --checkpoint EXACT_CHECKPOINT --num_envs 1
```

A passing near-grasp policy can still fail from home. A passing fixed-position
policy can still fail randomization. Diagnose the failed stage before continuing.
For a candidate you want to reevaluate:

```bash
python scripts/train_pipeline.py --stage evaluate --evaluate-stage lift --checkpoint EXACT_CHECKPOINT
```

To continue a failed stage's compatible checkpoint, provide `--checkpoint` with
that individual training stage. Its preceding-stage gate still applies. Checkpoint
hashes and configuration/asset fingerprints invalidate stale gate evidence.
Old v3 policies are incompatible with the corrected anchored workcell and v4
training objective; keep them as historical results and start this pipeline fresh.

## Why the workcell and feedback changed

A controlled probe showed `is_fixed_base=False` in the imported robot: the base
could move instead of being mounted to the table. The pipeline uses a separately
built PhysX-compatible asset and `fix_root_link=True`, preserving `usd/robot.usd`.
The asset builder strips conflicting Newton metadata, decomposes gripper mesh
collisions, aligns the moving silicone pad with the finger tip and moves the tool
frame to the grasp cavity. The gripper opens to 1.20 rad and uses a separate
0.20 Nm torque limit. The controlled test held the cube at ~0.96 rad, so the
held-gripper gate is <1.10 rad rather than the old incompatible <0.65 rad.
The installed Isaac Lab version uses xyzw quaternions; the -30 degree base yaw
is retained in that convention.

Repeated cube-proximity reward encouraged stationary behavior. It is disabled;
direct new-distance progress, centered closure, elevation and completion provide
feedback. Premature closure, dragging, displacement, table penetration, abrupt
actions and joint limits remain penalized. Neutral gripper output stays open.
These changes make the objective more useful, but only evaluation proves mastery.

## Limits

Lift success now requires two-sided cube contact and a sustained 0.5-second elevated hold.
Check playback and the contact metric for actual finger grasping. Placement must follow a prior sustained lift and a settled,
released cube inside the box. Passing simulation stages does not certify real-robot
safety or vision-only deployment: the current actor also receives the privileged
cube-to-tool position. Camera estimation, measured calibration, actuator limits
and supervised real-robot validation are later gates, not automatic pipeline steps.

## Numerical failure diagnosis and stability correction

The previous grasp run stopped at iteration 798 because arm joint positions,
velocities and link poses became NaN. A bounded-random-action reproduction
failed at step 4,315 with finite base, cube and action targets; this identifies
articulation dynamics as the failing layer. It is not evidence of broken TD/GAE.

The derived asset now gives the virtual tool frame positive mass (0.001 kg) and
inertia (1e-7 kg m^2 per axis). The arm has a 1.5 rad/s target slew limit,
16 position and 4 velocity solver iterations, and small motor armature values.
The reset synchronizes observed filtered targets with the actual filter state.
These are simulation-stability settings, not measured real-robot calibration.

The corrected configuration passed 12,000 bounded-random steps across 64
parallel environments (768,000 transitions), exceeding the failed reproduction.
This physics stress test caches RGB between renders for speed; full-rate camera
and PPO smoke checks are separate. Passing these checks does not guarantee that
a new 2,000-iteration policy will master grasping or never encounter another issue.

If dynamics become invalid, `physics_health` raises with the affected environment
IDs and fields and writes `outputs/pipeline/invalid_physics_state.json`.
No NaNs are replaced with zeros and no failed trajectories count as mastery.
Keep the diagnostic and checkpoint; do not repeatedly restart an unexplained failure.

## Editor diagnostics

Workspace settings select `/home/ai/miniconda3/envs/env_isaaclab/bin/python`,
provide Isaac Lab source/Kit extension paths, and align formatting with the
120-character project limit. Type checking remains enabled. If VS Code retains
an older interpreter, run **Python: Select Interpreter**, choose that path, then
**Developer: Reload Window**. The pipeline files pass Ruff, Flake8 and Pyright;
runtime-supported stateful manager classes are explicitly adapted at their
framework typing boundary rather than disabling diagnostics globally.

## Actor comparison and the current camera observation

The completed CNN grasp baseline achieved 0/150 successful lifts. Its reports
are preserved in `outputs/pipeline/vision_baseline/` and its checkpoints remain
under `logs/rsl_rl/so101_picklift_vision/`.

Use `--policy state` for a fresh MLP actor with 32 numerical inputs. The camera
estimator processes RGB-D and supplies cube geometry to this actor. It does not
feed raw pixels directly to the MLP. `--policy vision` additionally feeds RGB
through a CNN. Both actor variants use camera estimates instead of simulator cube pose. PPO, exploration,
critic inputs, rewards, physics, initial joint-target mean and success gates
are unchanged. This is an architecture/input ablation, not a SAC implementation.

```bash
# Headless comparison: same 64-env transition budget as the CNN baseline.
python scripts/train_pipeline.py --stage grasp --policy state --headless --envs 64 --lift-iterations 2000 --dashboard

# Newton comparison: 32 training environments, all visible.
python scripts/train_pipeline.py --stage grasp --policy state --viz newton_gl --envs 32 --max_visible_envs 32 --lift-iterations 2000 --dashboard
```

Do not pass a CNN checkpoint to this experiment. Its input-layer dimensions are
incompatible; checkpoint loading explicitly rejects cross-variant loading.
MLP checkpoints use `logs/rsl_rl/so101_picklift_state/<timestamp>/`; the training
log is `outputs/pipeline/grasp_state_training.log`. Evaluation reports use
`grasp_state_eval_<seed>.json`, so they do not overwrite CNN evaluation results.
The variant gate is `grasp_state_gate.json`; the canonical stage gate also records
which policy produced the result. Advance with the same `--policy state` only
if its success gate passes. Checkpoints and gates are never evidence that the
architecture itself is superior without comparable measured task success.

```bash
python scripts/play.py --task lift --curriculum grasp --policy state --checkpoint EXACT_STATE_CHECKPOINT --num_envs 1
```

At 2,000 iterations, 64 environments collect twice the transitions of 32.
Use equal transition budgets and evaluation conditions for a fair comparison.
The `state` option is now a numerical camera-feature policy, despite its legacy
CLI name. The old 28-feature state-assisted checkpoints are incompatible. Neither
variant has yet demonstrated learned camera-driven lift mastery or real-robot readiness.

## Fixed cube reset correction

Disabling pose randomization must not disable resetting the cube. The earlier
fixed-stage implementation removed the cube reset event, leaving cubes displaced
between episodes after robot contact. This invalidated the assumption that the
CNN baseline trained under fixed starting conditions; its failed result alone
cannot establish that CNN architecture was the cause.

The reset event is now always active as `reset_cube_pose`. `grasp` and `fixed`
use empty pose/velocity variation ranges, restoring the default position,
orientation and zero velocity on every episode reset. `lift` and `place` retain
random position/yaw offsets through the same reset event. Cube motion *within*
an episode can still differ because stochastic policies interact with it.

The physical regression displaces and rotates cubes and gives them nonzero
velocity, then verifies five repeated full and partial resets over four
environments. It also checks that a partial reset leaves the other cubes alone.
The running old simulation cannot adopt this correction: stop it and restart
fresh after preflight. Do not certify its data as a fixed-start comparison.
