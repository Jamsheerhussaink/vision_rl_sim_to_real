# Simulation-only global camera workflow

Complete camera-driven picking and lifting in simulation before placement and before
any hardware work. These commands do not open a real camera or communicate with a robot.
Run one simulator command at a time from the project directory.

```text
Simulated global RGB-D camera (512 x 512)
  -> yellow cube segmentation and calibrated depth projection
  -> upright 5 cm cube fit in robot-base coordinates
  -> relative XYZ, orientation, confidence, visible/not-visible
  -> actor + measured robot joint positions/velocities + target memory
  -> bounded joint targets, gripper open/close
  -> physics -> next camera observation

Training only: simulator state -> dense rewards, critic V(s), success scoring
Learning: Gaussian exploration + entropy -> TD/GAE -> clipped PPO updates
Stages: physical/camera preflight -> near grasp -> home-position lift
        -> randomized lift -> placement, with evaluation gates between stages
```

The actor never receives simulator cube coordinates. If a cube cannot be fitted,
the seven camera features are zero and visibility is zero; there is no ground-truth
or stale-position fallback. This does not guarantee that a learned policy will handle
every occlusion. Watch `Perception/cube_visible_fraction` and
`Perception/cube_confidence` in TensorBoard alongside completed lift success.
The global camera is fixed; robot motion changes the robot/cube image, not the camera mount.

`--policy state` means an MLP receiving numerical **camera-derived** features and
proprioception (34 features). It is the first camera-policy baseline to try.
`--policy vision` adds a CNN with a 128 x 128 RGB input, downsampled from the same
512 x 512 sensor. Both use PPO, a privileged training critic, dense shaping rewards
plus a completion reward, bounded Gaussian exploration and entropy regularization.
SAC and HER are not implemented by these options.

Before training, activate the existing environment:

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
```

For the latest reward correction and mild position randomization, use
[contact and randomized training](contact_randomized_training.md).

First inspect the camera-guided reference lift. Choose one command:

```bash
# Headless reference check; JSON and camera screenshots under outputs/perception/.
python scripts/camera_pick_lift.py --headless

# Visual reference check, with one robot for close inspection.
python scripts/camera_pick_lift.py --viz newton_gl
```

This controller uses camera estimates and inverse kinematics, and reacquires the
cube before descent and closure. It is a diagnostic reference, not a trained PPO
policy and not continuous image-servo control during each motion segment.
Its camera-derived approach first raises the open gripper near the base, avoiding
the straight joint interpolation that previously pushed the cube out of reach.
It aborts when required camera estimates are unavailable or IK is unreachable.
Cube ground truth is queried only after motion to score the held lift.

Before the first fresh training run, run the full preflight:

```bash
python scripts/train_pipeline.py --stage preflight --headless
```

This checks tests, table/box collisions, pose/velocity resets, the physical grasp,
camera estimation across four poses, and the camera-guided reference lift. Failed
or changed camera evidence blocks the training pipeline. The clear-view estimator
test hides the robot to isolate geometry; the reference lift keeps the robot visible
and tests detection around the gripper.

After preflight passes, choose **one** fresh near-grasp training mode:

```bash
# Headless, camera still renders. Start with 32 environments for this RGB-D path.
python scripts/train_pipeline.py --stage grasp --policy state --headless \
  --envs 32 --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard

# Newton GUI: train and display all 32 environments.
python scripts/train_pipeline.py --stage grasp --policy state --viz newton_gl \
  --envs 32 --max_visible_envs 32 --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard
```

In Newton, the large GL scene is an inspection view. The actual robot camera image
is in **Streaming View -> Toggle -> Open**, Source Camera `camera`, environment zero.
Camera geometry continues to update even when the streaming panel is hidden.
At [TensorBoard](http://localhost:6006), monitor visibility, noise standard deviation,
TD residual, rewards and held-lift success. Negative reward alone is not a stopping rule.

The pipeline evaluates the finished checkpoint on 50 episodes for each of three
seeds. Each seed needs at least 90% successful held lifts and mean table slide at
most 1 cm. Training completion or a reference lift does not pass this learned-policy gate.

After the grasp gate passes, replace `--stage grasp` with `--stage fixed` in the
same headless or GUI command. That trains from home at fixed cube position. After
that gate passes use `--stage lift` for randomized lift. Only after its gate passes
use `--stage place` (and `--place-iterations 2000`). Keep `--policy state` consistent
across stages. Do not run the next stage while the previous stage is still training.

Old checkpoints use simulator-assisted observations and are rejected by schema
`camera_grasp_assist_v7`. Fresh MLP runs save under
`logs/rsl_rl/so101_picklift_state/<timestamp>/`. Pipeline logs remain under
`outputs/pipeline/`; the checkpoint metadata records `global_rgbd_estimator`.

The verified reference is a fixed-scene pick/lift, not placement mastery. The estimator
currently assumes one upright yellow cube, known size, aligned pinhole RGB-D and
known simulated camera calibration. Multiple cubes, severe occlusion, tilted cubes
and arbitrary objects need additional perception work. No real-world transfer is
claimed by these simulation checks.
