# Contact-gated rewards and mild cube randomization

The previous grasp reward paid every control step for proximity and partial jaw
closure. It did not measure finger-to-cube contact. A policy could collect that
reward by touching or hovering near the cube without picking it up. Randomizing
the cube does not remove this reward loophole; the reward definition must change.

The scene now has two filtered contact sensors: the fixed gripper body and moving
jaw body, both filtered against the cube. Table contact cannot satisfy those filters.
Both need normal force above 0.02 N, plus the existing centred/closed-gripper geometry.
These are body-level measurements, not individual pad contact-point measurements.

Grasp and lift shaping now pay for **new best quality**, once per episode:

```text
best_t = max(best_(t-1), quality_t)
reward_t = weight * max(quality_t - best_(t-1), 0)
```

The implementation divides progress by the control dt because Isaac Lab's reward
manager multiplies reward terms by dt. Staying in the same grasp, opening and
reclosing, or returning to the same lift height cannot collect these rewards again.
Maximum accumulated grasp credit is 4 per episode. The final held-lift bonus is 240
at the current 0.04 s control dt. Reach-progress shaping and drag/table penalties remain.
This remains dense progress shaping with a sparse completion bonus.

Success now requires both cube contacts, cube centre above 0.47 m, a nearby closed
gripper and low cube speed for at least 0.5 seconds (13 control samples). With the current grasp assist, the critic
has 55 numerical inputs including grasp/lift and closure memory. The actor has
34 camera/proprioception/controller features. Neither contact force nor simulator cube position
is added to the actor. PPO, TD/GAE and Gaussian/entropy exploration are retained.

`--cube-jitter 0.01` randomizes cube reset X/Y independently by ±1 cm and yaw by
±0.15 radians. In the `grasp`/`fixed` stages mass and friction remain fixed so this
checks position-following first. Later `lift`/`place` stages retain their broader
position/yaw and mass/friction randomization when no jitter override is supplied.
Jitter changes reset conditions, not the cube's trajectory during an episode.

The old camera checkpoint `2026-10-08_10-48-22/model_100.pt` achieved 0/10 lifts
in a randomized diagnostic before these edits. It was an early checkpoint, not
evidence that PPO as an algorithm cannot work. Revised rewards/critic inputs require
a **fresh run**; schema `camera_grasp_assist_v7` rejects old checkpoints.

Run one command at a time, in `env_isaaclab`, from this project directory.
Before new training:

```bash
python scripts/train_pipeline.py --stage preflight --headless
```

To inspect the camera-guided reference controller under randomized poses, choose one:

```bash
python scripts/check_randomized_camera.py --cases 5 --seed 202 --cube-jitter 0.01 --headless

python scripts/check_randomized_camera.py --cases 5 --seed 202 --cube-jitter 0.01 --viz newton_gl
```

This is a camera/IK/physics check, not learned-policy evaluation. JSON files are in
`outputs/perception/randomized_camera/`. Five cases are a diagnostic sample, not a
mastery gate. The reference lift target is 8 cm above the observed cube centre to
allow for loaded actuator tracking error; the success-height threshold is unchanged.

After preflight passes, choose one fresh learning command:

```bash
# Headless; virtual camera rendering remains active.
python scripts/train_pipeline.py --stage grasp --policy state --cube-jitter 0.01 \
  --headless --envs 32 --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard

# Newton GUI with all 32 training environments visible.
python scripts/train_pipeline.py --stage grasp --policy state --cube-jitter 0.01 \
  --viz newton_gl --envs 32 --max_visible_envs 32 \
  --lift-iterations 2000 --device cuda:0 --seed 42 --dashboard
```

`state` is the numerical camera-feature MLP, not a simulator-coordinate policy.
In TensorBoard monitor `Grasp/bilateral_contact_fraction`, camera visibility,
held-lift success, progress rewards and TD residual. A higher return without
held lifts is not mastery. The pipeline automatically evaluates three seeds with
50 episodes each after training, forwarding the same jitter setting. Each needs
90% success and at most 1 cm mean table slide before moving to the next stage.

Evaluation now records initial/closest tool-to-cube distance and camera visibility
per episode as well as lift success and slide. These help distinguish following
the observed position from grasping it. They do not prove continuous tracking of
a moving cube or generalization beyond the tested ranges.
