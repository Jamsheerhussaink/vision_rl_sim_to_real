# SO-101 dense-reward vision PPO agent

This agent trains picking and lifting first, followed by placement. It combines an
RGB CNN actor, privileged MLP critic, dense task rewards, bounded Gaussian action
noise, entropy adaptation and the installed RSL-RL PPO implementation of TD/GAE.
Successful startup or a high reward does not prove grasping: measure success over
complete episodes and inspect playback.

## What each component does

| Component | Meaning | Project configuration |
| --- | --- | --- |
| MDP | State, actions, dynamics, rewards and episode endings | SO-101 workcell, 25 Hz policy, 100 Hz physics, 6-second deadline |
| Actor | Chooses six normalized joint actions | 34 numerical features including camera-estimated cube geometry; optional RGB CNN; MLP 256/128/64 |
| Critic, V(s) | Predicts expected discounted future task rewards | MLP 256/128/64, 55 privileged state features |
| r | Immediate task reward | Dense stage rewards, penalties and completion bonus |
| Gamma | Discounts rewards further in the future | 0.99 per 0.04-second policy step |
| TD error | Checks the one-step value prediction | r + gamma * V(next state) - V(current state), with terminal masking |
| GAE lambda | Blends TD errors into advantage estimates | 0.95, using 32-step rollouts |
| PPO | Updates policy using advantage estimates | Clip 0.2, five learning epochs, four minibatches |
| Action noise | Samples different actions during training | Learned per-action log standard deviation: initial 0.4, minimum 0.1, maximum 0.8 |
| Entropy | Encourages the policy to retain randomness | Coefficient starts at 0.01; bounded adaptation between 0.001 and 0.05 |

The actor sees RGB, relative joint positions/velocities, last actions, the filtered
joint targets, cube-to-tool relative position and remaining episode time. Filter targets
expose the controller's memory. Cube-relative position is estimated from the global RGB-D camera.

The critic additionally sees cube/box/tool positions relative to the environment
origin, relative distances, cube velocity/orientation, remaining time, best reach
distance and success-counter state. Environment grid positions no longer change
the task representation. Its scalar output is V(s), not a success probability.

This is a finite-horizon task: timeouts stop return bootstrapping. The actor sees
the time remaining. An unfinished rollout still bootstraps its final state value;
actual terminal episodes do not. The numerical TD/GAE calculation has a CPU test.

## Dense reward design

| Stage | Feedback |
| --- | --- |
| Reach | Direct new closest-distance progress; no repeated proximity reward, overhead waypoint or descent stage |
| Gripper sequence | The neutral action holds the jaws open; closing before centering is penalized |
| Grasp | New best grasp quality, requiring centering, closure and two-sided cube contact |
| Lift | New best lift quality gated by two-sided cube contact; repeated hovering earns no progress |
| Transport | Exponential proximity to the box, gated by elevation and a closed nearby gripper |
| Release | Lowering near the box interior and opening near the cube |
| Complete | One completion bonus on the terminating step |
| Penalties | Cube speed and cumulative tabletop displacement, premature closure, tool penetration, abrupt actions, joint speed and limits |

Stationary hovering above the cube and keeping the jaws open do not earn separate
rewards. Reach-progress credit cannot be collected again by moving away and returning.
The displacement term catches slow dragging that can evade a velocity-only penalty.
Completion rewards receive the same dt scaling as other Isaac Lab reward terms:
weight 6000 at dt 0.04 awards 240 on completion.

Lift success requires cube center above 0.47 m, horizontal tool distance below 0.035 m,
vertical tool distance below 0.045 m, gripper angle below 1.10 rad and cube speed below
0.25 m/s for at least 0.5 seconds with cube-filtered contact on both gripper bodies.
See [contact and randomized training](contact_randomized_training.md) for current commands.

The actor now receives a seven-value global-camera observation: estimated cube-to-tool
XYZ in robot-base axes, sin(4*yaw), cos(4*yaw), geometry confidence and a visibility flag.
The RGB-D estimator uses known yellow colour and 5 cm upright cube geometry. It never
reads simulator cube state. Missing detections return zeros with visibility=0. The numerical
actor has 34 features including proprioception and time; the optional CNN also sees RGB.
The older state-assisted checkpoints are rejected; start a fresh camera-policy run.
See [global camera training](global_camera_training.md) for commands and current limits.

The implementation uses PPO with a learned Gaussian actor, a privileged critic, dense shaping,
TD(lambda) generalized advantage estimates, and a clipped value/policy update. The one-step
TD residual monitored by the dashboard is
`delta = reward + gamma * V(next_state) - V(state)`; the critic learns the bootstrapped return,
while the actor uses the advantage estimate to increase the probability of actions that improve
the lift return. Entropy adapts the bounded action-noise scale during early exploration.

### Reference: the supplied Sutton--Barto PDF

The supplied September 2016 draft, [SuttonBartoRL.pdf](/home/ai/Downloads/SuttonBartoRL.pdf),
provides the theoretical foundation: Chapter 3 covers MDPs and discounted returns,
Chapter 6 covers TD learning, Chapter 12 covers lambda returns and eligibility traces,
and Chapter 13, especially Section 13.5 (printed pages 273--275), covers actor-critic
bootstrapping. Its episodic terminal-state value is zero, matching terminal masking here.
For this task, `r` is immediate reward, `gamma` is the discount factor, and
`V(s) = E[sum_k gamma^k r_(t+k+1) | s_t=s]` is the critic's expected future return.

PPO, GAE, adaptive entropy, reward weights and the lift curriculum are implementation
choices in this project; the book does not prescribe these settings or guarantee convergence
of this neural robot policy. Its exercises and pseudocode are reference material, not
instructions to execute or replace the existing agent.

Placement success additionally requires that sustained lift happened earlier, the
gripper is open and withdrawn, the entire rotated cube fits inside the box walls,
and the cube settles on the box floor for 0.4 seconds. Box dimensions match the
current USD asset and must be updated if that asset changes.

## Entropy and action noise

Entropy adaptation compares the current mean log standard deviation with a target
equivalent to normalized-action std 0.2. When exploration is below that target,
the entropy coefficient increases; above it, the coefficient decreases. The
coefficient and noise bounds prevent unbounded changes. The target is a starting
hyperparameter, not a guarantee of task success. Check all six noise values.

Entropy is measured on the Gaussian before clipping. The simulator clips actions
to [-1, 1] and smooths arm position targets with alpha 0.2. Consequently, inspect
action saturation alongside entropy: a random policy can still produce identical
clipped commands. Evaluation uses deterministic actor outputs without sampling noise.

HER is not enabled. Standard hindsight experience replay requires replay and goal
relabeling; this on-policy PPO agent uses dense rewards and GAE. Adding SAC/TD3 with
HER would be a separate algorithm/environment design, not a PPO toggle.

## Run step by step

Use the existing environment; the project must be installed with its local Isaac Lab dependencies.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl

# Verify a fresh agent starts and updates
python scripts/train.py --task lift --headless --envs 4 --iterations 2 --device cuda:0

# Train Stage 1 without a window
python scripts/train.py --task lift --headless --envs 64 --iterations 2000 --device cuda:0 --dashboard

# Alternative: train with a window
python scripts/train.py --task lift --viz kit --envs 4 --iterations 2000 --device cuda:0
```

Each run creates its own directory under `logs/rsl_rl/so101_picklift_vision/`.
Replace `NEW_RUN_TIMESTAMP` below with the directory from the full new run, not a
two-iteration smoke test. Legacy checkpoints are rejected because the observation
dimensions and distribution parameterization changed. Existing files are preserved.

```bash
# Count 50 complete episodes; default evaluation seed is 123
python scripts/eval_pipeline.py --task lift --checkpoint logs/rsl_rl/so101_picklift_vision/NEW_RUN_TIMESTAMP/model_1999.pt --episodes 50 --headless --output outputs/lift_evaluation.json

# Inspect actual finger grasp and lifting
python scripts/play.py --task lift --checkpoint logs/rsl_rl/so101_picklift_vision/NEW_RUN_TIMESTAMP/model_1999.pt --num_envs 1

# Train placement only after lift evaluation is satisfactory
python scripts/train.py --task place --headless --envs 64 --iterations 2000 --device cuda:0 --checkpoint logs/rsl_rl/so101_picklift_vision/NEW_RUN_TIMESTAMP/model_1999.pt --dashboard
```

Checkpoint transfer loads the actor, critic, optimizer and iteration state. It is
a full training-state continuation; its displayed iteration numbers continue from
the lift run. To evaluate placement, specify its actual saved checkpoint explicitly,
use `--task place`, and change the output path to `outputs/place_evaluation.json`.

## Read the diagnostics

At `http://localhost:6006`, monitor these together:

| TensorBoard metric | What it tells you |
| --- | --- |
| Episode_Reward/grasp_cube, lift_cube | Whether dense grasp/lift terms activate |
| Episode_Termination/lift_success or placed | Mean termination activation, not an independent episode evaluation |
| Loss/entropy | Raw policy differential entropy; negative values are valid |
| Loss/entropy_coefficient | Adaptive entropy bonus weight |
| Loss/noise_std_0 through noise_std_5 | Noise in action order: pan, lift, elbow, wrist flex, wrist roll, gripper |
| Loss/action_saturation | Fraction of sampled rollout action elements outside [-1, 1] |
| Loss/td_error_abs | Mean absolute TD error before the PPO update |
| Loss/value_mean, return_mean | Critic predictions and GAE return targets |
| Loss/explained_variance | How well rollout value predictions track return targets; constant targets report zero |
| Loss/value | Critic training loss |
| Train/mean_reward | Total episode reward, including shaping and penalties |

The terminal displays noise std and success activation and prints small value losses
in scientific notation. Evaluation JSON reports actual completed-episode success
fraction, reward, duration, peak cube elevation and measured tabletop sliding.
Evaluation captures terminal observations before automatic reset.

## Verification and limitations

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test*.py' -v
```

Tests cover real CNN PPO updates and reload, noise bounds, TD/GAE, progress reset,
translation invariance, rotated-cube containment and placement phase requirements.
They do not establish a trained policy's success rate.

The robot USD currently has finger-pad colliders only. Full arm collision coverage,
physical joint zero/direction alignment, real camera preprocessing and actuator timing
remain separate sim-to-real work. Encoder-range midpoints in calibration.py are an
unverified simulation-zero assumption. These changes build a simulation training
agent; they do not validate hardware deployment.

Use [the evidence-gated pipeline](training_pipeline.md) for physical preflight and curriculum advancement.
