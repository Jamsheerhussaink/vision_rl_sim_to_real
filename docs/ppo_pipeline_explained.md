# Your SO-101 PPO pipeline, explained from first principles

Inspected on 10 October 2026. This document describes the current repository and
its installed RSL-RL implementation. It does not change training code or weights.
Small numerical examples illustrate the equations; they are not training measurements.
Python examples below can run sequentially in a notebook using CPU PyTorch.

## 1. What your agent is learning

The present objective is to approach a yellow cube using a global RGB-D camera,
capture it between the fingers, lift it and maintain a stable held lift. Placement
is a later stage. PPO means Proximal Policy Optimization: it learns a policy by
collecting experience and making controlled updates to that policy.

Your system combines:

- A vision actor: proposes arm and gripper actions.
- A privileged critic: predicts future return using extra simulator information.
- A camera-triggered gripper controller: can override the policy's jaw command.
- Physics: turns motor targets into actual robot and cube movement.
- Rewards and success conditions: score progress and identify completed tasks.

Thus the evaluated system is a hybrid policy and controller. The present results
do not prove that PPO independently learned every part of gripper closure.

## 2. Where to find each part

Paths below are relative to the project root.

| File | Responsibility |
|---|---|
| [scripts/train.py](../scripts/train.py) | Parse options and dispatch to Isaac Lab's training backend |
| [env_cfg.py](../src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py) | Scene, observations, action configuration, reward weights, timing |
| [observations.py](../src/isaac_vision_rl/tasks/pickplace/mdp/observations.py) | Camera estimation, proprioception and critic features |
| [actions.py](../src/isaac_vision_rl/tasks/pickplace/mdp/actions.py) | Arm filtering, grasp assist and optional wrist control |
| [rewards.py](../src/isaac_vision_rl/tasks/pickplace/mdp/rewards.py) | Progress, penalties and strict held-lift condition |
| [rsl_rl_ppo_cfg.py](../src/isaac_vision_rl/tasks/pickplace/config/so101/agents/rsl_rl_ppo_cfg.py) | Actor, critic and PPO hyperparameters |
| [rl_agent.py](../src/isaac_vision_rl/utils/rl_agent.py) | Exploration adaptation, diagnostics and checkpoint compatibility |
| [pipeline.py](../src/isaac_vision_rl/utils/pipeline.py) | Curriculum and evidence gates |
| [scripts/eval_pipeline.py](../scripts/eval_pipeline.py) | Completed-episode evaluation, video and traces |

The PPO equations themselves run in the installed package:
`/home/ai/miniconda3/envs/env_isaaclab/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py`.
`ExplorationPPO` inherits these equations rather than replacing TD learning or GAE.

## 3. Execution flow: one training iteration

```text
train.py reads options
  → select lift/place task and vision/state actor
  → set curriculum, jitter and controller options
  → Isaac Lab creates environment and RSL-RL runner
  → observe all environments
  → sample actions and predict values
  → apply clipping, filtering and controller overrides
  → advance physics and compute reward/episode endings
  → store transition; repeat for 32 steps per environment
  → calculate TD errors, GAE advantages and critic targets
  → perform 5 epochs × 4 minibatches of PPO optimization
  → collect a new rollout
  → save every 50 iterations and at the end
```

An episode is one attempt between resets. A rollout is a fixed amount of recently
collected experience; it may contain parts of several episodes. An iteration
contains a rollout and optimization, not one robot action or one episode.

At 48 environments, your batch sizes are:

```python
num_envs = 48
steps_per_env = 32
rollout_size = num_envs * steps_per_env
minibatch_size = rollout_size // 4
optimizer_steps = 5 * 4
assert (rollout_size, minibatch_size, optimizer_steps) == (1536, 384, 20)
```

The last rollout step is not automatically an episode ending. If the episode
continues, the critic bootstraps the remaining future return from its final observation.

## 4. MDP: state, observation, action and transition

An MDP (Markov Decision Process) describes the learning problem:

\[
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
\]

Here \(\mathcal S\) is the state space, \(\mathcal A\) the action space,
\(P\) the transition dynamics, \(R\) the reward rule and \(\gamma\) the discount factor.
The full state includes robot/cube motion, contacts, controller memory, reward
progress memory and time remaining. Remembering best progress matters because
the reward depends on whether an improvement has already happened.

At time step \(t\), the actor sees observation \(o_t=h(s_t)\), chooses action
\(a_t\), and physics produces \(s_{t+1}\sim P(\cdot\mid s_t,a_t)\).
The function \(h\) selects what the camera and robot measurements reveal.

The actor does not see the complete physical state. Camera occlusion makes this
a partially observable control problem, even though the simulator has full state.
Previous actions and controller features help, but do not guarantee a Markov observation.

Your physics step is 0.01 seconds, with four physics steps per policy step:

\[
\Delta t=4(0.01)=0.04\text{ s},\qquad f=1/\Delta t=25\text{ Hz}.
\]

```python
physics_dt = 0.01
policy_dt = 4 * physics_dt
assert policy_dt == 0.04
assert round(6.0 / policy_dt) == 150  # maximum steps in a six-second episode
```

Curriculum changes task difficulty; it is still reinforcement learning. Your
stages are grasp → fixed lift → randomized lift → placement. Fixed/grasp stages
disable cube mass/friction randomization. In randomized lift, position jitter is
applied at resets, while configured material/mass variation occurs at startup.
It is inaccurate to say every domain parameter changes every episode.

## 5. Observations: what the robot knows

The `policy` group concatenates 34 numbers:

| Component | Count | Meaning |
|---|---:|---|
| `joint_pos` | 6 | Joint positions relative to defaults |
| `joint_vel` | 6 | Joint velocities |
| `actions` | 6 | Previous environment action |
| `filtered_targets` | 6 | Arm/jaw references after controller processing |
| `assist_state` | 2 | Capture latch and closure-pause information |
| `camera_cube` | 7 | Estimated cube geometry, confidence and visibility |
| `time_remaining` | 1 | Fraction of episode time remaining |

The separate `image` group contains RGB with shape `(N, 3, 128, 128)`, where
`N` is the number of parallel environments. Pixel values are divided by 255.
The actor receives both groups in vision mode.

`camera_cube_observation()` uses rendered RGB-D, camera intrinsics/extrinsics and
robot pose to estimate the cube in robot-base coordinates. Its output is:

```text
[dx, dy, dz, sin(4*yaw), cos(4*yaw), confidence, valid]
```

The position offset is \(\Delta p=\hat p_{cube}-p_{tool}\), where the hat means
estimated from the camera. For example:

```python
import torch
cube_estimate = torch.tensor([0.20, 0.10, 0.425])
tool_position = torch.tensor([0.18, 0.12, 0.470])
offset = cube_estimate - tool_position
print(offset)  # approximately [0.02, -0.02, -0.045] metres
```

The factor four in yaw features represents square symmetry: yaw and yaw plus
90 degrees describe equivalent orientations of a square cube.

```python
import math
yaw = math.radians(20)
yaw_features = [math.sin(4 * yaw), math.cos(4 * yaw)]
```

A missing cube produces zero features and `valid=0`; this function does not fall
back to simulator cube position. It does use simulation camera/robot transforms,
which would need calibrated extrinsics and robot kinematics on real hardware.

The critic group has 55 numbers. It includes robot/controller measurements plus
privileged cube/box/tool positions, cube velocity/orientation and task memory.
This is asymmetric actor–critic training: privileged state teaches the critic,
while action selection uses deployment-oriented camera and robot inputs.

`cube_ee_relative()` still exists in the file, but the current actor observation
configuration uses `camera_cube_observation()` instead. A function's existence
does not mean it is active in training.

## 6. Policy network and exploration

The actor is the decision-making network. In vision mode it uses three CNN
layers with channels `[32, 64, 64]`, kernels `[8, 4, 3]` and strides `[4, 2, 1]`,
then combines image features with numerical features. Its MLP hidden widths are
`[256, 128, 64]`. MLP means a network of fully connected layers. ELU is its activation.

A layer calculates \(h=\operatorname{ELU}(Wx+b)\): \(x\) is its input,
\(W\) its weights and \(b\) its biases. These parameters are learned.

```python
layer = torch.nn.Linear(34, 256)
features = torch.randn(1, 34)
hidden = torch.nn.functional.elu(layer(features))
assert hidden.shape == (1, 256)
```

This is one illustrative numerical layer, not the complete vision actor.
The actor predicts six action means \(\mu_\theta(o)\). During training it samples:

\[
a=\mu_\theta(o)+\sigma\odot\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
\]

\(\theta\) denotes actor parameters, \(\sigma\) the learned per-action noise,
\(\epsilon\) standard Gaussian noise, and \(\odot\) elementwise multiplication.

```python
mean = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.1, -0.4]])
std = torch.full_like(mean, 0.4)
distribution = torch.distributions.Normal(mean, std)
sample = distribution.sample()
log_prob = distribution.log_prob(sample).sum(dim=-1)
```

Your initial standard deviation is 0.4, or 0.15 in the grasp curriculum. Its
learned bounds are 0.1 to 0.8. The grasp initialization also uses a physically
verified initial arm pose; this is an initial bias, not evidence of learned mastery.

Deterministic evaluation uses the actor mean. `--sample-actions` instead requests
stochastic evaluation. Do not mix these when comparing models.

## 7. Actions: policy output versus actual motor commands

Your six outputs correspond to shoulder pan, shoulder lift, elbow flex, wrist
flex, wrist roll and gripper. They are position-reference inputs, not direct torque.
The environment wrapper clips raw actions to `[-1, 1]`.

For a clipped arm action \(u\), joint-limit mapping is:

\[
q^*=q_{min}+\frac{u+1}{2}(q_{max}-q_{min}).
\]

\(q^*\) is the desired angle and \(q_{min},q_{max}\) its soft limits.

```python
u = 0.5
q_min, q_max = -1.0, 1.0
target = q_min + (u + 1) * 0.5 * (q_max - q_min)
assert target == 0.5
```

Baseline EMA filtering uses \(\alpha=0.2\):

\[
\bar q_t=\alpha q_t^*+(1-\alpha)\bar q_{t-1}.
\]

The bar identifies a filtered reference. A slew limit then restricts its change
to \(v_{max}\Delta t\), with baseline \(v_{max}=1.5\) rad/s.

```python
previous = 0.0
desired = 1.0
ema = 0.2 * desired + 0.8 * previous
max_delta = 1.5 * 0.04
processed_target = previous + max(-max_delta, min(max_delta, ema - previous))
assert abs(processed_target - 0.06) < 1e-8
```

The nominal jaw mapping is:

\[
q^*_{jaw}=\operatorname{clip}(1.2u+1.2,0,1.2).
\]

```python
def nominal_jaw_target(u):
    return max(0.0, min(1.2, 1.2 * u + 1.2))
assert nominal_jaw_target(-1) == 0.0
assert nominal_jaw_target(0) == 1.2
```

But with assistance enabled, `CameraGripperAction` overrides this nominal command.
Capture requires a finite estimate, visibility, confidence at least 0.65, tool-down
orientation, XY offset within 0.020 m and Z offset within 0.025 m for two steps.
It then latches closure and pauses arm references for 0.6 seconds in baseline mode.
Occlusion alone does not reopen a latched grasp.

Optional full smooth mode changes acceleration, speed, closure ramp and pause.
Wrist-only mode preserves baseline approach/closure and holds measured wrist roll
after capture. Holding this joint does not hold the entire tool orientation.
These limits constrain references; actual tracking and forces require measurement.

## 8. Rewards: dense progress plus sparse completion

Your rewards are neither entirely dense nor entirely sparse. New approach,
grasp/lift quality and penalties provide frequent feedback. Completion is a sparse event.
Isaac Lab computes:

\[
r_t=\Delta t\sum_i w_i f_i(s_t,a_t).
\]

\(f_i\) is a term value, \(w_i\) its weight and \(r_t\) the scalar step reward.

| Active lift term | Weight |
|---|---:|
| Reach progress | 3 |
| Grasp progress | 4 |
| Lift progress | 8 |
| Completed held lift | 6000 |
| Premature closure | -3 |
| Table drag | -5 |
| Table displacement | -12 |
| Table collision | -0.5 |
| Action change | -0.02 |
| Joint velocity | -0.01 |
| Joint limits | -0.05 |

The manager scales by time, so a completion Boolean contributes
\(6000\times0.04=240\) reward on its successful step, before reset.

```python
completion_reward = 6000.0 * 0.04 * 1.0
assert completion_reward == 240.0
```

Grasp/lift progress functions pay only a new best score:

\[
\Delta g_t=\max(g_t-g_{best},0),\qquad
g_{best}\leftarrow\max(g_{best},g_t).
\]

They return `progress / env.step_dt`, so grasp contribution is
\(\Delta t\,4\,\Delta g_t/\Delta t=4\Delta g_t\).

```python
best = 0.6
score = 0.8
progress = max(score - best, 0.0)
grasp_reward = 0.04 * 4.0 * (progress / 0.04)
best = max(best, score)
assert abs(grasp_reward - 0.8) < 1e-6
assert max(score - best, 0.0) == 0.0  # no repeat reward for staying here
```

Approach progress has its own distance-based formula, without this `1/dt`
cancellation. Check each function rather than treating all weights identically.
Progress memory resets per episode. This reduces hovering/repeated-touch farming,
but reward design still requires independent success and video checks.

## 9. Return and the value network

Return means accumulated discounted future reward:

\[
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},\qquad\gamma=0.99.
\]

\(T\) is the terminal step and \(k\) the number of steps into the future.

```python
rewards = [0.2, 0.4, 2.0]
gamma = 0.99
G = sum(gamma**k * r for k, r in enumerate(rewards))
assert abs(G - 2.5562) < 1e-6
```

The critic predicts \(V_\phi(c_t)\approx\mathbb E[G_t\mid c_t]\), where
\(c_t\) is its privileged observation and \(\phi\) its parameters. It uses
MLP hidden widths `[256, 128, 64]` and produces one scalar per environment.
It does not choose a robot action. Your PPO has no separate learned Q network.

The Bellman idea decomposes future return into immediate reward plus remaining value:

\[
V^\pi(s_t)=\mathbb E[r_t+\gamma V^\pi(s_{t+1})].
\]

```python
reward, next_value = 0.2, 1.2
one_step_target = reward + gamma * next_value
assert abs(one_step_target - 1.388) < 1e-6
```

Numeric observations are normalized by the model's running normalizers. Rewards
are not thereby converted into normalized pixels or guaranteed to have unit scale.

## 10. TD error: a one-step prediction discrepancy

Your `ExplorationPPO.compute_returns()` logs:

\[
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t.
\]

\(d_t=1\) marks an episode ending, so no value is taken from the reset episode.
\(V_t\) is the critic's old prediction. Positive TD error means the reward plus
next prediction exceeded that old prediction.

```python
reward, value, next_value = 0.2, 1.0, 1.2
done = False
delta = reward + 0.99 * (1 - int(done)) * next_value - value
assert abs(delta - 0.388) < 1e-6
terminal_delta = 2.0 - 1.5
assert terminal_delta == 0.5
```

`td_error_abs` is \(N^{-1}\sum_t|\delta_t|\), the mean absolute error across
stored transitions. Here \(N\) is the number of samples.

```python
td_errors = torch.tensor([0.388, -0.2, 0.5])
td_mae = td_errors.abs().mean()
```

Your environment sets `is_finite_horizon=True`: the six-second deadline is part
of the observed task. The wrapper does not supply continuing-task timeout
bootstrapping for it. Rollout-boundary bootstrapping still occurs when an episode
has not ended. These are different kinds of boundaries.

## 11. Advantages and GAE: credit for an action

The conceptual advantage is:

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

\(Q\) means expected return after that particular action; \(V\) averages over
the policy's actions from the state. Your code estimates advantages without
training a separate Q network.

Generalized Advantage Estimation (GAE) recursively combines TD errors:

\[
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\qquad\lambda=0.95.
\]

The hat means an estimate. Lambda controls how much later TD errors contribute,
balancing reliance on critic predictions against longer sampled outcomes.
The critic target is \(\hat R_t=\hat A_t+V_{old,t}\).

```python
rewards = [0.2, 0.4, 2.0]
values = [1.0, 1.2, 1.5]
dones = [False, False, True]
advantages, targets = [0.0] * 3, [0.0] * 3
gae = 0.0
for t in reversed(range(3)):
    next_value = values[t + 1] if t < 2 else 0.0
    mask = 1.0 - float(dones[t])
    delta = rewards[t] + 0.99 * mask * next_value - values[t]
    gae = delta + 0.99 * 0.95 * mask * gae
    advantages[t] = gae
    targets[t] = gae + values[t]
print(advantages)  # [1.474512625, 1.15525, 0.5]
print(targets)     # [2.474512625, 2.35525, 2.0]
```

At the last step: delta and advantage are `2 - 1.5 = 0.5`.
At the preceding step: delta is `0.4 + 0.99*1.5 - 1.2 = 0.685`;
advantage is `0.685 + 0.99*0.95*0.5 = 1.15525`.
This is how later success teaches earlier approach actions.

Return targets are bootstrapped GAE estimates, not always full Monte Carlo returns.
Advantages are normalized for the actor:

\[
\tilde A_t=\frac{\hat A_t-\operatorname{mean}(\hat A)}{\operatorname{std}(\hat A)+10^{-8}}.
\]

```python
a = torch.tensor(advantages)
normalized_advantages = (a - a.mean()) / (a.std() + 1e-8)
```

This changes scale and centres the batch. The normalized sign is relative to
the rollout mean, not an absolute success label.

## 12. PPO actor loss: change probabilities carefully

The runner stores old log-probabilities when collecting experience. Updating
recalculates the probability of those same stored actions under the new actor.
For independent Gaussian actions, log-probabilities add across six dimensions:

\[
\log\pi(a\mid o)=\sum_j\left[-\frac{(a_j-\mu_j)^2}{2\sigma_j^2}
-\log\sigma_j-\tfrac12\log(2\pi)\right].
\]

The index \(j\) identifies an action component. PyTorch implements this with
`Normal.log_prob(action).sum(dim=-1)` as in section 6.

The likelihood ratio is:

\[
\rho_t=\exp(\log\pi_\theta(a_t\mid o_t)-\log\pi_{old}(a_t\mid o_t)).
\]

```python
old_log_prob = torch.tensor([-2.0])
new_log_prob = torch.tensor([-1.8])
ratio = torch.exp(new_log_prob - old_log_prob)
print(ratio)  # approximately 1.2214
```

The actor loss, minimized by the optimizer, is:

\[
L_\pi=-\mathbb E\left[\min\left(\rho_t\tilde A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\tilde A_t\right)\right],
\qquad\epsilon=0.2.
\]

Expectation is implemented as a minibatch mean. Positive advantage encourages
making an action more likely; negative advantage encourages making it less likely.
The clipped expression stops rewarding certain overly large probability changes.

```python
ratio = torch.tensor([1.5, 0.5])
advantage = torch.tensor([2.0, -2.0])
unclipped = ratio * advantage
clipped = ratio.clamp(0.8, 1.2) * advantage
actor_loss = -torch.minimum(unclipped, clipped).mean()
assert abs(actor_loss.item() + 0.4) < 1e-6
```

For the first sample the benefit is capped at `1.2*2 = 2.4`; for the negative
sample PPO uses the more conservative `0.8*(-2) = -1.6` term.
Your upstream code uses `max(-A*ratio, -A*clipped_ratio)`, which is mathematically
equivalent. Clipping does not guarantee that all ratios remain in `[0.8, 1.2]`.

PPO stores raw sampled actions. The environment clips and overrides them later.
Entropy and action probabilities therefore describe the raw Gaussian, not the
final filtered motor targets. In particular the assisted gripper can ignore a
sampled jaw action. Your `action_saturation` diagnostic helps reveal this mismatch.

## 13. Critic loss and entropy bonus

Basic value fitting minimizes squared return-prediction error:
\(L_V=\mathbb E[(V_\phi-\hat R)^2]\). Your implementation uses clipped value fitting:

\[
V_{clip}=V_{old}+\operatorname{clip}(V_\phi-V_{old},-\epsilon,\epsilon),
\]
\[
L_V=\mathbb E[\max((V_\phi-\hat R)^2,(V_{clip}-\hat R)^2)].
\]

```python
old_value = torch.tensor([1.0])
new_value = torch.tensor([1.5])
target = torch.tensor([2.0])
clipped_value = old_value + (new_value - old_value).clamp(-0.2, 0.2)
value_loss = torch.maximum((new_value - target).square(),
                           (clipped_value - target).square()).mean()
assert abs(value_loss.item() - 0.64) < 1e-6
```

Gaussian entropy is:

\[
H=\sum_j\left(\log\sigma_j+\tfrac12\log(2\pi e)\right).
\]

It measures distribution spread. It is differential entropy and can be negative.

```python
dist = torch.distributions.Normal(torch.zeros(6), torch.full((6,), 0.4))
entropy = dist.entropy().sum()
```

The minimized combined loss is:

\[
L=L_\pi+c_VL_V-c_HH.
\]

Your value coefficient \(c_V=1\), and initial entropy coefficient \(c_H=0.01\).
The minus sign encourages exploration by making larger entropy reduce the loss.

```python
combined_loss = actor_loss + 1.0 * value_loss - 0.01 * entropy
```

`ExplorationPPO.update()` adapts the entropy coefficient using average per-action
entropy, targeting std 0.2 (0.15 in grasp mode):

\[
c_H' = \operatorname{clip}\left(c_H\exp[\operatorname{clip}(
0.05(H_{target}-H_{actual}),-0.1,0.1)],0.001,0.05\right).
\]

```python
current_std, target_std = 0.1, 0.2
constant = 0.5 * math.log(2 * math.pi * math.e)
target_h = math.log(target_std) + constant
actual_h = math.log(current_std) + constant
change = max(-0.1, min(0.1, 0.05 * (target_h - actual_h)))
entropy_coefficient = max(0.001, min(0.05, 0.01 * math.exp(change)))
assert entropy_coefficient > 0.01
```

This increases exploration pressure when spread is below the target. It does not
force the robot to try meaningful new grasp strategies, nor guarantee that every
joint's std equals 0.2. No HER, SAC or RND intrinsic bonus is configured here.

## 14. Backpropagation and policy updates

A simple gradient-descent equation is:

\[
\theta_{new}=\theta_{old}-\eta\nabla_\theta L.
\]

\(\eta\) is the learning rate and the gradient tells how parameters affect loss.
Your actual optimizer is Adam, which uses moving gradient statistics rather
than applying this elementary update directly. The learning rate is fixed at 0.0003.

```python
parameter = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.Adam([parameter], lr=3e-4)
loss = (parameter - 3.0).square().mean()
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
optimizer.step()
assert parameter.item() > 1.0
```

In the actual PPO code, actor and critic parameters share an optimizer, with
their gradient norms clipped separately to 1.0 before `optimizer.step()`.
Each current rollout is reused for five epochs, then the storage cursor clears
and new experience is collected. This is on-policy learning, not a replay-buffer method.

| Hyperparameter | Current setting |
|---|---:|
| Steps per environment per rollout | 32 |
| Epochs | 5 |
| Minibatches | 4 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| Learning rate | 0.0003, fixed |
| Value coefficient | 1.0 |
| Initial entropy coefficient | 0.01 |
| Entropy coefficient bounds | 0.001–0.05 |
| Gradient norm limit | 1.0 per network |
| Checkpoint interval | 50 iterations |

`desired_kl=None` means KL-based adaptive learning-rate control is disabled.
Checkpoints include model/training state and project compatibility metadata.
Resume options affect how that state is loaded; do not assume an arbitrary
checkpoint matches changed observations or a different policy variant.

## 15. Evaluation: measure the completed task

`eval_pipeline.py` loads a checkpoint, uses actor means by default, and runs
complete episodes without optimizer updates. It measures terminal observations
before autoreset so the next cube spawn does not corrupt the previous episode's result.

Strict lift success requires all of these simultaneously:

- Cube-specific contact on both fingers, using a 0.02 N contact threshold.
- Cube centre above 0.47 m.
- Cube/tool XY separation below 0.035 m and Z separation below 0.045 m.
- Actual gripper angle below 1.10 rad.
- Cube linear speed below 0.25 m/s.
- Sustained conditions for at least 0.5 seconds.

The contact-force threshold detects contact; it is not a calibrated grip-force
target or a guarantee that a real cube cannot be damaged.

At 25 Hz, the consecutive-step requirement is:

\[
n_{hold}=\lceil0.5/0.04\rceil=13.
\]

```python
required = math.ceil(0.5 / 0.04)
held_steps = 0
for conditions_met in [True] * 13:
    held_steps = held_steps + 1 if conditions_met else 0
assert held_steps >= required
```

Success rate is \(\sum_i I_i/N\), where \(I_i=1\) for a successful episode.
Episode reward is the undiscounted sum \(\sum_t r_t\); PPO return targets use
discounting and bootstrapping, so these are different quantities.

```python
successes = [True] * 44 + [False] * 6
success_rate = sum(successes) / len(successes)
episode_reward = sum([0.2, 0.4, 2.0])
assert success_rate == 0.88
assert episode_reward == 2.6
```

| Metric | Interpretation and limitation |
|---|---|
| Strict evaluation success | Completed held lifts; not a reward proxy |
| Training lift-success activation | Training log aggregation; not an independent evaluation |
| Peak lift | Maximum cube height above nominal 0.425 m centre; throwing can also raise it |
| Table slide | Accumulated XY path while cube is near the table; not just final displacement |
| Camera-visible fraction | Fraction of steps with valid cube detection |
| Wrist speed | Measured roll-joint velocity; approach peaks and held-grasp speeds differ |
| Supported tool angular speed | Full tool rotation while both fingers contact the cube |
| Finger normal force | Cube-specific normal contact; not shear or a damage threshold |
| TD error | One-step critic prediction discrepancy |
| Value loss | Squared target-fitting loss; reward scale affects its magnitude |
| Explained variance | How well value predictions explain target variation |

Your explained variance calculation is:

\[
EV=1-\frac{\operatorname{Var}(\hat R-V)}{\operatorname{Var}(\hat R)}.
\]

```python
targets = torch.tensor([1.0, 2.0, 3.0])
predictions = torch.tensor([0.9, 2.0, 3.1])
variance = targets.var(unbiased=False)
ev = 1 - (targets - predictions).var(unbiased=False) / variance.clamp_min(1e-8)
```

Near one means residual variation is small; zero resembles predicting a constant,
and negative values are possible. A constant prediction offset can still give
high explained variance, so this does not establish calibration or task success.
Your code reports zero when target variance is too small to interpret safely.

Recorded seed-101 evaluations of the same checkpoint, inspected on 10 October:

| Controller/run | Strict success |
|---|---:|
| Earlier baseline | 42/50 = 84% |
| Baseline with diagnostics | 46/50 = 92% |
| Full smooth experiment | 34/50 = 68% |
| Wrist-only experiment | 44/50 = 88% |

These are measured reports, not cross-seed mastery evidence. Identical seeds have
not produced identical baseline outcomes across runs. Repeat across seeds and
inspect movement quality before promoting a model or starting placement.

## 16. Python syntax you will see in these files

- `super().compute_returns(obs)`: call the inherited PPO method; custom diagnostics
  surround upstream GAE rather than replacing it.
- `@torch.no_grad()`: disable gradient recording for a function used as a diagnostic
  or parameter projection, not to train the actor.
- `.detach()`: disconnect a tensor from its gradient graph; rollout values are old
  predictions, not targets that should move while the loss is differentiated.
- `.clone()`: copy data; useful when simulation buffers will change next step.
- `torch.cat(..., dim=-1)`: join features along their last dimension.
- `.unsqueeze(-1)`: add a size-one dimension, such as changing `(N,)` to `(N,1)`.
- `torch.where(condition, x, y)`: choose elementwise between x and y.
- `cast(Any, ...)`: a type-checker hint; it does not transform a value at runtime.
- `os.environ[...]`: pass settings to modules initialized later by the launcher.

## 17. How to verify without starting another training run

Read existing reports; this command does not modify files or launch simulation:

```bash
cd /home/ai/jamshi/isaac_vision_rl
/home/ai/miniconda3/envs/env_isaaclab/bin/python -B -c "import json; from pathlib import Path; names=['lift_3898_baseline_diagnostics','lift_3898_smooth_seed101','lift_3898_wrist_only_seed101']; [print(n, json.loads((Path('outputs/evaluations')/(n+'.json')).read_text())['success_rate']) for n in names]"
```

For a new recorded wrist-only evaluation, use an unused output filename so you
preserve earlier evidence. This command runs simulation and writes new reports/video:

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/eval_pipeline.py \
  --task lift --curriculum lift --policy vision \
  --checkpoint logs/rsl_rl/so101_picklift_vision/2026-10-10_09-44-58/model_3898.pt \
  --episodes 50 --seed 202 --cube-jitter 0.01 \
  --wrist-only-control --headless --record-video \
  --output outputs/evaluations/lift_3898_wrist_only_seed202.json
```

Visible evaluation uses the same options but replaces `--headless` with
`--viz kit` and uses a distinct output filename. The evaluator uses one environment;
the 32-environment GUI preference applies to training, not this single-episode evaluator.
Run one evaluation at a time. For the baseline, omit `--wrist-only-control`.

The four distinctions to keep in your notebook are:

1. Reward scores environment behaviour; loss trains networks.
2. Value predicts future return; TD error measures a prediction discrepancy.
3. Advantage gives relative action credit; return estimates accumulated reward.
4. Contact and height alone are insufficient: verify a sustained held lift and
   controlled movement using completed episodes and recorded evidence.
