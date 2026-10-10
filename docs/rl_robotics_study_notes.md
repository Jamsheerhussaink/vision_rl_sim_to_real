# Reinforcement learning for robotics: theory, worked examples, and SO-101 troubleshooting

Prepared on 8 October 2026 for your SO-101 Isaac Lab project.

These notes explain RL through one continuing example: your SO-101 arm learning to pick up a cube. Each section begins with a short beginner lesson and a sentence you can copy into your notebook. Open the detailed explanation for the maths, source references, and project evidence. All 52 equations include a plain-language reading, calculation steps, and a symbol guide. Numerical examples are teaching examples unless identified as actual project settings or recorded results.

## How to use these notes

Read the short explanation first. Copy its **Write in your notebook** sentence. Then open the detailed notes for that topic. For a formula: read it in words, follow the calculation steps, name every symbol, and repeat the example yourself. For a failure: describe what happened, suggest why, test the explanation, and check whether task success improved.

Start with sections 4–7 because you are studying rewards now. Next read sections 8–9 for state, value, and Bellman ideas. Study TD and GAE before PPO. Read the advanced architecture sections after these basics feel familiar. You can be comfortable with maths and still be new to RL; the explanations assume no prior RL vocabulary.

## Contents

1. Sources, scope, and corrections
2. Notation and mathematical operators
3. The RL problem and the robot interaction loop
4. Rewards: meaning, timing, and design
5. Reward shaping and reward loopholes
6. Your implemented reward function
7. Return, discounting, and time
8. States, observations, Markov models, and resets
9. Policies, values, advantages, and Bellman equations
10. Bandits and exploration
11. Dynamic programming and planning
12. Monte Carlo, TD, SARSA, and Q-learning
13. Multi-step learning, eligibility traces, and GAE
14. Function approximation and numerical stability
15. Policy gradients and actor–critic learning
16. PPO and your training configuration
17. SAC, replay, and HER
18. Robotics control, coordinates, and contact physics
19. Simulation, vectorization, and training budgets
20. Vision, privileged information, and perception
21. Curriculum, domain randomization, and sim-to-real
22. Imitation learning, behavioural cloning, and DAgger
23. ACT, diffusion policies, and vision-language-action models
24. Evaluation, diagnostics, and deployment
25. What your saved project evidence establishes
26. Training hurdles: what, why, how, and verification
27. Worked exercises with answers
28. Study sequence and experiment notebook
29. Source map and further reading

## 1. Sources, scope, and corrections

### Start here: What these notes are for

We will use one example throughout: **your SO-101 arm picking up a cube**. Each new idea answers a question about that example. This lets you learn the meaning before learning the formula.

Your Sutton–Barto book explains how RL works. Your second PDF explains how those ideas are used in robotics. Your project shows what happens when we try them in practice.

Read the short explanation at the start of a section first. Open the detailed notes when you are ready for the equations, extra examples, and source references. You do not need to memorise all the algorithms at once.

**Words to know:** a *policy* is the robot's action-selection rule; a *training run* is a period of learning; an *evaluation* is a test of the learned rule.

**Write in your notebook:** RL theory explains how learning works. A robotics experiment checks whether that learning produces the intended physical behaviour.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

**Sutton and Barto:** `/home/ai/Downloads/SuttonBartoRL.pdf` contains 455 PDF pages. Its title page identifies a second-edition draft from 2016. Chapters 1–13 supply the main learning foundations. Chapters 14–17 discuss psychology, neuroscience, applications, and open questions. Some draft sections contain unfinished text; do not mistake a placeholder for a finished result.

**Robotics excerpt:** `/home/ai/Downloads/reinforcement_learning_robotics_index_to_page70.pdf` contains 74 PDF pages, including contents and duplicate screenshots, reaching printed page 70. It contains image pages rather than an extractable text layer; all 74 pages were OCR-processed and the topic text reviewed. OCR can lose mathematical symbols and labels in figures. Use the corrected equations below rather than copying OCR strings. In these notes, references to its pages mean the **printed page number**, unless explicitly called a PDF page.

The excerpt covers foundations, PPO/SAC, simulation, randomization, imitation, ACT/diffusion, VLAs, locomotion, industrial control, and deployment. Its broad industry claims and illustrative benchmark graphics should not be used as measured evidence for your robot. We use its useful concepts and verify important technical details against original papers and your implementation.

### Corrections to remember

| Simplification in the robotics excerpt | Precise interpretation |
| --- | --- |
| State is everything the agent can observe | State is sufficient information for the transition model; observation is what the agent receives. They can differ. |
| Every problem is an MDP | An appropriate full-state model can be Markov. Limited robot sensors generally produce partial observability. |
| Policy gradient uses log-probability times advantage | The gradient uses the **gradient of log-probability** times advantage. A scalar log-probability is not itself a parameter update direction. |
| PPO data is used only once | PPO collects a recent rollout, then reuses that rollout over multiple optimization epochs. It generally does not train from an unrestricted archive of old rollouts. |
| PPO clipping forces the ratio inside its interval | The clipped objective removes particular incentives for excessive changes. It does not force every ratio or the entire policy to obey a hard bound. |
| A small TD error means success | A critic can accurately predict the return of a policy that never lifts anything. |
| Entropy guarantees robust behaviour | Entropy encourages distributional spread. Robust task performance needs evaluation. Your actor's state-independent noise scale is not a calibrated uncertainty detector. |
| Reward penalties ensure safety | Penalties influence an objective; they do not impose physical limits or certify safety. |
| Randomization guarantees transfer if reality is within the bounds | Covering plausible parameters helps, but representation, optimization, calibration, and coupled effects still matter. |
| ACT executes only open-loop chunks | Action chunks can be combined with repeated replanning and temporal ensembling. Chunk length and execution horizon are separate design choices. |
| ACT and diffusion are automatically RL algorithms | Their standard demonstration-training versions are imitation learning. RL fine-tuning is a separate addition. |
| A simulator is always CPU-only or always GPU-only | Backend and version matter. MuJoCo also has accelerated implementations such as MJX; your task explicitly selects PhysX. |

The clipping and rollout corrections follow the [PPO paper](https://arxiv.org/abs/1707.06347). ACT's action-sequence approach is described in the [ACT paper](https://arxiv.org/abs/2304.13705). Backend details should be checked in the [MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) and [Genesis documentation](https://genesis-world.readthedocs.io/en/latest/).

</details>

## 2. Notation and mathematical operators

### Start here: How to read the symbols without getting lost

Think of an equation as a sentence written in a shorter language. Start by naming the quantities. Then explain what the equation does to them.

For example, `t` means “the current step.” `t+1` means “the next step.” It does not mean one second later. In your project, one action step lasts 0.04 seconds.

`R` means reward: a score for one transition. `G` means return: rewards combined over the future. `V` means a prediction of that future score. `γ`, pronounced “gamma,” tells us how strongly to count later rewards.

A sum means “add these pieces.” An expectation means “average over possible outcomes, using their probabilities.” A gradient tells us how changing the network's weights changes a quantity.

**Example:** if a lift succeeds in some attempts and fails in others, its expected score combines both outcomes. One successful attempt does not describe the average.

**Write in your notebook:** Before using a formula, explain what each symbol represents and what each operation does.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

Uppercase letters such as `S_t`, `A_t`, and `R_t` usually describe random variables. Lowercase `s`, `a`, and `r` describe particular values. The subscript `t` means a decision time index, not necessarily seconds. The next reward is written `R_(t+1)` because it follows action `A_t`.

| Symbol | How to read it | Meaning |
| --- | --- | --- |
| `S_t`, `O_t`, `A_t` | state, observation, action at time t | Full model state, received information, and chosen command |
| `R_(t+1)` | reward at t plus one | Reward following the current action |
| `G_t` | return from t | Accumulated future reward |
| `π`, `π_θ` | pi, pi parameterized by theta | Policy; theta is its trainable parameter vector |
| `V^π`, `Q^π` | V under pi, Q under pi | State value and action value for a specified policy |
| `γ`, `λ` | gamma, lambda | Discount factor and trace/advantage weighting parameter |
| `α`, `η` | alpha, eta | Learning step sizes where defined; SAC also uses alpha for temperature |
| `δ_t` | delta at t | Temporal-difference residual |
| `Â_t`, `Ŷ_t` | estimated advantage, estimated target | A hat indicates an estimate |
| `E[X]` | expectation of X | Probability-weighted average, not necessarily one observed value |
| P(X \| Y) | probability of X given Y | The vertical bar means conditioning on information |
| `Σ` | sum | Add terms over the stated index |
| `∇_θ` | gradient with respect to theta | Vector of partial derivatives with respect to parameters |
| `∂` | partial derivative | Change with respect to one variable while holding others fixed |
| `←` | is updated to | Replace an old stored quantity with a new value |
| `∝` | proportional to | Equal up to a multiplicative factor |
| `≈` | approximately equal | An approximation rather than exact equality |
| `∈` | belongs to | Membership in a set |
| `~` | is sampled from | Random sampling from a distribution |
| `‖x‖₂` | Euclidean norm of x | Length of a vector |
| `‖x‖₂²` | squared Euclidean norm | Sum of squared vector components |
| `max`, `min` | maximum, minimum | Largest or smallest numerical value |
| `argmax` | argument attaining the maximum | The action or input giving the largest value, not that value itself |
| `1[condition]` | indicator of condition | One when true, zero when false |
| `clip(x,l,u)` | clip x between l and u | Restrict x to a lower and upper bound |
| `exp(x)`, `log(x)` | exponential, natural logarithm | Exponential base e and its inverse |

A superscript `π` names the policy; it is **not an exponent**. A superscript `2` in a squared error is an exponent. The same letter can have different meanings in different subjects: robot joint vector `q` is different from action-value function `Q`; physical motor torque `τ` is different from a trajectory sometimes also called tau.

</details>

## 3. The RL problem and the robot interaction loop

### Start here: The robot learns through repeated attempts

Imagine the arm starting with its gripper open. It receives information about the scene, chooses a command, and the simulator moves the robot. The simulator then provides a reward and new information.

At first, the commands may be unhelpful. Training changes the policy so that commands associated with better future outcomes become more likely.

The **agent** is the learning decision-maker. The **environment** is what responds: motors, links, cube, table, sensors, and simulated physics. The **observation** is the information received. The **action** is the command chosen.

**Example:** the agent requests gripper closure. The environment decides what actually happens through physics. The cube may become held, get pushed, or remain untouched.

**Write in your notebook:** The policy chooses a command; the environment determines its physical consequence. Learning uses experience from those consequences.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

**Notebook definition:** Reinforcement learning studies how an agent learns behaviour from interaction to improve an objective expressed through rewards over time.

The learner needs to decide, observe consequences, and use experience. It usually has no label saying which action was correct at every step. Supervised learning instead learns from example targets. Imitation learning uses expert actions as supervised targets; it can then be combined with RL.

The agent is the decision-making system. The robot's physical links, motors, simulator, and objects are part of its environment at the policy boundary. The whole robot is not automatically the agent: your actor chooses joint references, while actuator control and physics determine actual motion.

The loop is:

```text
Receive observation → choose action → process command → simulate physics
→ compute reward and episode ending → receive next observation → learn
```

For SO-101, the actor chooses six action components: five arm joints and one gripper. The arm actions become position targets; they are not direct torque commands. Motor dynamics and contact forces make the cube move.

**Credit assignment** means determining which earlier choices contributed to a later outcome. Opening the jaws, aligning with the cube, closing, and raising the arm all matter to a final lift bonus. The bonus alone does not label each action.

**Exploration** tries behaviours to gather information. **Exploitation** uses current knowledge. A random command sequence rarely produces a complete grasp, which is why task design, easier initial conditions, and demonstrations can matter as much as algorithm selection.

</details>

## 4. Rewards: meaning, timing, and design

### Start here: Reward is the score after one action

A reward is one number returned after an action changes the environment. You choose the reward rule to encourage the task you want.

Suppose the robot moves closer to the cube. You may give a small positive score for progress. Suppose it pushes the cube away. You may give a negative dragging score. Suppose it lifts and holds the cube. You may give a larger completion score.

These scores are **teaching examples**, not the exact numbers from your configuration. Section 6 explains your actual settings.

The policy does not receive a sentence saying “that was the correct grasp.” It receives the score. If the score rewards hovering, the policy may learn hovering.

**Words to know:** *sparse* reward appears rarely; *dense* reward gives more frequent feedback; a *penalty* is a negative contribution.

**Write in your notebook:** Reward tells the learner how a transition was scored. The score must encourage the behaviour we actually want.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 4.1 What reward is

**Notebook definition:** A reward is a scalar signal returned by the environment after an interaction. The learning objective usually combines rewards across time rather than selecting only the largest immediate reward.

Scalar means one number. Your task can calculate many components, but the learner receives their combined reward. Negative rewards are often called costs or penalties. Zero means a numerical value, not necessarily failure or missing feedback.

A common reward equation is:

**Read equation E1 in words:** Use the reward rule to score what happened after the action.

```text
R_(t+1) = r(S_t, A_t, S_(t+1))                         [E1]
```

**Calculate it step by step:** Identify the state before the action, the action itself, and the state afterward. Feed those into the reward rule. If the rule awards ten for a completed lift, this transition gets ten only when it meets that rule.


- `R_(t+1)` is the observed reward after the current action.
- `r` is the reward function; lowercase r here names a function.
- `S_t` is the state before the transition.
- `A_t` is the action used for that transition.
- `S_(t+1)` is the resulting state.
- The parentheses mean that these quantities are inputs to the function; the equals sign says the function produces the reward.
- This formulation assumes deterministic reward given the listed inputs. A stochastic model can instead specify probabilities over next states and rewards.

**Example:** The gripper closes beside the cube and pushes it away. The action looked like a grasp command, but the transition can receive a negative dragging contribution. Reward describes the outcome according to the designed objective, not the robot's intention.

### 4.2 Reward is not a label or probability

A reward of 4 does not mean 4% success or an action label of class 4. A negative reward does not prove an action was globally wrong: lifting may require movement penalties before completion. Reward is also not the critic's prediction. The environment supplies reward; the critic estimates future accumulated reward.

### 4.3 Sparse, dense, and terminal rewards

**Sparse reward:** positive feedback occurs on relatively few transitions, such as only on a completed lift. Advantage: direct relationship to the goal. Difficulty: early random experience may contain no positive successes.

**Dense reward:** intermediate progress receives feedback, such as improved reach distance or increased cube elevation. Advantage: more learning signal. Difficulty: intermediate behaviour can become a substitute for success.

**Terminal reward:** a reward on the transition ending an episode. This describes timing, not density. A task can have dense shaping plus a terminal completion bonus.

**Intrinsic reward:** an added signal related to novelty or information acquisition. **Extrinsic reward:** a task signal such as lifting. Novelty can aid exploration but does not establish task completion.

**Delayed reward:** feedback occurs well after actions contributing to it. Reward delay motivates value functions and multi-step credit assignment.

### 4.4 Combining reward terms

A general weighted reward is:

**Read equation E2 in words:** Multiply each raw score by its weight, then add the contributions.

```text
R_(t+1) = Σ_(i=1)^m w_i f_i(S_t, A_t, S_(t+1))        [E2]
```

**Calculate it step by step:** For grasp score 0.5 with weight four, multiply to get two. For dragging score 0.1 with weight negative five, multiply to get negative 0.5. Add them to get 1.5 before framework time scaling.


- `m` is the number of components.
- `i` identifies one component; the sum adds components 1 through m.
- `w_i` is its numerical weight.
- `f_i` is its raw feature or score, evaluated on the transition.
- `S_t`, `A_t`, and `S_(t+1)` identify the transition as in E1.
- Multiplication by `w_i` controls the sign and influence of that component.
- `R_(t+1)` is the final scalar reward.

If the raw grasp score is 0.5 with weight 4, its unscaled weighted contribution is 2. If raw dragging cost is 0.1 with weight -5, its contribution is -0.5. Their sum is 1.5 before any framework time scaling or other terms.

### 4.5 Designing reward: what, why, how

**What:** specify the intended outcome first: a cube physically held above a height for a duration. Separately specify failures and practical constraints.

**Why:** a convenient measurable score can differ from your intention. End-effector proximity is easy to compute but does not mean a stable grasp.

**How:** add intermediate signals only after checking their incentives. Test known successful, stationary, dragging, and dropping trajectories. Record each component separately so you can see which term dominates.

Do not tune a weight just because its numerical value looks small. A frequently active penalty can outweigh a large bonus the agent never discovers. Compare actual per-step contributions and entire discounted returns.

### 4.6 Reward scaling and shifting

Multiplying all rewards by the same positive constant preserves the ranking of returns in an ordinary reward-only objective, but changes critic target scales, gradients, and the balance with an entropy bonus or fixed optimization settings.

Adding a constant every step can change a variable-duration task: a positive constant may favour delaying termination; a negative constant may favour finishing earlier or even ending through failure. It is not a universally harmless transformation.

Reward clipping can reduce extreme magnitudes but change which behaviours are preferred. Diagnose the source of spikes before hiding them. A physically invalid state must not become ordinary valid data because its reward was clipped.

</details>

## 5. Reward shaping and reward loopholes

### Start here: Helpful rewards can accidentally teach the wrong behaviour

Imagine paying the robot one point every time it stays close to the cube. Staying close for a long time may earn more points than quickly finishing the lift. The robot can learn to hover because that earns the score we specified.

This is a **reward loophole**: the score improves while the intended task does not. *Reward shaping* means adding intermediate scores to make learning easier. Those added scores can create loopholes too.

A progress reward gives credit for moving closer. Your project records the closest distance reached and rewards a new improvement. Returning to an already reached distance gives no new reach credit.

This helps with repeated approach rewards, but it does not teach the whole grasp by itself. Closure and lifting still need useful feedback and experience.

**Write in your notebook:** Ask whether the robot can repeatedly earn a reward while avoiding task completion.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 5.1 Hovering near the cube

Consider an illustrative reward of +1 every step for proximity. At discount 0.99, hovering for 150 steps produces:

**Read equation E3 in words:** Add one point per hovering step, making later points count less.

```text
G_hover = Σ_(k=0)^149 0.99^k
        = (1 - 0.99^150) / (1 - 0.99)
        ≈ 77.85                                      [E3]
```

**Calculate it step by step:** The first point counts as one. The second counts as 0.99. The third counts as 0.9801 because 0.99 is multiplied by itself. Continue for 150 points; their sum is about 77.85.


- `G_hover` is the discounted return of the hovering behaviour.
- `k` indexes the reward's delay from the start, from 0 to 149.
- `0.99` is the discount factor; raising it to `k` reduces later contributions.
- Each term's reward is one, so it is omitted as a multiplier.
- The fraction is the finite geometric-series identity.
- `150` is the number of reward-bearing steps.
- `≈` means the decimal result is rounded.

A bonus of 10 received after 50 actions has discount exponent 49 under this indexing, giving roughly 6.11 from the bonus alone. The agent may rationally prefer hovering under that reward specification. This example is not your current reward scale.

### 5.2 Progress shaping

Simple distance progress can be written:

**Read equation E4 in words:** Progress is the old distance minus the new distance.

```text
f_progress,t = d_t - d_(t+1)                           [E4]
```

**Calculate it step by step:** Start at 0.12 metres and finish at 0.09 metres. Subtract 0.09 from 0.12 to get positive 0.03 metres. The positive sign means closer. Staying at the same distance gives zero.


- `f_progress,t` is the raw progress score for the transition at t.
- `d_t` is tool-to-cube distance before the action.
- `d_(t+1)` is the distance afterward.
- Subtraction makes approaching positive, moving away negative, and hovering zero.
- Distances must use the same units and coordinate frame.

Reducing distance from 0.12 m to 0.09 m gives 0.03 m of progress. If you keep only positive progress and forget to penalize regression, moving away and back can repeatedly collect credit. Your implementation instead stores the **best distance reached so far** and rewards only a new improvement. This removes that particular repetition incentive, but does not guarantee eventual grasping.

### 5.3 Potential-based shaping

A special shaping construction is:

**Read equation E5 in words:** Take discounted next potential, then subtract current potential.

```text
F(s,a,s') = γ Φ(s') - Φ(s)                            [E5]
```

**Calculate it step by step:** Potential means a chosen progress score. Multiply next score negative 0.09 by discount 0.99 to get negative 0.0891. Subtract current score negative 0.12. Subtracting a negative adds 0.12, giving 0.0309.


- `F` is the extra shaping reward for a transition.
- `s` is the current state, `a` the action, and `s'` the next state; the prime means next, not a derivative.
- `Φ` is a chosen scalar potential function assigning a progress score to a state.
- `γ` is the same discount factor as the task objective.
- `γ Φ(s')` is discounted next potential; subtracting `Φ(s)` gives the shaping contribution.

For potential -0.12 now, -0.09 next, and discount 0.99, the shaping contribution is `0.99 × (-0.09) - (-0.12) = 0.0309`.

Under the theorem's assumptions, including appropriate terminal handling, this form preserves optimal policies of the original MDP. In an episodic setting, use a consistent terminal potential, commonly zero, to avoid an unwanted terminal residual. Your current best-distance bonus is an engineering shaping rule; it is not automatically this theorem's construction. See [Ng, Harada, and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

### 5.4 Penalties are incentives, not hard constraints

A dragging penalty discourages dragging only relative to other rewards. A sufficiently profitable loophole may still justify it. A joint-limit penalty does not prevent commanding a dangerous reference. Physical limits and controller checks act at a different layer.

For learning, the useful question is: does the reward make genuine success attractive while still providing discoverable feedback? For physical validity, the useful question is: can the action interface and simulator execute those commands correctly?

</details>

## 6. Your implemented reward function

### Start here: What your reward numbers really mean

Your reward is built from several parts: reaching, grasping, lifting, completion, and penalties. Add their contributions to get the single score sent to the learner.

A configured **weight** tells us how strongly a part counts. The simulator's reward manager also multiplies by the action-step duration, which is 0.04 seconds.

**Example:** the completion part has weight 6000. On its successful step, the raw completion signal is one. Multiplying 6000 by 0.04 gives **240 reward from that part**. Other parts can still add or subtract on the same step.

A successful lift also requires holding, closeness, suitable gripper angle, and slow cube motion. Briefly bouncing above the height threshold is insufficient.

**Write in your notebook:** A raw score, its configured weight, and its final reward contribution are three different quantities.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

Sources: `env_cfg.py`, `mdp/rewards.py`, and the installed Isaac Lab reward manager. The current manager multiplies each active term by both its weight and policy-step duration.

**Read equation E6 in words:** Add the weighted scores, then multiply by the policy-step duration.

```text
R_(t+1) = Δt Σ_(i=1)^m w_i f_i                       [E6]
```

**Calculate it step by step:** If the weighted contributions sum to 1.5, multiply by 0.04 seconds to get 0.06 step reward. The duration is a rule of your reward manager; check it before comparing raw weights.


- `R_(t+1)` is the combined step reward supplied to training.
- `Δt` is one policy step's duration in seconds, currently 0.04 s.
- `m` counts active reward terms; `i` selects one term.
- `w_i` is the configured weight, positive for rewards and negative for penalties.
- `f_i` is that term's raw output on the current step.
- The sum adds all weighted terms; multiplying by `Δt` scales the result.

This explains why a configured weight is not the same as the reward appearing in the rollout.

| Current lift term | Weight | Meaning |
| --- | ---: | --- |
| `reach_progress` | 3 | New best distance improvement toward the cube |
| `premature_close` | -3 | Closing before the cube is centered |
| `grasp_cube` | 4 | Geometrically centered cube and closed fingers |
| `lift_cube` | 8 | Gated elevation and upward-motion feedback |
| `lift_completed` | 6000 | Completion signal at sustained lift |
| `drag_penalty` | -5 | Cube horizontal speed near the tabletop |
| `displacement_penalty` | -12 | Excess horizontal displacement from reset position |
| `table_collision` | -0.5 | Tool-frame penetration proxy |
| `action_rate` | -0.02 | Change between successive actions |
| `joint_vel` | -0.01 | Bounded squared joint velocities |
| `joint_limits` | -0.05 | Joint limit violations |

The old repeated `reach_cube` proximity term is disabled in the active configuration. A function remaining in `rewards.py` does not mean it is an active reward. Similarly, older overhead-approach functions should not be assumed to describe current training.

### 6.1 Actual completion bonus

**Read equation E7 in words:** A successful completion signal activates its weighted, time-scaled reward.

```text
R_completion = 0.04 × 6000 × 1 = 240                  [E7]
```

**Calculate it step by step:** The signal is one. Multiply one by weight 6000. Multiply the result by 0.04. The completion part contributes 240. Other parts can change the total reward on that same step.


- `R_completion` is the completion term's contribution on its activating step.
- `0.04` is the policy-step duration in seconds.
- `6000` is the configured completion weight.
- `1` is the true completion indicator, converted to a numerical score.
- Multiplication is the manager's rule; 240 is this term alone, before other simultaneous rewards and penalties.

A large completion bonus can improve the intended objective, but cannot teach from successful examples that never occur.

### 6.2 Actual best-distance reward

Your raw reach-progress score has the form:

**Read equation E8 in words:** Reward a new closest approach and update the closest-distance record.

```text
f_reach,t = max(b_t - d_t, 0) / 0.05
b_(t+1)  = min(b_t, d_t)                              [E8]
```

**Calculate it step by step:** Suppose the old best distance is 0.10 metres and current distance is 0.09 metres. Improvement is 0.01. Divide by 0.05 to get raw score 0.2. Store 0.09 as the new best. Returning later to 0.10 earns no new improvement.


- `f_reach,t` is the raw reach score at the current reward evaluation.
- `b_t` is the best distance recorded before this evaluation.
- `d_t` is the measured current tool-to-cube distance.
- `max(...,0)` removes negative credit for regression; `min` updates the record to the smallest distance.
- `0.05` is a 5 cm normalization distance in metres.
- `b_(t+1)` is the updated memory for the next evaluation.
- On the first evaluation after reset, the implementation initializes the reference to current distance, avoiding an infinite-distance bonus.

Improving the record by 1 cm gives raw score 0.2. With weight 3 and duration 0.04, that term contributes 0.024. It is zero when the robot hovers or returns only to a previously achieved distance.

Because this reward has memory, a full Markov description includes the best-distance record. Your critic observes it. The actor does not receive every internal task counter, so its input should not be described as a complete Markov state.

### 6.3 Actual lift-success detector

The cube must satisfy all of these conditions continuously:

- Center height above 0.47 m.
- Horizontal tool distance below 0.035 m.
- Absolute vertical tool distance below 0.045 m.
- Gripper angle below 1.10 rad.
- Cube linear speed below 0.25 m/s.
- Hold time at least 0.2 s, which is five 0.04 s policy steps.

The tabletop is at 0.40 m and the 5 cm cube initially has center height 0.425 m. Center height 0.47 m corresponds to 4.5 cm of center elevation above its resting position. It is not 47 cm of lifting.

The detector is a **sustained geometric proxy**. It does not measure both finger contacts directly. Evaluation counts episodes independently of the shaped reward, but still uses this same success definition; video or contact inspection gives another kind of evidence.

</details>

## 7. Return, discounting, and time

### Start here: Return adds the rewards that come later

The robot may need to do something useful now that earns no immediate reward. Opening its fingers can make a later grasp possible. We therefore judge actions using future rewards too.

**Return** combines the rewards from a starting step onward. **Discounting** gives less weight to rewards that arrive later. Gamma controls this weighting.

**Example:** rewards arrive as zero, zero, then ten. With gamma 0.9, the first zero is unchanged, the second is multiplied by 0.9, and the ten is multiplied by 0.9 twice. The return is 8.1.

The robot is learning to improve expected return. It is not simply choosing the largest immediate reward.

A **rollout** is a batch of collected experience for a training update. An **episode** is one whole task attempt. They can end at different times.

**Write in your notebook:** Reward scores one transition. Return combines future rewards. Gamma controls how much later rewards count.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 7.1 Finite-episode return

**Read equation E9 in words:** Add the remaining rewards, discounting each according to its delay.

```text
G_t = Σ_(k=0)^(T-t-1) γ^k R_(t+k+1)                  [E9]
```

**Calculate it step by step:** For rewards zero, zero, ten and gamma 0.9: first count zero; next count 0.9 times zero; last count 0.9 times 0.9 times ten, which is 8.1. Add the three contributions.


- `G_t` is return starting before action t.
- `T` is the terminal state index; the last reward is `R_T`.
- `t` identifies where we start evaluating.
- `k` counts how many steps into the future a reward occurs, beginning at zero.
- `γ` is the discount factor; `γ^k` is its kth power.
- `R_(t+k+1)` is the reward after the corresponding future action.
- The upper sum index ensures rewards stop at the episode boundary.

For rewards 0, 0, 10 and discount 0.9, the return is 8.1. For rewards 2, 2, 2 it is 5.42. The same comparison changes at discount 0.5: the returns become 2.5 and 3.5. Discounting changes preferences over timing.

### 7.2 Why the Bellman recursion works

**Read equation E10 in words:** Current return is reward now plus discounted remaining return.

```text
G_t = R_(t+1) + γ G_(t+1)                            [E10]
```

**Calculate it step by step:** If reward now is one, return from the next step is ten, and gamma is 0.9, first multiply ten by 0.9 to get nine. Add the immediate one to get current return ten.


- `G_t` is current return; `G_(t+1)` is the return beginning one step later.
- `R_(t+1)` is the first reward.
- `γ` discounts the whole remainder by one step.
- Addition separates immediate feedback from future feedback.

If the first reward is 1, next-step return is 10, and discount is 0.9, current return is 10. This recursion leads directly to Bellman value equations and TD targets.

### 7.3 Discount in physical time

Your policy runs at 25 Hz. One second is 25 decisions. With gamma 0.99, rewards one second later have discount multiplier approximately 0.778; four seconds later approximately 0.366; six seconds later approximately 0.221.

A rough discount horizon is:

**Read equation E11 in words:** Estimate how many decisions the discount strongly weights, then convert to seconds.

```text
H_steps ≈ 1 / (1 - γ)
H_seconds ≈ Δt / (1 - γ)                             [E11]
```

**Calculate it step by step:** Subtract 0.99 from one to get 0.01. Divide one by 0.01 to get about 100 steps. Multiply 100 by 0.04 seconds to get about four seconds. This is a rough horizon, not a cutoff.


- `H_steps` is a rough effective horizon measured in decision steps.
- `H_seconds` converts it to physical time.
- `γ` is a per-decision discount strictly below one.
- `Δt` is policy-step duration in seconds.
- The reciprocal comes from the total weight of an infinite unit-reward sequence.
- `≈` emphasizes that this is a heuristic, not a hard cutoff: later rewards still contribute.

For gamma 0.99 and duration 0.04 s, these estimates are 100 decisions and 4 seconds.

If the decision interval changes while preserving the same continuous-time discount rate:

**Read equation E12 in words:** Adjust per-step discount when the duration of a step changes.

```text
γ_new = γ_old^(Δt_new / Δt_old)                       [E12]
```

**Calculate it step by step:** Divide new duration 0.02 by old duration 0.04 to get one half. Raising old gamma 0.99 to the power one half means taking its square root, about 0.99499. This preserves the assumed physical-time discount rate.


- `γ_old` is the previous discount per decision.
- `γ_new` is the corresponding discount for the new decision interval.
- `Δt_old` and `Δt_new` are old and new intervals in the same time unit.
- Division forms their ratio; the superscript means exponentiation.
- This assumes an exponential discount convention in physical time.

Moving from 0.04 s decisions to 0.02 s decisions gives a new gamma around 0.99499. Keeping 0.99 would shorten the physical discount horizon.

### 7.4 Episode, rollout, and timeout

An **episode** is a task attempt. A **rollout** is a fixed collection of transitions used for an update and can contain portions of multiple episodes. Your rollout length is 32 steps per environment, or 1.28 s; the episode deadline is 6 s, or 150 decisions.

Ending a rollout does not mean the task ended. Bootstrap from its final nonterminal state. A genuine task terminal state has no further task return. An artificial collection cutoff can require bootstrapping.

Your configuration treats the 6 s deadline as part of the finite-horizon task and supplies remaining time as an observation. The installed wrapper does not apply timeout continuation bootstrapping in this finite-horizon mode. Do not universally treat every timeout in every project the same way.

</details>

## 8. States, observations, Markov models, and resets

### Start here: What information does the robot need?

Imagine two identical pictures of a cube. In one situation, the cube is stationary. In the other, it is sliding quickly. The same picture may require different actions because velocity matters.

The **state** contains enough information to predict what can happen next after an action. The **observation** is the information the policy receives. These can differ.

The **Markov property** means the current state already contains the useful predictive information from the past. If important information is missing, observation history or an estimator may help.

Your action filter remembers its previous target, so that memory affects the next command. Your policy observes filtered targets to account for it.

A reset starts a new attempt. Even a fixed-position cube must be returned to its pose with its velocity cleared.

**Write in your notebook:** Good observations make useful decisions possible. Correct resets make the starting conditions match the experiment.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 8.1 State and observation

**State:** information sufficient to specify the system's next-transition distribution given an action. In a robot model this can include joint positions and velocities, cube pose and velocity, controller memory, task phase, time, and parameters affecting dynamics.

**Observation:** information exposed to the policy. A camera frame, encoder readings, and previous command form an observation. Hidden object velocity, occlusion, friction, or actuator delay can make it incomplete.

Two pictures showing the cube in the same position can require different actions if the cube is moving left in one case and right in the other. An image alone may not identify velocity. A history, recurrent network, or state estimator can help.

The Markov property is:

**Read equation E13 in words:** Once the state contains enough information, extra history does not improve the next-state prediction.

```text
P(S_(t+1) | S_t,A_t,S_(t-1),A_(t-1),...)
    = P(S_(t+1) | S_t,A_t)                            [E13]
```

**Calculate it step by step:** Compare predictions for two histories that end in the same full state and action. Under the Markov assumption, their next-state distributions match. If hidden velocity makes them differ, your supposed state was missing useful information.


- `P` describes a conditional next-state distribution.
- `S_(t+1)` is the next state.
- `S_t` and `A_t` are the current state and action.
- Earlier indexed states and actions form the history; the dots mean still earlier history.
- The vertical bar means given; equality says history adds no predictive information once current state and action are known.
- This is a property of the representation and dynamics, not an assertion that the robot has no memory.

### 8.2 MDP formulation

An MDP is commonly specified by state space, action space, transition model, and reward model; a discounted learning problem also specifies discount and initial-state distribution. A finite-horizon problem includes a deadline. A partially observable problem adds an observation model or equivalent description of available information.

The joint transition-and-reward model in Sutton–Barto notation is:

**Read equation E14 in words:** The transition model tells us the probability of a next-state and reward outcome.

```text
p(s',r | s,a) = P(S_(t+1)=s', R_(t+1)=r | S_t=s,A_t=a) [E14]
```

**Calculate it step by step:** Choose one current state and action. Ask for one possible next state and reward, such as a held cube with reward ten. The model gives the probability of that outcome. Other outcomes need their own probabilities.


- Lowercase `p` is the model's probability function.
- `s` and `a` are particular current state/action values.
- `s'` is a possible next state; the prime means next state.
- `r` is a possible numerical reward, not the reward function in E1.
- Uppercase variables describe random outcomes; equals signs inside the probability select particular values.
- The conditional bar says the model is evaluated given the current state and action.

In your training, the physics simulator generates transitions, and reward functions score them. PPO does not need a table of all probabilities. Having a simulator does not make PPO model-based: the policy learner uses sampled transitions rather than planning through an explicit learned transition model.

### 8.3 Hidden controller and task memory

A low-pass action filter makes the next target depend on its previous target. Include that filter memory in the full state. Your actor receives filtered targets for this reason. Remaining time matters because a good action with 5 s available can differ from one with 0.1 s available.

Best-distance records, lift hold counters, and prior-lift flags also influence rewards or termination. Your critic receives task memory. This helps value prediction, but it does not guarantee that the actor's limited observation is Markov.

### 8.4 Reset is part of the experiment

The initial-state distribution says where attempts begin. If the cube is not returned to its starting pose and velocity, a supposed fixed-start experiment actually trains on drifting conditions.

Reset the robot, cube pose, cube velocity, controller filter memory, reward memory, success counters, and episode time for the selected environments. A partial reset must leave other environments alone. In your project, zero randomization ranges still require a cube reset event.

</details>

## 9. Policies, values, advantages, and Bellman equations

### Start here: Policy, value, and advantage answer different questions

The **policy** answers: “What action should I choose?” The **value** answers: “If I continue with this policy, how much future score should I expect from here?” The **action value** answers the same question after fixing one first action.

**Advantage** asks whether a particular action is better or worse than the policy's average choice from that situation.

**Example:** expected future score from a situation is 20. An action is expected to give 24. Its advantage is positive four. Another action expected to give 18 has advantage negative two.

The **Bellman equation** breaks future value into two pieces: reward now, and discounted value after the next step. If several outcomes are possible, average them using their probabilities.

**Write in your notebook:** Value predicts future score. Advantage compares an action with the usual choice. Bellman reasoning connects now with next.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 9.1 Policy

A deterministic policy picks one action for a given input. A stochastic policy specifies a distribution and samples from it. In fully observed textbook equations, the input is state. In your actor, it is an observation.

**Read equation E15 in words:** Sample an action from the distribution produced for the current observation.

```text
A_t ~ π_θ(· | O_t)                                   [E15]
```

**Calculate it step by step:** Give the observation to the actor. It produces action-distribution parameters. Draw an action from that distribution. Repeating with the same observation can give a different sample because training includes randomness.


- `A_t` is the sampled action.
- `~` means sampled from, not approximately equal.
- `π_θ` is the policy parameterized by neural weights `θ`.
- `O_t` is the current observation.
- The dot is a placeholder for possible actions.
- The conditional bar means the action distribution depends on the observation.

Training samples actions for exploration. Deterministic evaluation commonly uses the actor mean. These can produce different outcomes, especially when the mean lies in an ineffective region.

### 9.2 State value

**Read equation E16 in words:** State value is the average future return when following this policy.

```text
V^π(s) = E_π[G_t | S_t=s]                             [E16]
```

**Calculate it step by step:** Imagine two equally likely future returns, ten and zero. Their probability-weighted average is five. That is a value prediction in reward units, not a 5% success probability.


- `V^π(s)` is the expected return starting in state s while following policy pi.
- Superscript `π` identifies the policy; it is not a power.
- `E_π` averages over trajectories generated under that policy.
- `G_t` is future return from E9.
- `S_t=s` conditions on starting in the specified state.

**Interpretation:** How much discounted reward should I expect from here if I continue using this policy? The value is expressed in reward units. It is not a probability unless a specially chosen reward and discount make it one.

### 9.3 Action value and advantage

**Read equation E17 in words:** Action value fixes the first action; advantage subtracts the usual state value.

```text
Q^π(s,a) = E_π[G_t | S_t=s,A_t=a]
A^π(s,a) = Q^π(s,a) - V^π(s)                         [E17]
```

**Calculate it step by step:** If the policy average is twenty and taking a particular first action is expected to give twenty-four, subtract twenty to get advantage positive four. Expected return eighteen gives advantage negative two.


- `Q^π(s,a)` is return after taking a particular first action a and following pi afterward.
- `E_π`, `G_t`, and conditioning have the same meanings as E16.
- `A_t=a` fixes the initial action.
- `A^π(s,a)` is advantage, different from the sampled action symbol `A_t`.
- `V^π(s)` is the ordinary expected return under the policy from that state.
- Subtraction compares the specified action with the policy's average action choice.

If a state's value is 20 and an action's value is 24, advantage is +4. An action value of 18 gives advantage -2. A negative advantage can coexist with a positive immediate reward if the action has worse long-term consequences than alternatives.

### 9.4 Bellman expectation equation

**Read equation E18 in words:** For each action and outcome, average reward now plus discounted next value.

```text
V^π(s) = Σ_a π(a|s) Σ_(s',r) p(s',r|s,a)[r + γV^π(s')] [E18]
```

**Calculate it step by step:** For the single-action example below, the favourable target is one plus 0.9 times ten, which is ten. Weight it by probability 0.8 to get eight. The other target is negative two; weight by 0.2 to get negative 0.4. Add to get 7.6.


- `V^π(s)` is the state value being expressed recursively.
- The outer sum ranges over possible actions a.
- `π(a|s)` is the probability of choosing action a in state s.
- The inner sum ranges over next-state and reward outcomes.
- `p(s',r|s,a)` is the transition-and-reward probability from E14.
- `r` is immediate reward; `γ` is discount; `V^π(s')` is next-state value.
- Brackets group immediate reward plus discounted future value.
- Multiplication weights that quantity by action and outcome probabilities.
- For continuous spaces, suitable integrals replace sums.

**Why:** E10 separates return into a first reward and the remaining return; taking conditional expectations and averaging possible actions/outcomes gives E18.

**Worked example:** There is one available action. With probability 0.8 it yields reward 1 and next-state value 10; with probability 0.2 it yields reward -2 and next-state value 0. With gamma 0.9, current value is `0.8 × (1 + 0.9 × 10) + 0.2 × (-2 + 0.9 × 0) = 7.6`. This is an expectation, not the observed return of one episode.

### 9.5 Bellman optimality

**Read equation E19 in words:** Choose the action with the best expected immediate-plus-future score.

```text
V*(s) = max_a Σ_(s',r) p(s',r|s,a)[r + γV*(s')]        [E19]
```

**Calculate it step by step:** Compute each candidate action value using outcome probabilities. If one candidate gives expected score five and another seven, take the larger number seven. The star means optimal under this model and objective.


- `V*` means the optimal value achievable under the model and objective; the star is a label for optimality.
- `s`, `a`, `s'`, `r`, `p`, and `γ` mean the same as in E18.
- The sum averages the transition outcomes of a fixed action.
- `max_a` selects the largest expected value among available actions.
- Next-state value is optimal too, because future action choices are also optimized.

This is a characterization of an optimal solution. A neural PPO run does not automatically satisfy it or attain a global optimum.

</details>

## 10. Bandits and exploration

### Start here: Trying new actions versus using what you know

Suppose you are comparing three grasp settings. One has worked before, while the others are less tested. Always choosing the familiar setting may prevent discovering a better one.

**Exploitation** uses current knowledge. **Exploration** gathers new experience. A **bandit problem** studies this choice without the full sequence of changing robot states.

Epsilon-greedy means mostly choosing the current best option, while sometimes choosing randomly. Upper confidence bound adds an exploration bonus to less-tested options.

These are useful small learning examples. Your PPO robot instead samples continuous actions using a Gaussian distribution, a bell-shaped noise distribution.

**Write in your notebook:** Exploration is valuable when it reveals useful behaviour. More random numbers do not automatically mean more useful robot experience.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

Bandits isolate action choice without sequential state transitions. They help explain exploration but do not model an entire grasp sequence. A contextual bandit includes current context, while still omitting the full effect of actions on future states.

### 10.1 Incremental action-value estimate

**Read equation E20 in words:** Move the old action estimate a small part of the way toward the new reward.

```text
Q_(n+1)(a) = Q_n(a) + α_n[R_n - Q_n(a)]               [E20]
```

**Calculate it step by step:** Subtract old estimate two from observed reward five to get error three. Multiply by learning step size 0.1 to get correction 0.3. Add it to the old estimate to obtain 2.3.


- `a` is the selected bandit action.
- `n` counts its observed samples, rather than physical decision time.
- `Q_n(a)` is its old estimated mean reward.
- `R_n` is the newly observed reward for that action.
- `α_n` is a learning step size; using one divided by the sample count gives a running average with consistent indexing.
- The bracket is prediction error; multiplication controls how much to correct it.
- `Q_(n+1)(a)` is the updated estimate.

Old estimate 2, new reward 5, and step size 0.1 give updated estimate 2.3. A constant step size emphasizes recent samples and can track changing rewards; a decreasing sample-average step size is useful for stationary means.

### 10.2 ε-greedy and optimism

ε-greedy chooses a uniformly random available action with probability epsilon and a current best-estimated action otherwise. Epsilon is a probability between zero and one. The random branch can still choose the greedy action, so greedy-action probability is not always exactly one minus epsilon.

Optimistic initial values encourage trying poorly measured actions because their initial estimates are high. Neither scheme is your continuous Gaussian PPO exploration mechanism.

### 10.3 Upper-confidence-bound selection

**Read equation E21 in words:** Choose the action with the largest estimated score plus exploration bonus.

```text
a_t = argmax_a [Q_t(a) + c sqrt(log(t)/N_t(a))]        [E21]
```

**Calculate it step by step:** For 100 selections, log of 100 is about 4.605. Divide by an action count of ten, then take the square root, about 0.679. With coefficient one, add this bonus to that action estimate. Compare all actions, and return the winning action.


- `a_t` is the selected action at selection index t.
- `argmax_a` returns the action attaining the largest bracketed score.
- `Q_t(a)` is the estimated reward of action a.
- `c` is a nonnegative exploration coefficient.
- `sqrt` is square root; `log(t)` is the natural logarithm of the number of selections.
- `N_t(a)` counts prior selections of action a.
- The added term is an uncertainty-inspired bonus that is larger for rarely tried actions.
- An action with zero count is tried explicitly; do not literally divide by zero.

If t is 100, an action count is 10, and c is 1, its bonus is about 0.679. This is a bandit rule under its assumptions, not a grasp-success guarantee.

</details>

## 11. Dynamic programming and planning

### Start here: Learning with a model of what will happen

A **model** predicts the next state and reward for an action. If we know a small environment's model, we can compare possible outcomes without physically trying every action.

**Dynamic programming** repeatedly updates values using that model. First we can evaluate the current policy. Then we can improve action choices. **Value iteration** repeatedly chooses the best expected outcome while updating the values.

**Example:** one action earns a point now but leads somewhere poor. Another earns nothing now but leads somewhere valuable. Looking at future value can make the second action preferable.

A simulator contains a physics model. However, your PPO learner uses the experiences generated by that simulator; it does not search predicted action sequences before each command.

**Write in your notebook:** Planning uses model predictions to choose behaviour. Model-free learning improves behaviour from sampled experience.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

**Prediction:** estimate a fixed policy's value. **Control:** improve action selection. **Policy iteration:** alternate evaluating a policy with improving it. **Value iteration:** interleave short evaluation and greedy improvement using optimality backups.

Dynamic programming uses an available transition model and averages possible outcomes. It is useful for learning Bellman reasoning on small problems. Tabulating every SO-101 configuration, velocity, image, and contact state is infeasible.

### 11.1 Value iteration

**Read equation E22 in words:** Use the model and current future estimates to make an improved value table.

```text
V_(k+1)(s) = max_a Σ_(s',r) p(s',r|s,a)[r + γV_k(s')] [E22]
```

**Calculate it step by step:** For the first action below, one plus 0.9 times four gives 4.6. For the second, zero plus 0.9 times six gives 5.4. Store the larger candidate, 5.4, as the updated state value.


- `k` counts value-iteration sweeps, not robot steps.
- `V_k` is the current value table; `V_(k+1)` is the updated table.
- `s` is a current state and `a` an action considered.
- `s'` and `r` are next-state and immediate-reward outcomes.
- `p` weights those outcomes by model probability.
- `γ` discounts the current estimate of next-state value.
- `Σ` averages outcomes; `max_a` chooses the best expected action result.

For deterministic action outcomes, suppose one action gives reward 1 and next value 4, and another gives reward 0 and next value 6. At gamma 0.9, the candidate values are 4.6 and 5.4. The updated value is 5.4, despite that action's smaller immediate reward.

### 11.2 Dyna and model errors

Dyna combines real interaction, learning a model, direct value updates, and updates from simulated model experience. Prioritized sweeping allocates planning work to important changes. A wrong model can propagate wrong values through many planning updates.

In robotics, model predictive control uses a model to compare short-horizon action sequences and repeatedly replan. It can serve as a baseline, demonstration source, or component of a hybrid system. PPO here learns from simulator samples; it does not run this planning search for every action.

</details>

## 12. Monte Carlo, TD, SARSA, and Q-learning

### Start here: Learning from the result or from a prediction

**Monte Carlo learning** waits for a completed attempt and uses the rewards that actually followed. **Temporal-difference learning**, or TD, can update earlier using the next state's predicted value.

That use of a prediction inside a learning target is called **bootstrapping**. It can provide feedback sooner, but an inaccurate prediction can make the target inaccurate too.

**Example:** the current value prediction is four. The next reward is two, and discounted predicted future value is 4.5. The new target is 6.5, so the prediction was too low by 2.5.

SARSA uses the next action chosen by its policy in the target. Q-learning uses the best estimated next action. This explains their on-policy/off-policy difference.

**Write in your notebook:** TD compares a prediction with reward plus predicted continuation. After a true episode end, continuation is zero.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 12.1 Monte Carlo prediction

Monte Carlo learns from observed complete returns. First-visit methods use a state's first visit per episode; every-visit methods use every visit. Both need sufficient relevant experience, and correlated visits should not be mistaken for independent trials.

**Read equation E23 in words:** After an episode, correct the value prediction toward the observed return.

```text
V(S_t) ← V(S_t) + α[G_t - V(S_t)]                    [E23]
```

**Calculate it step by step:** Observed return ten minus old prediction four gives error six. Multiply by step size 0.2 to get correction 1.2. Add to old prediction four to obtain 5.2.


- `V(S_t)` is the stored estimated value of the visited state.
- `←` replaces its old estimate with the right-hand-side result.
- `α` is the step size.
- `G_t` is actual observed discounted return from that visit to the episode end.
- The bracket is return minus prediction.

If predicted value is 4, observed return is 10, and alpha is 0.2, the estimate becomes 5.2. Monte Carlo does not use an estimated next-state value in this target. Complete returns can be noisy and delayed.

### 12.2 TD residual and prediction update

**Read equation E24 in words:** Compare current prediction with reward plus predicted continuation.

```text
δ_t = R_(t+1) + γ m_t V(S_(t+1)) - V(S_t)
V(S_t) ← V(S_t) + α δ_t                              [E24]
```

**Calculate it step by step:** For a continuing transition, multiply next value five by discount 0.9 to get 4.5. Add reward two to get target 6.5. Subtract current value four to get TD error 2.5. At step size 0.1, increase value by 0.25. At true termination, omit continuation.


- `δ_t` is the one-step TD residual.
- `R_(t+1)` is observed immediate reward.
- `γ` is discount; `m_t` is one if future task return continues and zero for a true terminal transition.
- `V(S_(t+1))` estimates next-state return; `V(S_t)` is the old current prediction.
- Their combination compares a bootstrapped target with the current prediction.
- `α` is a learning step size; `←` denotes updating the stored value.

With reward 2, gamma 0.9, continuing mask 1, next value 5, and current value 4, the residual is 2.5. At alpha 0.1, current estimate becomes 4.25.

At a terminal transition with reward 2 and current value 4, the residual is -2 because future return is zero. Bootstrapping from the next episode's reset state would be incorrect.

**Bootstrapping** means using an estimate inside the target used to update another estimate. It can reduce variance and support incremental learning, but errors in the estimate affect the target.

### 12.3 SARSA

**Read equation E25 in words:** SARSA updates using the value of the next action actually chosen by its policy.

```text
Q(S_t,A_t) ← Q(S_t,A_t)
    + α[R_(t+1) + γ m_t Q(S_(t+1),A_(t+1)) - Q(S_t,A_t)] [E25]
```

**Calculate it step by step:** Take immediate reward. Add the discounted value of the recorded next state and next action if the episode continues. Subtract the current action estimate. Multiply by the learning step size and add the correction to that estimate.


- `Q` is the estimated action-value table.
- `S_t,A_t` identify the current state and action.
- `S_(t+1),A_(t+1)` are the next state and next action chosen under the behaviour policy.
- `R_(t+1)` is reward; `α` is step size; `γ` is discount.
- `m_t` prevents future value at a true terminal transition.
- The bracket is the TD target minus the current action-value estimate.
- `←` writes the correction to that table entry.

SARSA is on-policy when the updated values evaluate the same policy generating actions, including its exploration. This can prefer a safer route near a failure boundary because exploratory behaviour is included in expected consequences.

### 12.4 Q-learning and Expected SARSA

**Read equation E26 in words:** Q-learning updates using the best estimated next action.

```text
Q(S_t,A_t) ← Q(S_t,A_t)
    + α[R_(t+1) + γ m_t max_a Q(S_(t+1),a) - Q(S_t,A_t)] [E26]
```

**Calculate it step by step:** If next actions have estimated values three and five, use five in the target, even if exploration chose the other action. With reward two and discount 0.9, continuation gives target 6.5. Then correct the old estimate toward that target.


- Current state/action, reward, step size, discount, and terminal mask have the meanings in E25.
- `a` ranges over possible next actions; it is not necessarily the action actually sampled.
- `max_a Q(S_(t+1),a)` is the largest estimated next-state action value.
- The bracket compares this greedy bootstrapped target with the old value.
- `←` updates the selected entry.

Q-learning's target evaluates greedy behaviour while experience can come from an exploratory behaviour policy: this is off-policy learning. Expected SARSA replaces the sampled next action's value with a policy-weighted expected next-action value.

**Maximization bias:** maximizing noisy estimates can select overestimated actions. Double-learning methods decouple selection and evaluation to reduce this effect. Deep Q-networks add neural approximation and implementation devices such as replay and target networks; their finite-action maximization is awkward for your continuous joint commands.

Tabular convergence results require conditions such as sufficient visits and appropriate step sizes. They do not transfer unchanged to arbitrary deep robot policies.

</details>

## 13. Multi-step learning, eligibility traces, and GAE

### Start here: Connecting a later lift to earlier actions

A successful lift may depend on several earlier choices: approaching, opening, centering, closing, then raising the arm. Learning needs to connect the later outcome with those earlier decisions.

An **n-step target** combines several observed rewards before using a value prediction for the rest. An **eligibility trace** keeps fading learning credit for recently active features.

**GAE**, generalized advantage estimation, combines TD errors from several nearby steps to estimate how favourable an action was. Gamma discounts later information. Lambda controls how strongly the estimate includes more distant errors.

**Example:** one positive error is followed by another positive error. Both can contribute to the earlier action's advantage, with the later one weighted less.

Your current PPO uses gamma 0.99 and lambda 0.95. These are settings to understand and evaluate, not proof the robot will learn.

**Write in your notebook:** Multi-step methods let feedback from later events influence earlier decisions.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 13.1 n-step target

**Read equation E27 in words:** Use n observed rewards, then predict the rest.

```text
G_t^(n) = Σ_(k=0)^(n-1) γ^k R_(t+k+1) + γ^n V(S_(t+n)) [E27]
```

**Calculate it step by step:** For two rewards one and two, discount 0.9, and next value five after two steps: count one; add 0.9 times two, giving 1.8; add 0.9 squared times five, giving 4.05. Total target is 6.85.


- `G_t^(n)` is an n-step target; the parenthesized superscript labels the horizon.
- `n` is the number of actual reward steps before bootstrapping.
- `k` indexes those steps from zero through n minus one.
- `γ^k` discounts each observed reward `R_(t+k+1)`.
- `V(S_(t+n))` estimates return after the nth step; `γ^n` discounts it.
- Addition combines observed short-horizon return and estimated continuation.
- This displayed form assumes the episode continues to that bootstrap point. If it ends sooner, stop at terminal reward and omit future value.

For two observed rewards 1 and 2, gamma 0.9, and value 5 after two steps, the target is 6.85. Longer horizons use more observed outcomes but often increase variance.

### 13.2 Eligibility traces

One common accumulating trace for a linear value predictor is:

**Read equation E28 in words:** Keep fading credit for recent features, then use it to assign the current learning correction.

```text
e_t = γλ e_(t-1) + x(S_t)
w_(t+1) = w_t + α δ_t e_t                            [E28]
```

**Calculate it step by step:** Previous trace 0.5 is multiplied by gamma 0.9 and lambda 0.8, giving 0.36. Add current feature one to get trace 1.36. For TD error two and step size 0.1, a scalar weight correction would be 0.272.


- `e_t` is a vector retaining decayed credit for recently active features.
- `e_(t-1)` is its previous value, reset at episode boundaries as required.
- `γ` is discount and `λ` controls trace persistence.
- `x(S_t)` is the current state's feature vector.
- `w_t` and `w_(t+1)` are old and updated linear value weights.
- `α` is learning step size; `δ_t` is TD residual.
- Vector addition updates traces; multiplying the trace by the residual assigns credit to recent features.

If the previous scalar trace is 0.5, gamma is 0.9, lambda is 0.8, and current feature is 1, the new trace is 1.36. Eligibility traces concern learning memory; they are different from an actuator's low-pass filter memory.

### 13.3 Generalized advantage estimation

For a rollout that stays within one episode:

**Read equation E29 in words:** Combine nearby TD errors while weighting later ones less.

```text
Â_t = Σ_(l=0)^(K-1) (γλ)^l δ_(t+l)                  [E29]
```

**Calculate it step by step:** Multiply gamma 0.99 by lambda 0.95 to get 0.9405. For errors one, two, negative one: count one; add 0.9405 times two, giving 1.881; subtract 0.9405 squared, about 0.88454. The estimate is about 1.99646.


- `Â_t` is estimated advantage at t; the hat marks an estimate.
- `K` counts available residuals from t to the rollout boundary.
- `l` counts each residual's delay, beginning at zero.
- `δ_(t+l)` is the TD residual of that future transition.
- `γλ` multiplies discount and trace weighting; its lth power controls decay.
- The sum combines residuals over the available segment.
- Reset the accumulation across actual episode boundaries. The last nonterminal residual may use a bootstrapped value at the rollout end.

With residuals 1, 2, -1, gamma 0.99, and lambda 0.95, the weighting product is 0.9405 and the first advantage estimate is approximately 1.99646. Your PPO uses these gamma/lambda values.

An associated critic training target is:

**Read equation E30 in words:** Add the advantage estimate back to the old value prediction to obtain a return target.

```text
Ŷ_t = Â_t + V_old(S_t)                               [E30]
```

**Calculate it step by step:** Old value four plus advantage estimate 1.99646 gives target 5.99646. The critic learns toward that fixed target during the update. It is a constructed estimate, not necessarily a measured complete return.


- `Ŷ_t` is the bootstrapped return target for the critic.
- `Â_t` is the advantage estimate computed for the rollout.
- `V_old(S_t)` is the value estimate used during collection/target construction.
- Addition reconstructs a return estimate from advantage plus baseline.
- Targets should be detached from gradients during the subsequent update.

With advantage 1.99646 and old value 4, the target is 5.99646. These are estimates, not observed full-episode returns. GAE's weighted residual construction and bias–variance tradeoff are described in the [GAE paper](https://arxiv.org/abs/1506.02438).

Lambda near zero emphasizes short bootstrapped feedback. Larger lambda uses more distant sampled outcomes and often trades less bootstrap bias for more variance. At one, GAE on a terminal-complete trajectory approaches a return-minus-baseline estimate; a rollout cutoff still involves a bootstrap.

</details>

## 14. Function approximation and numerical stability

### Start here: Why we use neural networks

There are too many possible robot angles, velocities, cube positions, and images to store a separate table entry for every situation. A neural network predicts outputs for many situations using shared weights.

This is **function approximation**: using a model to represent a function that is too large to list completely. Sharing weights lets experience in one situation influence predictions in another.

A **loss** measures an error we want to reduce. The critic's loss compares its future-score prediction with a training target. A **gradient** indicates how weights affect that loss. The optimizer changes the weights.

**Example:** the critic predicts four but the target is five. Its error is negative one; squaring gives one. Reducing this error improves prediction of the target.

**Write in your notebook:** A low prediction error does not prove the robot performs the task. Check the behaviour separately.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

Tabular methods store values separately. Function approximation uses shared parameters so updating one experience affects other inputs too. Linear features, tile coding, radial basis features, and neural networks are alternatives. A CNN extracts spatial image features; an MLP processes numerical inputs.

### 14.1 Linear value prediction

**Read equation E31 in words:** Multiply each feature by its matching weight and add the results.

```text
V_w(s) = wᵀ x(s) = Σ_(j=1)^d w_j x_j(s)             [E31]
```

**Calculate it step by step:** For weights two and negative one, and features three and four, multiply to get six and negative four. Add them to predict value two. The transpose symbol simply lets the vector product express this sum.


- `V_w(s)` is predicted state value under parameters w.
- `w` is a vector of d weights; `x(s)` is a vector of d state features.
- Superscript `ᵀ` means transpose, turning a column vector into a row vector for the dot product.
- `j` indexes matching weight/feature components.
- `w_j x_j(s)` is each feature's weighted contribution; the sum adds them.

Weights 2 and -1 with features 3 and 4 give predicted value 2. Neural predictors generalize this computation using nonlinear layers.

### 14.2 Critic squared-error loss

**Read equation E32 in words:** Find each prediction error, square it, then average the squared errors.

```text
L_V(w) = (1/B) Σ_(i=1)^B [V_w(s_i) - Ŷ_i]²          [E32]
```

**Calculate it step by step:** Predictions four and seven with targets five and five give errors negative one and two. Squaring gives one and four. Add to get five; divide by two samples to get mean loss 2.5.


- `L_V` is the critic loss; w are critic parameters.
- `B` is the minibatch sample count; `i` identifies a sample.
- `s_i` is the critic input for that sample.
- `V_w(s_i)` is its current prediction; `Ŷ_i` is a fixed constructed return target.
- Subtraction gives prediction error; squaring treats positive and negative errors symmetrically.
- The sum divided by B averages squared errors.

Predictions 4 and 7 with targets 5 and 5 give errors -1 and 2; mean squared loss is 2.5. A lower loss means better fitting to these targets, not necessarily a better grasp policy. Your implementation also enables clipped value loss, so E32 is the basic explanatory form rather than every implementation detail.

### 14.3 Gradient descent

**Read equation E33 in words:** Change the weights in the direction that reduces loss.

```text
w_new = w_old - η ∇_w L_V(w_old)                     [E33]
```

**Calculate it step by step:** For scalar weight one, loss derivative two, and learning rate 0.1, multiply derivative by rate to get 0.2. Subtract from the old weight to get 0.8. A derivative says how a small weight change affects loss locally.


- `w_old` and `w_new` are old and updated parameter vectors.
- `η` is the optimizer learning rate, distinct from a robot's action filter coefficient.
- `∇_w` is the vector of derivatives of loss with respect to weights.
- `L_V(w_old)` evaluates loss at old weights.
- Subtraction moves opposite the local increase direction.

For scalar weight 1, loss derivative 2, and learning rate 0.1, the new weight is 0.8. Your optimizer uses adaptive updates rather than necessarily this exact plain-gradient rule, but the descent principle is the same.

### 14.4 The deadly triad and other failures

Sutton–Barto highlights potential instability when combining bootstrapping, function approximation, and off-policy learning. This does not mean every SAC run diverges or every PPO run is stable. Optimization, representation, data, and physical simulation can cause separate failures.

Inspect inputs, predictions, targets, gradients, actions, and physical state for nonfinite values. If robot joint state becomes NaN under bounded random commands without learning, the evidence points to physics/articulation dynamics before critic optimization. A neural network cannot recover the correct world state from invalid input by receiving more iterations.

Normalization can improve conditioning when inputs have different scales, but training and evaluation must use the same learned normalization. Reward-scale changes alter target size. Unit tests of a formula establish limited numerical behaviour; successful task learning requires actual episodes.

</details>

## 15. Policy gradients and actor–critic learning

### Start here: The actor chooses; the critic estimates

Your learning system has two jobs. The **actor** chooses robot actions. The **critic** estimates future return and helps judge those actions.

A **policy gradient** is a direction for changing the actor's weights to improve expected return. It uses experience to make actions with better-than-usual outcomes more likely, and worse-than-usual actions less likely on average.

**Example:** from a similar aligned pose, closure followed by lifting leads to a better result than moving away. That experience can guide the actor toward more useful choices.

The critic does not send commands to the motor. It provides a prediction used in learning. Its quality matters because misleading predictions can give poor guidance.

**Write in your notebook:** Actor learning improves action choices. Critic learning improves future-score predictions. Evaluate both, but task success is the final outcome measure.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 15.1 Why directly learn a policy?

In a six-dimensional continuous action space, enumerating every candidate joint command is impractical. A policy network instead outputs distribution parameters directly. Value-based methods can also handle continuous control with additional machinery; continuous action space does not make value functions irrelevant.

The objective for a specified initial-state distribution is:

**Read equation E34 in words:** Judge the policy by its average starting return.

```text
J(θ) = E_(π_θ)[G_0]                                 [E34]
```

**Calculate it step by step:** Run or imagine many attempts using the policy and the defined starting conditions. Calculate each attempt return. Average across the possible outcomes. Training aims to increase that expected quantity.


- `J` is expected performance as a function of policy parameters theta.
- `θ` includes trainable policy weights and potentially distribution parameters.
- `E_(π_θ)` averages trajectories generated by that policy, the environment, and the chosen initial-state distribution.
- `G_0` is discounted return from the beginning of an episode.
- Maximizing J asks for larger expected return, not success on every individual attempt.

### 15.2 The policy-gradient direction

One finite-episode discounted form is:

**Read equation E35 in words:** Add learning directions that favour actions with better-than-usual outcomes.

```text
∇_θ J(θ) = E_(π_θ)[Σ_(t=0)^(T-1) γ^t
                    ∇_θ log π_θ(A_t|S_t) A^(π_θ)(S_t,A_t)] [E35]
```

**Calculate it step by step:** The log-policy gradient is a vector describing how weights affect the chosen action likelihood. Multiply it by advantage and the time weighting, then combine steps and average trajectories. For a one-component direction 0.5, advantage four, and time factor one, the contribution is two.


- `∇_θ J` is a vector showing the local increase direction of expected performance.
- `E_(π_θ)` averages sampled policy/environment trajectories.
- `t` indexes decisions up to terminal time T; the sum combines their contributions.
- `γ^t` accounts for discount from the episode start in this formulation.
- `log π_θ(A_t|S_t)` is log-probability or log-density of the sampled action.
- `∇_θ log π_θ` is its derivative vector, **not merely its scalar log value**.
- `A^(π_θ)(S_t,A_t)` is advantage; the superscript names the policy.
- Multiplication scales the action's parameter direction by how favourable its consequences are relative to the baseline.
- For partial observations, a corresponding formulation uses policy inputs/history. Implementations may express discounted state weighting through the sampling distribution rather than this explicit episodic sum.

The identity behind this is that a probability's log derivative equals its derivative divided by that probability. The environment's unknown dynamics need not be differentiated when they are independent of theta; trajectory likelihood differentiation supplies the policy terms.

REINFORCE estimates the direction using sampled complete returns. A state-dependent baseline can reduce variance without changing the expected score-function direction when it is independent of the sampled action and handled correctly during differentiation. A bootstrapped actor–critic uses estimated value information to get earlier, often lower-variance feedback at the cost of possible approximation bias.

### 15.3 Actor and critic in SO-101

**Actor:** produces six action means and learns Gaussian exploration parameters. The CNN variant combines RGB with numerical features. The state-only MLP variant uses the numerical features without RGB.

**Critic:** predicts a scalar expected future return from richer numerical information. It helps build targets and advantages. It does not command the motors.

Your networks have separate configurations. It is not correct to assume a shared backbone because another tutorial uses one. The privileged critic can see information unavailable on hardware, but an actor receiving a simulator cube-position cue still has its own deployment gap.

</details>

## 16. PPO and your training configuration

### Start here: PPO learns in repeated collect-and-update cycles

Your project uses **PPO**, Proximal Policy Optimization. First it runs the current policy and collects experience. Then it uses that recent experience for several rounds of updates.

A very large change can spoil behaviour learned earlier. PPO's clipped objective reduces some incentives for excessively large changes. It is not a physical safety limit and does not force every change to remain inside a hard bound.

**Example:** an action had positive advantage. PPO encourages making it more likely, but the clipped objective stops increasing the sample's benefit beyond a certain amount in that direction.

The actor also has **exploration noise**, which produces different actions during training. **Entropy** describes the spread of its action distribution. Useful exploration must still produce useful physical movement.

**Write in your notebook:** PPO collects recent experience, estimates advantages, and updates the policy several times before collecting more.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

PPO alternates experience collection and several optimization passes over that recent rollout. The objective discourages certain excessive changes relative to the collecting policy; its clipping is not a hard bound on all policy ratios. These corrections are supported by the [original PPO paper](https://arxiv.org/abs/1707.06347).

### 16.1 Probability ratio

**Read equation E36 in words:** Compare the likelihood of the same recorded action under new and old policies.

```text
ρ_t(θ) = π_θ(A_t|O_t) / π_old(A_t|O_t)                [E36]
```

**Calculate it step by step:** New density 0.3 divided by old density 0.2 gives ratio 1.5. Use the same recorded action and observation on both sides. The ratio is a likelihood comparison, not the physical action size.


- `ρ_t` is the new-to-old probability-density ratio for the recorded action.
- `θ` names current policy parameters being updated.
- `π_θ` is the current policy and `π_old` the policy that collected the rollout.
- `A_t` and `O_t` are the recorded sampled action and observation.
- Division compares how likely that same action is under the two policies.
- In continuous control these are densities, not probabilities of an exact real-valued point.

Old density 0.2 and new density 0.3 give ratio 1.5. Density is not bounded by one in general; the ratio is still meaningful.

### 16.2 Clipped actor objective

**Read equation E37 in words:** Compare the usual advantage-weighted ratio with the clipped version, and keep the smaller result.

```text
L_clip(θ) = E_t[min(ρ_t(θ) Â_t,
                    clip(ρ_t(θ),1-ε,1+ε) Â_t)]       [E37]
```

**Calculate it step by step:** With width 0.2, the clipped interval is 0.8 to 1.2. Positive advantage two and ratio 1.5 give three before clipping and 2.4 after clipping; keep 2.4. With negative advantages, work through both products before choosing the minimum.


- `L_clip` is the surrogate objective the actor attempts to maximize.
- `E_t` is a sample average over rollout items, not an average over seconds.
- `ρ_t` is the ratio from E36.
- `Â_t` is the fixed rollout advantage estimate.
- `ε` is the clipping width, 0.2 in your configuration.
- `clip` bounds the ratio within the interval for the clipped branch only.
- `min` chooses the smaller of the ordinary and clipped advantage-weighted terms.
- Parentheses group function inputs; products multiply a ratio by an advantage.

For advantage +2 and ratio 1.5, the ordinary term is 3 and clipped term 2.4, so this sample contributes 2.4. For advantage -2 and ratio 0.5, the ordinary term is -1 and clipped term -1.6; the minimum is -1.6. The sign matters: clipping removes excessive improvement incentives in different directions for positive and negative advantages.

The ratio itself can still move beyond the interval because shared parameters, other samples, entropy, and repeated updates affect the network. Clipping is also different from physically clipping an action command.

### 16.3 Actor, critic, and entropy losses

A common minimization convention is:

**Read equation E38 in words:** Combine learning goals using a loss-minimization convention.

```text
L_total = -L_clip + c_V L_V - c_H H                   [E38]
```

**Calculate it step by step:** Negative actor objective encourages its increase. Positive critic loss encourages prediction-error reduction. Negative entropy encourages distributional spread. If actor objective is two, critic loss three, critic coefficient one, entropy 0.5, and entropy coefficient 0.01, this explanatory total is 0.995.


- `L_total` is a combined explanatory loss convention.
- `L_clip` is the actor objective from E37; the minus sign converts maximization to minimization.
- `L_V` is critic prediction loss, with coefficient `c_V`.
- `H` is policy entropy; coefficient `c_H` weights exploration.
- The negative entropy term makes minimizing loss favour higher entropy.
- Coefficients are hyperparameters. Separate actor/critic optimizers can express these objectives separately; this formula does not imply shared parameters in your implementation.

### 16.4 Gaussian exploration

**Read equation E39 in words:** Start with the mean action and add a scaled random sample.

```text
a_j = μ_j(o) + σ_j z_j,       z_j ~ Normal(0,1)       [E39]
```

**Calculate it step by step:** Mean negative 0.3, standard deviation 0.4, and random standard-normal sample 0.5 give noise 0.2. Add it to the mean to obtain sampled action negative 0.1. Physical clipping comes later.


- `a_j` is action component j.
- `μ_j(o)` is that component's policy mean given observation o.
- `σ_j` is its positive learned standard deviation.
- `z_j` is a standard-normal sample with mean zero and variance one.
- Multiplication scales random noise; addition shifts it to the mean.
- `j` runs over the six commanded components in this task.

Mean -0.3, standard deviation 0.4, and sampled z of 0.5 produce action -0.1 before physical clipping.

For one Gaussian component, differential entropy is:

**Read equation E40 in words:** Compute a Gaussian action distribution spread using its standard deviation.

```text
H_j = 0.5 log(2πe σ_j²)                              [E40]
```

**Calculate it step by step:** At standard deviation 0.1, first square it to get 0.01. Multiply by two, the pi constant, and the e constant to get about 0.1708. Its natural log is about negative 1.7673. Half gives entropy about negative 0.8836, which is valid for a continuous distribution.


- `H_j` is entropy of one continuous Gaussian action component, measured in nats under these coordinates.
- `log` is natural logarithm; `π` here is the constant approximately 3.14159, not the policy symbol.
- `e` is the exponential constant approximately 2.71828.
- `σ_j²` is variance, the square of the standard deviation.
- The coefficient 0.5 multiplies the logarithm.
- Independent-component entropies add to obtain joint entropy.

With standard deviation 0.1, component entropy is about -0.884. Negative differential entropy is valid. It is not a negative probability or automatic failure. Your dashboard entropy is measured before clipping; large raw entropy can coexist with many identical saturated physical commands.

### 16.5 Current settings and their meaning

| Setting | Current value | Meaning |
| --- | ---: | --- |
| Physics interval | 0.01 s | 100 physics updates per second |
| Decimation | 4 | Four physics updates per policy decision |
| Policy interval | 0.04 s | 25 actions per second |
| Episode duration | 6 s | At most 150 decision steps |
| Rollout steps per environment | 32 | Experiences before each PPO update cycle |
| Discount | 0.99 | Future reward weighting per policy step |
| GAE lambda | 0.95 | Multi-step advantage weighting |
| Clip parameter | 0.2 | Surrogate ratio clip width |
| Learning rate | 0.0003 | Fixed optimizer rate in this configuration |
| Learning epochs | 5 | Five passes over each collected rollout |
| Minibatches per epoch | 4 | Each pass subdivides rollout data |
| Gradient norm bound | 1 | Limits optimizer gradient norm, not joint torque |
| Initial action standard deviation | 0.4 | Initial exploration scale |
| Allowed standard deviations | 0.1–0.8 | Bounds on learned raw Gaussian noise |
| Entropy coefficient | Initially 0.01 | Adapted within 0.001–0.05 |
| Entropy adaptation target | Equivalent to std 0.2 | A heuristic target for exploration spread |

The adaptation responds to mean log standard deviation; it is not a novelty estimate based on the current scene. Check each joint's noise and action saturation, rather than concluding that one entropy curve proves good exploration.

### 16.6 One PPO iteration

1. Run the current stochastic actor for 32 decisions in each environment.
2. Save observations, actions, old log-densities, rewards, episode boundaries, and old values.
3. Bootstrap the final nonterminal state and compute GAE without crossing episode boundaries.
4. Construct return targets and advantage estimates.
5. Optimize actor and critic over five epochs and four minibatches per epoch.
6. Bound learned exploration scales and adapt the entropy coefficient.
7. Log diagnostics and continue collecting fresh experience.

Policy learning is the repeated change in network parameters; motor movement occurs during environment stepping. A training iteration is not a single robot action.

</details>

## 17. SAC, replay, and HER

### Start here: SAC learns from stored experience

**SAC**, Soft Actor-Critic, is another continuous-control algorithm. It stores transitions in a **replay buffer**, a collection of past experience, and samples them again for learning.

PPO mainly learns from recent rollouts. SAC can reuse older experience. SAC also includes an entropy preference to encourage a spread of actions. Its practical behaviour depends on the task and implementation.

**HER**, hindsight experience replay, changes the goal label of suitable past experience. If the robot reached position B instead of intended A, the transition may teach it about reaching B. Rewards must be recomputed for that goal.

Your state-only option still uses PPO. Removing images does not change the algorithm to SAC. HER is not currently a switch in your PPO pipeline.

**Write in your notebook:** Actor inputs, learning algorithm, and replay method are separate design choices.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 17.1 SAC's objective

SAC is an off-policy maximum-entropy actor–critic method. Its reward objective includes an entropy preference, and replay allows reusing older transitions. Modern common implementations use twin action-value critics and can adapt the entropy temperature. These are algorithm features, not guarantees of superior performance on every task. See the [SAC paper](https://arxiv.org/abs/1801.01290) and [SAC algorithms and applications](https://arxiv.org/abs/1812.05905).

An explanatory maximum-entropy objective is:

**Read equation E41 in words:** SAC counts reward and an entropy preference in its learning objective.

```text
J_SAC(π) = E_π[Σ_t γ^t(R_(t+1) + α H(π(·|S_t)))]     [E41]
```

**Calculate it step by step:** At one step, reward two and entropy 0.5 with temperature 0.1 combine to 2.05 before discounting. Then weight later combined scores by gamma and average possible experience. Temperature controls the entropy preference, not robot temperature.


- `J_SAC` is the policy's entropy-augmented objective.
- `E_π` averages experience under policy pi and the environment.
- The sum ranges over task steps; `γ^t` discounts each step from the start.
- `R_(t+1)` is environmental reward.
- `H(π(·|S_t))` is entropy of the action distribution at the current state.
- The dot is an action placeholder, and the conditional bar means given the state.
- `α` is a nonnegative temperature balancing entropy with reward, different from the tabular learning rate using the same Greek letter earlier.
- Addition combines the two preferences.

### 17.2 Replay and comparison

A replay buffer stores transitions and allows repeated sampling for updates. Target critics change more slowly than online critics to stabilize bootstrap targets. Twin critics help reduce overestimation. Action squashing and its probability-density correction are important implementation details for continuous SAC.

PPO uses a value baseline and recent rollouts. SAC typically uses action-value critics and replay. Your `--policy state` option changes actor inputs/architecture; it does **not** change PPO into SAC.

The reasonable next algorithm experiment would hold the environment, observations, action mapping, reset distribution, evaluation protocol, and transition budget constant. Changing all of them at once cannot tell us whether SAC helped. Algorithm comparisons are premature if the task is physically impossible or its resets are invalid.

### 17.3 Hindsight experience replay

HER relabels goal-conditioned transitions with goals actually achieved. A failed attempt to move the cube to target A may supply useful learning about reaching another achieved target B. Rewards must be recomputed consistently for the relabeled goal, including relevant termination semantics.

Standard HER is designed around goal-conditioned replay. It is not enabled in your on-policy PPO pipeline and is not a switch that can be added without changing the learner/environment design. It also does not magically transform a robot that never contacts an object into a successful lifting policy: achieved-goal coverage still matters.

</details>

## 18. Robotics control, coordinates, and contact physics

### Start here: A command and an actual movement are different

The actor requests a joint angle. The motor controller tries to reach it. Actual movement depends on force limits, inertia, gravity, friction, and contact with objects.

Your arm targets are filtered and speed-limited so they do not jump abruptly. A filter gradually moves the command toward a new target. The speed limit further restricts how quickly that target changes.

A **coordinate frame** defines an origin and directions. Two positions must be expressed consistently before you subtract them. **Kinematics** describes how joint angles relate to link and tool positions.

**Example:** the actor requests a closed gripper, but the cube blocks further closure. The measured angle can remain around 0.97 radians while holding the object. That is not automatically a failed command.

**Write in your notebook:** Check the requested action, processed target, measured joint motion, and object response separately.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 18.1 What the action interface means

Your actor commands five arm position targets and one gripper target. A position target is a desired angle; actual angle depends on motor gains, effort limits, inertia, gravity, and contacts. The task has five arm coordinates plus gripper opening, not six freely controllable Cartesian pose coordinates.

For an arm action normalized between -1 and +1, a linear soft-limit mapping is:

**Read equation E42 in words:** Convert a normalized arm action to a desired angle inside the joint limits.

```text
q_target = q_min + ((a+1)/2)(q_max-q_min)              [E42]
```

**Calculate it step by step:** With action 0.5, add one and divide by two to obtain range fraction 0.75. With limits negative one and positive one, the range is two. Multiply to get 1.5, then add lower limit negative one to obtain target 0.5 radians.


- `q_target` is desired joint angle in radians.
- `q_min` and `q_max` are that joint's lower and upper soft angle limits.
- `a` is its normalized action.
- Adding one and dividing by two converts the normalized range into a fraction from zero to one.
- The limit difference is available angular range; multiplication selects a location within it.
- This is the arm mapping principle before filtering and slew limiting.

With limits -1 and +1 rad and action 0.5, desired angle is 0.5 rad. Zero action maps to the midpoint, which need not equal the robot's home pose. Your near-grasp actor initialization accounts for this mapping.

For your gripper specifically:

**Read equation E43 in words:** Convert the gripper action to a target angle and restrict it to its valid range.

```text
q_gripper,target = clip(1.20 + 1.20 a_gripper, 0, 1.20) [E43]
```

**Calculate it step by step:** For action negative 0.5, multiply by 1.20 to get negative 0.60. Add open offset 1.20 to get 0.60 radians. For action positive 0.5, the raw target is 1.80, so clipping leaves 1.20 radians open.


- `q_gripper,target` is the commanded gripper angle in radians.
- `a_gripper` is the normalized gripper action.
- The first 1.20 is the open-angle offset; the second 1.20 is the scaling gain.
- `clip` limits the result between zero and 1.20 rad.
- Action -1 requests closed, -0.5 requests 0.60 rad, zero requests fully open, and positive actions saturate at fully open.

A held cube can prevent the actual joint from reaching its requested closing angle. Requesting zero radians does not imply the measured gripper should be zero while physically holding the cube.

### 18.2 Position control

An illustrative PD motor law is:

**Read equation E44 in words:** Combine angle-error correction and velocity-error correction to request torque.

```text
τ = K_p(q_target-q) + K_d(qdot_target-qdot)            [E44]
```

**Calculate it step by step:** Angle error 0.1 multiplied by gain 80 gives eight. Velocity error negative 0.2 multiplied by gain six gives negative 1.2. Add to request 6.8 newton-metres before effort limits. Your actuator may not be allowed to produce that much.


- `τ` is motor torque, measured in newton-metres.
- `K_p` is proportional gain and `K_d` derivative/damping gain.
- `q_target` and `q` are desired and actual joint angle.
- `qdot_target` and `qdot` are desired and actual angular velocity; dot means time derivative.
- Each subtraction is tracking error; gains convert errors into torque contributions.
- Addition combines angle correction and damping/velocity correction.
- This is explanatory; implicit simulator drives may use solver-specific forms and limits.

If angle error is 0.1 rad, proportional gain 80, velocity error -0.2 rad/s, and derivative gain 6, the unconstrained law gives 6.8 Nm. Your arm effort limit is 2.94 Nm, so the actual actuator cannot be assumed to apply 6.8 Nm.

### 18.3 Filtering and slew limits

An exponential moving average has the form:

**Read equation E45 in words:** Move the filtered target partway toward the new raw target.

```text
q_filtered,t = β q_target,t + (1-β) q_filtered,t-1    [E45]
```

**Calculate it step by step:** With new raw target one, old filtered target zero, and coefficient 0.2, the new weighted sum is 0.2 radians. Your additional slew limit can reduce the step further to 0.06 radians in this example.


- `q_filtered,t` is the new filtered reference.
- `q_target,t` is the new raw desired reference.
- `q_filtered,t-1` is previous filter memory.
- `β` is the filter coefficient, 0.2 for your arm.
- `1-β` weights previous memory; the weighted sum smooths target changes.
- This describes the smoothing principle before the additional speed bound.

A raw target of 1 rad, previous filtered target 0, and beta 0.2 produce 0.2 rad before slew limiting. Your additional target-speed limit is 1.5 rad/s; at a 0.04 s step, permitted reference change is 0.06 rad. The final command is therefore more tightly limited in this example.

Smoothing can reduce abrupt movement but adds dynamics and delay. Reset filter memory consistently and expose it when needed for prediction. A reward for smooth actions cannot replace a physically valid actuator model.

### 18.4 Coordinate frames and kinematics

A frame defines origin and axis orientation. Position subtraction is meaningful only when both positions are expressed in the same frame. Your cube-to-tool input subtracts world positions, producing a relative vector expressed along world axes; it is not automatically a vector in the tool's rotated axes.

Forward kinematics predicts link poses from base pose, joint axes, and angles. Inverse kinematics finds angles for a desired tool pose, subject to reachability and joint limits. Compare forward-kinematic predictions with simulator poses before trusting a grasp target.

Parallel environments have different grid origins. Absolute global positions can inject irrelevant offsets; subtracting environment origins gives consistent local coordinates. Quaternion order also matters: your installed configuration uses xyzw. A quaternion with the same four numbers in wxyz order represents a different rotation.

### 18.5 Contacts, friction, and a physically held cube

Visual meshes are for appearance; collision shapes determine contact. A realistic-looking finger can fail to grip if its collider misses the pad. A grasp frame placed at the wrong point can reward proximity outside the actual cavity.

An idealized static friction limit is:

**Read equation E46 in words:** The tangential force needed to prevent sliding must fit within the available friction.

```text
|F_tangent| ≤ μ F_normal                              [E46]
```

**Calculate it step by step:** Multiply friction coefficient 0.8 by normal force one newton to obtain 0.8 newtons of available tangential force in this simplified contact model. Compare the needed tangential force with that limit.


- `F_tangent` is tangential contact force tending to slide the surfaces.
- Vertical bars mean absolute magnitude in this scalar illustration.
- `μ` is static friction coefficient, dimensionless.
- `F_normal` is compressive force perpendicular to the contact surface, in newtons.
- `≤` means the tangential force must stay below the available friction magnitude for sticking in this simplified model.

Friction coefficient 0.8 and normal force 1 N allow up to 0.8 N tangential force at that contact. Real grasping depends on multiple contacts, moment balance, geometry, compliance, and contact solver behaviour. Increasing friction is not a substitute for fixing misplaced pads.

Positive masses, valid inertias, fixed-base constraints, suitable solver settings, and bounded targets matter for stability. A low-mass virtual link with invalid inertia can make an articulation ill-conditioned. A scripted physical grasp test is valuable because it isolates the simulator from policy discovery.

</details>

## 19. Simulation, vectorization, and training budgets

### Start here: Parallel environments collect more experience

One environment is one simulated task instance. Many parallel environments let the learner collect several robot attempts at the same time.

Your PPO rollout collects 32 decisions per environment. With 64 environments, that is 2048 transitions before an update cycle. A transition is one action and its resulting state and reward.

Two runs with the same iteration count may have different amounts of experience if they use different numbers of environments. More environments can also require more memory and rendering work.

**Vectorization** means calculating across a batch of environments together rather than running a separate Python calculation for each one.

**Write in your notebook:** Compare training experience in transitions as well as iterations and wall-clock time. A fast simulator still needs correct physics and task definitions.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 19.1 What a simulator provides

Simulation offers repeatable resets, controllable parameters, faster-than-real-time experience, and access to diagnostic state. It still approximates physical reality. Photorealistic images do not validate friction, actuator response, collision geometry, or base constraints.

MuJoCo, Isaac Lab/Isaac Sim, and Genesis provide different modelling and computational options. Choose according to measured task behaviour, supported sensors and physics, usable hardware, and your workflow. Your current environment explicitly selects PhysX and uses an RTX camera; backend claims from a generic tutorial should not override this configuration.

### 19.2 Vectorization

Parallel simulation stores batches of state and reward tensors. A numerical observation batch has dimensions for environment count and feature count; your image batch also has channel, height, and width dimensions. Compute distances, masks, and rewards across the batch rather than writing one Python loop per environment.

Repeated GPU-to-CPU transfers and scalar extraction can slow training. Rendering every environment can dominate memory and runtime; increasing environment count is not guaranteed to increase throughput. Measure actual transitions per second, update time, rendering cost, and GPU memory.

### 19.3 Experience per iteration

**Read equation E47 in words:** Count experience across environments, rollout steps, and iterations.

```text
N_transitions = N_env × N_rollout × N_iterations      [E47]
```

**Calculate it step by step:** Multiply 64 environments by 32 rollout decisions to get 2048 transitions per iteration. Multiply by 2000 iterations to obtain 4,096,000 transitions. Reusing data during optimizer epochs does not add new environment experience.


- `N_transitions` counts individual state-action transitions collected.
- `N_env` is parallel environment count.
- `N_rollout` is decisions per environment before one PPO update cycle.
- `N_iterations` is the number of collection/update cycles.
- Multiplication counts experience across all environments and cycles; optimizer epochs do not add new physical transitions.

At 64 environments, 32 rollout steps, and 2000 iterations, the budget is 4,096,000 transitions. At 32 environments it is 2,048,000. Comparing runs only by iteration count can therefore be misleading. Evaluation episodes are separate experience and should be reported separately.

### 19.4 Installed-version accuracy

Use actual project CLI flags and the installed interpreter. The base shell currently has `python3`, while the project's Conda environment contains its runtime `python`. Generic commands copied from another version can fail before RL even starts.

Logs and recorded tests can establish historical behaviour. A current launch, physics stress test, and evaluation establish stronger current evidence. These study notes inspect existing evidence and code; they do not launch a new training job or reproduce the recorded simulator tests.

</details>

## 20. Vision, privileged information, and perception

### Start here: Seeing an image is not the same as knowing the cube position

A camera returns an image. A controller may need the cube's location in metres. A **perception system** estimates that location from sensor data.

Your current default actor receives images and exact simulator cube-relative position. The position is **privileged information**: the simulator knows it directly, but the real robot does not automatically have it.

**Example:** a camera estimator is wrong by two centimetres. The robot may close beside the cube even though the same policy succeeds with exact simulated position.

Camera calibration describes how image measurements relate to rays and physical coordinate frames. Tracking and history can help when the cube moves or becomes hidden.

**Write in your notebook:** Before transfer, identify where every policy input will come from on the real robot and measure its error.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 20.1 Three policy designs

| Design | Inputs | Practical implication |
| --- | --- | --- |
| State-assisted | Simulator-derived object information plus joint data, possibly RGB | Easier to learn; some inputs may be unavailable on hardware |
| Perception-assisted | Camera estimator produces object coordinates, used with joint data | Separates perception errors from control errors |
| Image-based | Camera images and deployable joint/history inputs | Must learn or use visual features sufficient for control |

Your default actor uses RGB plus 28 numerical features, including `cube_ee_relative`. The state-only actor uses those numerical features and ignores RGB. Both still depend on simulator object information. A setting named `vision` does not by itself establish camera-only control.

### 20.2 Camera calibration

Intrinsics describe focal length and image coordinates; extrinsics describe camera position/orientation relative to another frame. Depth or a calibrated geometric model is needed to resolve 3D position reliably from images. A detector returning pixel coordinates does not directly provide world metres.

Your `perception/cube.py` and camera verification/calibration scripts provide places to investigate this interface. Verify detection accuracy, coordinate conversion, timestamps, and uncertainty before substituting estimated positions for simulator truth.

**Example:** A perception estimator is consistently 2 cm wrong sideways, while the grasp-centering reward tolerates only a few centimetres. The control policy may approach the wrong point even though it works with perfect simulated position. Diagnose the estimator's error distribution before changing PPO.

### 20.3 Partial observability and history

Occluded cubes, missing detections, moving objects, sensor delay, and unknown friction can require history. Frame stacking provides a fixed recent window; recurrent networks learn a memory representation. Neither automatically estimates the correct hidden state.

The current policy already observes last actions and filtered targets. It does not automatically receive arbitrary camera history or all critic-only information. Adding a recurrent model changes training/storage requirements and should be treated as a deliberate experiment.

### 20.4 Avoiding visual shortcuts

A policy may learn that a fixed yellow patch at a fixed pixel corresponds to a cube without learning transferable localisation. Test new positions, backgrounds, illumination, and occlusion. Compare state-assisted and image-dependent models only after confirming identical resets and meaningful use of images.

Randomizing visuals too early can increase difficulty before control is learnable. First demonstrate the task with reliable information, then introduce realistic perception noise and image variation in measured stages.

</details>

## 21. Curriculum, domain randomization, and sim-to-real

### Start here: Teach an easier task before a harder one

A **curriculum** is a planned sequence of increasing task difficulty. Start near the cube to learn grasp-and-lift. Then learn to reach from home. Then vary the cube's position and physics. Finally learn placement.

**Domain randomization** changes selected simulation conditions so the policy learns across a range. Examples include cube position, mass, friction, and lighting. Check which are actually enabled in your code.

**Sim-to-real transfer** means using simulation learning on physical hardware. Differences in sensors, motors, timing, and contacts can cause failure.

**Calibration** corrects known mismatches. Randomization trains for plausible remaining variation. Randomizing around a wrongly placed finger collider does not fix that collider.

**Write in your notebook:** Pass an easier stage using measured success, then increase difficulty. Test the specific variations you expect during use.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 21.1 Your actual curriculum

| Stage | Start and variation | What passing demonstrates |
| --- | --- | --- |
| Physical preflight | Controlled commands, validated workcell and grasp | The simulator can execute a real dynamic grasp in the tested setting |
| `grasp` | Verified near-cube pose, open jaws, fixed cube/physics | Policy learns closure and lifting from an easier start |
| `fixed` | Home pose, fixed cube/physics | Policy learns approach and lift from home |
| `lift` | Home pose; randomized cube position/yaw, friction, mass | Policy handles the configured distribution of lift conditions |
| `place` | Full reach, grasp, lift, transport, release, withdrawal | Policy completes the larger task under its evaluation definition |

Curriculum is a sequence of learning problems. It changes the starting-state distribution and sometimes the task objective. A near-cube success does not establish home-pose success; fixed-position success does not establish generalization.

The near-grasp initialization biases initial action means toward the verified pose. It is not a scripted demonstration dataset and is not itself behavioural-cloning training.

### 21.2 What is randomized now?

The current randomized lift configuration includes cube position offsets, cube yaw, startup mass variation, and startup material variation. The easy `grasp` and `fixed` stages disable mass/material variation while keeping explicit cube resets.

Important distinction: position/yaw are reset-time variations; mass and friction are configured as **startup events**, not newly sampled every episode. Parallel environments can therefore provide persistent different physics instances. Do not describe every randomization as per-episode sampling.

Code currently varies cube mass between 0.045 and 0.065 kg. The accompanying percentage comment is approximate; use the actual numerical bounds. Lighting and camera variation described by the robotics excerpt are useful future experiments, not automatically active features in your current events.

### 21.3 Training across parameter variation

An idealized randomized objective is:

**Read equation E48 in words:** Average policy performance across the simulated parameter conditions used for training.

```text
J_DR(θ) = E_(ξ~p(ξ))[J(θ;ξ)]                         [E48]
```

**Calculate it step by step:** If two equally likely physics settings give expected returns ten and six, their average is eight. Changing the parameter distribution changes this objective. A strong average does not guarantee success in every condition.


- `J_DR` is expected performance across domains.
- `θ` is policy parameters.
- `ξ` is a vector of simulation parameters, such as mass, friction, latency, and camera pose.
- `p(ξ)` is the distribution sampling those parameters.
- `~` means sampled from; the outer expectation averages domains.
- `J(θ;ξ)` is policy performance in the fixed domain ξ; the semicolon separates policy parameters from environment parameters.

This optimizes average performance over the specified distribution, not worst-case success across all possible real conditions. Wide unrealistic ranges can make discovery harder or encourage conservative behaviour. Measure real variation, start with relevant bounded ranges, and broaden when evaluation supports it.

### 21.4 System identification and adaptation

System identification estimates physical parameters from measurements. For SO-101, useful measurements include joint zero and direction, actual commanded-to-observed motion, latency, gripper geometry, object mass, and contact behaviour.

A history-based adaptation module can infer features related to unknown dynamics. Teacher–student methods may train with privileged parameters and later infer a useful representation from deployable sensors. This is an additional learning system, not something the current actor–critic automatically performs.

Randomization and calibration serve different purposes. Randomization handles plausible uncertainty; calibration corrects systematic mismatch. Randomizing around a wrongly placed grasp frame does not make that frame correct.

### 21.5 Transfer gaps to test

For this arm, separate observation/perception error, actuator/timing error, joint calibration, contact geometry/friction, and task/reset variation. Introduce one controlled mismatch at a time in simulation and measure its effect. A policy sensitive to 1 cm localization error needs perception work or retraining with that error, not a claim of deployment readiness.

</details>

## 22. Imitation learning, behavioural cloning, and DAgger

### Start here: Learning from a demonstrated grasp

Instead of discovering every action from scratch, a robot can learn from examples. **Imitation learning** uses expert behaviour. **Behavioural cloning** trains a predictor to reproduce recorded expert actions from observations.

**Example:** record the gripper approaching, closing, and lifting successfully. The learner sees which command the expert used at each recorded observation.

The learner can still make mistakes and reach situations absent from the demonstrations. It needs recovery examples too. **DAgger** gathers expert advice for situations the learner itself visits, then adds those labels to the training data.

The observation and action definitions must match how the robot will be controlled. Adjacent frames from one attempt should not leak across training and test data.

**Write in your notebook:** Demonstrations can make useful behaviour easier to discover, but a good imitation loss still needs a closed-loop task test.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 22.1 Behavioural cloning

Behavioural cloning learns from expert observation/action pairs. It needs demonstration quality, synchronised sensing, and action labels with the same interpretation as deployment.

For a deterministic continuous-action predictor, a common loss is:

**Read equation E49 in words:** Measure how different predicted commands are from demonstrated commands.

```text
L_BC(θ) = (1/N) Σ_(i=1)^N ||π_θ(o_i) - a_i^expert||₂² [E49]
```

**Calculate it step by step:** For predicted components 0.2 and negative 0.1, and expert components 0.3 and 0.1, errors are negative 0.1 and negative 0.2. Square and add to get 0.05 for that pair. Average this error across the demonstration samples.


- `L_BC` is supervised imitation loss.
- `θ` is policy weights; `N` is demonstration sample count.
- `i` indexes a recorded observation/action pair.
- `o_i` is the recorded observation.
- `π_θ(o_i)` is predicted action vector in this deterministic-predictor notation.
- `a_i^expert` is the expert's recorded action vector; superscript expert labels its source.
- Subtraction gives prediction error; squared Euclidean norm sums squared component errors.
- Averaging over N produces mean error. A probabilistic predictor can instead use a likelihood-based loss.

For two action components predicted as 0.2 and -0.1 with expert targets 0.3 and 0.1, squared vector error is 0.05. Small supervised loss on expert states does not guarantee successful closed-loop execution.

### 22.2 Distribution shift

An imitation policy's small error changes the next state, which may be absent from its demonstrations. Further errors can accumulate. To learn recovery, include meaningful perturbations and corrective examples, not only perfect nominal trajectories.

Split by entire trajectories or scenarios, not randomly adjacent frames. Nearby frames are strongly correlated; leaking them across train/test splits exaggerates generalization. Record timestamps, camera settings, joint conventions, commanded references, actual motion, failures, and recovery actions.

### 22.3 DAgger

DAgger repeatedly runs a learner, obtains expert labels for states the learner actually visits, aggregates those labels, and retrains. Its benefit is correcting the learner-induced state-distribution mismatch; guarantees depend on assumptions about expert access and learning. It is not simply collecting more successful expert trajectories. See the [DAgger paper](https://arxiv.org/abs/1011.0686).

### 22.4 How demonstrations could help your project

The successful controlled grasp sequence is a potential source of simulation demonstrations after its observation/action sequence is recorded consistently. Initializing policy means to one pose is less informative than teaching a closure-and-lift trajectory with recovery examples.

A possible experiment is to record diverse physically valid grasp attempts, pretrain a compatible actor, evaluate it, and then fine-tune with RL. This is a proposal, not a feature presently enabled by your PPO command. Evaluate whether the initialization actually increases successful experience before adopting it.

</details>

## 23. ACT, diffusion policies, and vision-language-action models

### Start here: Advanced policies: learn the basic idea first

**ACT** predicts a short sequence of actions, called an action chunk. **Diffusion Policy** generates action sequences through repeated refinement. Both can learn from demonstrations.

**Example:** a chunk might describe several commands during closure and lifting. Repeated observations and replanning are still important if the cube moves.

A **vision-language-action model**, or VLA, uses visual and language information to generate robot actions. It may accept an instruction such as “put the cube in the box.” Its actions must still match the robot's actual controls.

These are advanced choices. A larger model cannot fix a cube that does not reset or finger collision shapes that miss the object.

**Write in your notebook:** Better action models can help represent useful behaviours. Correct data, control interfaces, physics, and evaluation remain necessary.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 23.1 Multimodal actions

Two different approach paths may both work. A deterministic squared-error predictor can average conflicting labels into a path that works poorly. A richer action distribution can represent alternatives, but architecture alone does not fix inconsistent labels or missing task information.

### 23.2 ACT

ACT predicts action sequences from observations using a transformer and a conditional variational model. It supports temporal ensembling of overlapping predictions rather than requiring every full chunk to execute without feedback. Chunk prediction can improve temporal coherence and reduce compounding error, with task-specific latency and responsiveness tradeoffs. See the [ACT paper](https://arxiv.org/abs/2304.13705).

For SO-101, longer chunks could describe approach, closure, and lift. A sudden object displacement during a chunk still needs timely replanning. Chunks do not create a valid grasp if motor targets, calibration, or contacts are wrong.

### 23.3 Diffusion Policy

Diffusion Policy learns conditional action-sequence distributions using a denoising process and can represent multiple action modes. At inference it generates an action sequence conditioned on observations, typically in a receding-horizon control loop. Denoising compute adds latency. It is trained on demonstrations in its standard form, rather than by environmental reward alone. See [Diffusion Policy](https://arxiv.org/abs/2303.04137).

The number of denoising steps, network type, action horizon, and executed prefix are configuration choices. The excerpt's timing and sample-count examples are not guaranteed budgets for your GPU or task.

### 23.4 Vision-language-action models

VLAs connect images and language with robot action generation. OpenVLA uses a vision-language backbone with action-token prediction; pi0 uses flow matching for continuous action generation. They are different architectures, and pi0 should not be described as literally the original ACT plus DDPM diffusion. See the [OpenVLA paper](https://arxiv.org/abs/2406.09246) and [pi0 paper](https://arxiv.org/abs/2410.24164).

A large pretrained model does not remove embodiment adaptation, action normalization, timing, data quality, or evaluation requirements. LoRA updates a restricted set of added low-rank parameters and can reduce trainable-parameter cost; it does not eliminate all inference memory requirements or guarantee retention of old skills.

For your present single-cube lifting task, the immediate learning bottleneck is establishing meaningful successful grasp-and-lift experience and valid evaluation. A language model is an advanced option for broader task conditioning, not an evidenced cure for the recorded zero-success baseline.

### 23.5 Locomotion and industrial robotics connections

The excerpt also applies RL to quadrupeds, humanoids, and industrial manipulation. The transferable ideas are tracking rewards, contact transitions, actuator dynamics, curricula, disturbances, and independent qualification.

Locomotion often rewards command tracking while penalizing excessive effort and unwanted motion. Balancing is dynamic: the static support-polygon rule is not a complete account of fast gaits or flight phases. Whole-body control couples arm motion with balance and leg contact forces.

Industrial insertion emphasizes force/torque sensing, compliance, and precise frame calibration. A force sensor measures a wrench containing force and moment components, which must be expressed in the appropriate frame. These topics are useful future extensions; your fixed-base arm should remain mounted rather than learning balance through a drifting base.

Multi-agent robot learning concerns multiple decision makers and possibly shared rewards. Parallel copies of your single-robot environment are multiple data collectors, not automatically a multi-agent coordination problem.

</details>

## 24. Evaluation, diagnostics, and deployment

### Start here: Test whether the robot really completes the task

An **evaluation episode** is one complete test attempt. Count success using a precise definition and include failed attempts in the total.

**Example:** 45 successful lifts out of 50 attempts give 90% success. If the robot dragged the cube excessively, your project's gate can still fail.

Reward, critic loss, entropy, and success measure different things. A critic can accurately predict a poor policy's score. A policy can collect approach rewards without lifting. Videos help reveal how a score was earned.

A **held-out condition** is a test condition that was not used to tune the policy. Use such conditions to check whether learning extends beyond familiar starts.

**Write in your notebook:** Training numbers explain parts of the learning process. Completed task outcomes show what the robot has learned to do.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 24.1 Success rate

**Read equation E50 in words:** Divide completed successful attempts by all completed test attempts.

```text
p_hat = N_success / N_episodes                        [E50]
```

**Calculate it step by step:** Count 45 successful attempts out of 50. Divide 45 by 50 to get 0.9. Multiply by 100 when expressing it as 90 percent. Include failures in the denominator.


- `p_hat` is measured episode success fraction; hat indicates an estimate.
- `N_success` counts completed episodes satisfying the success definition.
- `N_episodes` counts all evaluated complete episodes, including failures.
- Division gives a fraction from zero to one; multiply by 100 only when expressing a percentage.

45 successes among 50 episodes give 0.9, or 90%. This is different from average reward and a manager's mean per-step termination activation.

Your pipeline requires 50 completed deterministic episodes for each seed 101, 202, and 303. Each seed must reach at least 90% success and mean tabletop slide no greater than 1 cm. Do not replace this gate with “the training finished.” Fixed deterministic conditions across different seeds can produce nearly identical trajectories; seed count alone does not establish diverse tested conditions.

### 24.2 Finite sampling

For approximately independent Bernoulli episode outcomes, a rough standard error is:

**Read equation E51 in words:** Estimate sampling uncertainty for a success fraction under the stated assumptions.

```text
SE ≈ sqrt(p_hat(1-p_hat)/N)                           [E51]
```

**Calculate it step by step:** For success fraction 0.9, multiply by failure fraction 0.1 to get 0.09. Divide by 50 attempts to get 0.0018. Take the square root to get about 0.0424, or 4.24 percentage points. This shortcut is unreliable at zero or perfect observed success.


- `SE` is standard error of the estimated success fraction.
- `p_hat` is observed success fraction.
- `1-p_hat` is observed failure fraction.
- `N` is episode count; multiplication and division form the variance estimate.
- `sqrt` converts variance into standard deviation.
- The approximation assumes sufficiently suitable sampling; correlated episodes weaken the interpretation.

At observed success 0.9 over 50 episodes, the approximate standard error is 0.0424, about 4.24 percentage points. This approximation is poor at zero or perfect observed success: a computed zero standard error does not prove zero uncertainty. An operational pass threshold is also different from a statistical confidence guarantee.

### 24.3 Critic diagnostics

Explained variance is often computed as:

**Read equation E52 in words:** Compare the spread of prediction errors with the spread of targets.

```text
EV = 1 - Var(Y - V) / Var(Y)                         [E52]
```

**Calculate it step by step:** If target variance is four and error variance is one, divide one by four to get 0.25. Subtract from one to get explained variance 0.75. Variance means spread around the average; this metric does not prove task success.


- `EV` is explained variance for a set of predictions and targets.
- `Y` is the vector of return targets, and `V` the corresponding value predictions.
- `Y-V` is their prediction-error vector.
- `Var` is sample variance under the chosen implementation convention.
- Dividing residual variance by target variance compares unexplained spread with total target spread.
- Subtracting from one makes values near one indicate strong tracking of variation.
- This requires nonzero target variance; your implementation reports zero when targets are essentially constant.

Target variance 4 and error variance 1 give EV 0.75. It measures fitting to targets, not success. A constant prediction offset can leave this variance metric deceptively high, so examine mean predictions, targets, and squared loss too.

| Diagnostic | What it means | What it cannot establish alone |
| --- | --- | --- |
| Mean episode reward | Shaping and penalties accumulated per episode | Physical grasp success |
| Grasp/lift reward terms | Those geometric reward features activated | Valid finger contact and sustained holding |
| Termination activation | Manager-specific success-term activity | Independent completed-episode success rate |
| Value mean | Predicted discounted return | Success probability |
| Return target mean | Bootstrapped critic training targets | Observed full-episode return in every sample |
| Absolute TD error | Average one-step prediction mismatch | Policy mastery |
| Value loss | How well critic fits targets | Good target semantics or a good policy |
| Entropy/noise standard deviations | Raw action-distribution spread | Calibrated scene uncertainty or useful motor exploration |
| Action saturation | How often raw actions reach/exceed action limits | Whether target changes execute physically |
| Success plus videos/contact inspection | Task outcome plus physical behaviour | Real-world transfer without further testing |

### 24.4 Evaluation protocol

Use the exact candidate checkpoint, matching actor variant, action mapping, observations, normalization, curriculum, and reward/success configuration. Save checkpoint hashes and relevant configuration/asset hashes. Changing definitions between evaluation runs changes what success means.

Measure complete episodes and preserve terminal observations before automatic reset. In your evaluator, this is the purpose of `final_obs`. Otherwise peak elevation and cube sliding can accidentally be measured from the next episode's reset state.

For generalization, explicitly vary starts, positions, physical parameters, lighting, perception error, and disturbances. Hold out meaningful scenarios. Report both nominal success and failure categories. More training seeds describe training variability; more evaluation seeds describe sampled test variability; they are different experiments.

### 24.5 Deployment and monitoring

Before transfer, verify deployable observations, measured joint calibration, command limits, camera transforms, latency, collision coverage, and stopping/recovery behaviour. Your project's simulation-zero assumptions and incomplete full-arm collision coverage remain separate work. A reward penalty or a passing simulation gate does not certify hardware operation.

Monitor outcomes, timing, motor state, perception failures, and input-distribution changes. Retraining on new conditions should also reevaluate old capabilities to detect forgetting. Preserve the earlier model and regression scenarios so an apparent improvement does not hide a lost skill.

Attention visualizations or raw action entropy are not automatically calibrated confidence. Your state-independent Gaussian noise cannot identify an unfamiliar object simply by increasing entropy for that image. An emergency stop's effect depends on the actual controller and hardware; do not assume it always preserves the current grasp or pose.

</details>

## 25. What your saved project evidence establishes

### Start here: What the saved tests tell us

A physical grasp test and a learned-policy test answer different questions.

The recorded controlled grasp succeeded. This showed that the tested simulator setup could hold and lift the cube when given suitable commands. A physics stress test also passed under its recorded conditions.

The recorded CNN and state-only policies each had zero successful lifts in 150 evaluation attempts. These policies did not master the tested task. Some relevant files have changed since those records were created, so these are historical results.

**Why this matters:** if physics can grasp but the policy cannot, investigate whether the policy discovers and chooses useful commands. If even controlled commands fail, investigate the environment first.

**Write in your notebook:** Physical possibility, numerical stability, policy learning, and real-robot transfer are separate questions.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

These results were read from files on 8 October 2026. They are saved results, not new experiments performed for these notes. Some fingerprints no longer match current files, so they do not certify the exact current checkout.

| Saved evidence | Recorded result | Interpretation |
| --- | --- | --- |
| `outputs/pipeline/before_pad_calibration.json` | Physical grasp failed; cube remained around 0.425 m; tool prediction discrepancy around 5.2 mm | There was an environment/geometry problem before any claim about policy learning |
| `outputs/pipeline/physical_grasp.json` | Controlled physical grasp passed; minimum hold height about 0.47243 m; final gripper angle about 0.97308 rad | A dynamic grasp was possible with the recorded asset/configuration and tested trajectory |
| `outputs/pipeline/random_before_stability.json` | Bounded random-action reproduction failed at step 4315 with invalid robot joint/link state | Failure occurred without requiring a learned policy update; articulation dynamics were implicated |
| `outputs/pipeline/stability_after_final.json` | 12,000 steps across 64 environments, 768,000 transitions, passed with cached RGB | Stronger evidence for tested physics stability; not a full-rate camera/PPO test or success evaluation |
| `outputs/pipeline/reset_check.json` | Five reset trials across four environments passed, including partial resets | Corrected pose/velocity reset behaviour passed those recorded checks |
| `outputs/pipeline/vision_baseline/grasp_gate.json` | 0 successful episodes across three sets of 50 | Recorded CNN candidate failed its grasp gate; baseline also had a documented fixed-reset confound |
| `outputs/pipeline/grasp_state_gate.json` | 0 successful episodes across three sets of 50 | Recorded state-only PPO candidate also failed; removing RGB did not establish mastery |

The state-only evaluation's mean peak elevation was about 0.0336 cm per seed, while all episodes lasted the full 150 steps. That describes nearly no average lifting in those deterministic tests. Mean tabletop sliding was small, but low sliding by itself is not success.

During this review, the state gate's recorded fingerprint differs from current `utils/grasp_geometry.py`. The older CNN gate differs from several configuration, agent, pipeline, evaluator, and geometry files. Preserve both as historical results and reevaluate a compatible candidate under the current configuration when studying its behaviour.

**What the combined evidence says:** tested physical feasibility and numerical stability are distinct from learned policy success. Both recorded learning candidates failed. The files do not isolate one confirmed cause of the learning failure or establish that CNNs, PPO, or state-only control are inherently unsuitable.

</details>

## 26. Training hurdles: what, why, how, and verification

### Start here: Find the failing layer before choosing a fix

Use four questions for every hurdle:

1. **What happened?** Describe visible behaviour and measurements.
2. **Why could it happen?** List plausible causes without claiming they are confirmed.
3. **How do I check?** Run a small experiment that separates the causes.
4. **What proves improvement?** Repeat the test and measure the intended behaviour.

**Example:** the robot closes but the cube stays on the table. First check whether the cube is between the fingers. Then check target and actual angle, collision shapes, and whether a controlled grasp works. Change rewards only after checking those layers.

The detailed cases below cover hovering, pushing, idle policies, geometry, a drifting base, NaNs, resets, failed architecture comparisons, checkpoints, GPU issues, and transfer.

**Write in your notebook:** Choose the smallest test that can explain the failure. Change one important factor at a time.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### Hurdle 1: robot approaches or hovers but does not grasp

**Related theory:** rewards, return, shaping, credit assignment.

**What:** the robot can collect approach feedback without advancing to closure and lifting. The project guide documents the earlier repeated proximity reward and its removal.

**Why:** a reward paid for a state occupied repeatedly can make waiting profitable. Even after removing this term, the actor may discover only reaching while grasp feedback remains too narrow or difficult to obtain.

**How to diagnose:** compare actual term contributions for stationary hovering, a physically successful grasp, and a moving failed attempt. Inspect whether `grasp_cube` ever activates and whether gripper targets execute. Verify alignment tolerances in the actual grasp cavity, not just distance to an arbitrary frame.

**Solution:** retain feedback for new progress rather than repeated hovering; use the easy near-grasp stage to make closure/lift discoverable. If success remains absent, investigate demonstrations or easier valid initial configurations as a controlled experiment.

**Why this helps:** it removes one local reward loophole while increasing the chance of observing outcomes the critic and actor can learn from.

**Verification:** evaluate completed held lifts and inspect fingers. More approach reward does not verify the solution.

### Hurdle 2: closing too early pushes the cube

**Related theory:** action semantics, reward gating, contact dynamics.

**What:** closed fingers strike the side of the cube during approach.

**Why:** neutral-action mapping may close the gripper; poorly conditioned exploration may close far from the cavity; geometric alignment signals can be incorrect. Simply commanding closure is not a grasp.

**How to diagnose:** plot gripper action, processed target, actual angle, lateral/vertical offset, and cube displacement along the same timeline. Check open, half-close, and closed commands with a controlled probe.

**Current correction:** zero gripper action requests 1.20 rad open; negative actions progressively close. Premature closure is penalized when centering conditions are false; grasp feedback rewards closure near the cube.

**Why this helps:** early neutral behaviour leaves space to envelop the cube, and feedback distinguishes appropriate closure from pushing.

**Verification:** show repeatable physical enclosure and sustained lifting. If fingers close in empty space, inspect tool/pad geometry before altering the penalty weight.

### Hurdle 3: cube slides toward the box instead of being lifted

**Related theory:** reward misspecification, phase-conditioned tasks.

**What:** an agent can satisfy position-related feedback by pushing rather than lifting.

**Why:** distance-to-box or instantaneous proximity rewards may ignore object support and task history. A velocity-only dragging cost can be weak for very slow sliding.

**How to diagnose:** inspect cube elevation, horizontal speed, displacement from reset position, prior-lift flag, and each placement reward term.

**Current correction:** lift feedback is gated by a geometric grasp proxy; the task penalizes both tabletop speed and cumulative displacement. Placement success requires a prior sustained lift, then release, tool withdrawal, full rotated-cube containment, and settling.

**Why this helps:** the desired sequence becomes part of the completion definition, while the displacement penalty catches slow pushing.

**Verification:** verify that pushing a cube into the box fails the completion detector and a genuine lift–place–release trajectory passes it. Check video because a geometric lift proxy can still be imperfect.

### Hurdle 4: idle policy avoids penalties and receives almost no success feedback

**Related theory:** exploration, local optima, sparse success, reward balance.

**What:** the saved deterministic candidates have zero success and predominantly near-stationary cube outcomes.

**Why:** plausible explanations include penalties making attempted movement unattractive, inadequate useful exploration, weak geometric feedback, action means drifting to ineffective regions, and failure to sample successful sequences. The evaluation summaries alone cannot select one cause.

**How to diagnose:** inspect actor mean versus sampled actions, saturation, processed targets, actual joint motion, per-term rewards, and the number of successful training transitions. Run playback from the intended curriculum and compare a known successful controlled trajectory's reward with idle and failed trajectories.

**Proposed solutions:** first verify resets and physical feasibility; then test one change at a time, such as a compatible demonstration initialization, a carefully chosen easier start, or a measured adjustment to reward balance/exploration. Keep a baseline and equal transition budgets.

**Why this helps:** PPO needs informative experience. A large terminal bonus unseen in the data does not by itself reveal how to earn it.

**Verification:** increase completed evaluation success, not just exploration entropy or reduced critic loss.

### Hurdle 5: visually plausible gripper cannot physically lift

**Related theory:** transition-model validity, contacts, state representation.

**What:** the earlier physical grasp report failed even with controlled commands.

**Why:** a finger collider, silicone pad, or grasp frame can be misplaced. The policy's reward coordinates then disagree with real contacts. Force limits and object size can also prevent enclosure.

**How to diagnose:** inspect collision shapes, verify forward kinematics against simulator poses, perform a controlled open–approach–close–raise sequence, and record cube height and measured angle.

**Recorded correction:** the derived asset realigned finger-pad geometry and the tool frame, adjusted the gripper limits/torque, and subsequently passed a dynamic grasp test. The recorded held angle was around 0.973 rad.

**Why this helps:** it makes the intended transition physically available and aligns reward geometry with it.

**Verification:** the unassisted cube remains held under gravity for the specified duration. A cube attachment, teleport, or a static picture is not equivalent evidence.

### Hurdle 6: success detector rejects a real grasp or accepts a false one

**Related theory:** task specification, measurement validity, temporal state.

**What:** an old closing-angle threshold could reject a physically held 5 cm cube; an instantaneous height check could accept a bounced or falling cube.

**Why:** the object prevents full closure, and brief elevation does not imply holding. A tool-center distance alone does not measure finger contact.

**How to diagnose:** evaluate known positive and negative trajectories: stable held cube, near-but-unheld cube, bounce, falling cube, and release. Inspect angle, offset, speed, and hold counter.

**Current correction:** held-angle threshold is below 1.10 rad, with height, distance, speed, and sustained-time conditions. It remains a geometric proxy.

**Why this helps:** tolerances reflect recorded grasp geometry, and the duration rule rejects many transients.

**Verification:** report detector behaviour on deliberate false positives/negatives and inspect physical contacts. Changing a detector merely to label a failing policy successful would invalidate evaluation.

### Hurdle 7: robot base drifts instead of being mounted

**Related theory:** environment dynamics and control authority.

**What:** the project documentation records an imported free base.

**Why:** unanchored dynamics let motor/contact forces move the entire robot, creating a different task from a tabletop-mounted arm.

**How to diagnose:** inspect root configuration and track base pose under gravity and controlled/random joint actions.

**Current correction:** the derived robot asset is used with `fix_root_link=True`.

**Why this helps:** it establishes the mounting constraint the real task assumes. The policy then learns joint motion about a stable base rather than exploiting base movement.

**Verification:** base pose remains anchored during relevant physical probes, and collision geometry stays aligned with the tabletop.

### Hurdle 8: training fails with NaN joint states or policy inputs

**Related theory:** numerical conditioning, valid data, distinguishing physics from optimization.

**What:** a previous training run stopped after invalid articulation state. Bounded random actions reproduced the issue at step 4315.

**Why:** the reproduction implicates articulation dynamics independently of PPO learning. Invalid inertias/masses, aggressive target changes, contacts, and insufficient solver settings were relevant candidates.

**How to diagnose:** identify the earliest nonfinite field. Inspect robot angles/velocities/link poses, cube state, processed targets, rewards, values, and gradients. Preserve the exact failing configuration, seed, and report. Do not start by assuming GAE is wrong.

**Recorded corrections:** positive tool-frame mass/inertia, bounded target slew, motor armature, stronger solver iteration settings, and consistent reset/filter state. The subsequent cached-camera stress test passed 768,000 transitions.

**Why this helps:** physically valid parameters and bounded reference changes improve the numerical problem presented to the solver.

**Verification:** repeat the bounded-action reproduction beyond the earlier failure, then separately verify full-rate rendering and PPO. Passing a stress test does not guarantee all future contacts are stable.

**Avoid:** replacing NaNs with zeros and counting those trajectories as valid. That hides the failing physical state and can corrupt learning.

### Hurdle 9: fixed cube conditions were not actually fixed

**Related theory:** initial-state distributions and experimental confounding.

**What:** disabling randomization previously removed the cube pose reset event; contact could leave the cube displaced in later episodes.

**Why:** clearing object buffers is different from writing its starting pose and zero velocity. A fixed curriculum needs a deterministic reset, not no reset.

**How to diagnose:** deliberately displace, rotate, and accelerate cubes, then test full and partial resets repeatedly.

**Current correction:** `reset_cube_pose` is always active. Easy stages use empty variation ranges to restore the default state; randomized stages use offsets through the same event.

**Why this helps:** it restores the intended initial-state distribution and makes comparisons interpretable.

**Verification:** position/orientation/velocity match reset specifications, while unselected environments stay unchanged. The recorded five-trial reset test passed. A running process started before the correction cannot automatically adopt changed Python code.

### Hurdle 10: changing CNN to MLP does not solve learning

**Related theory:** representation, ablation, causal inference.

**What:** both recorded CNN and state-only variants failed their success gates.

**Why:** failures may be shared across reward design, exploration, action mapping, and starting conditions. The old CNN baseline's reset confound prevents attributing its failure specifically to image processing.

**How to diagnose:** confirm both variants have correct resets, comparable budgets, identical relevant physics/rewards, and suitable initialization. Verify the numerical state inputs before evaluating the additional difficulty of vision.

**Solution:** run a controlled ablation. First establish a strong deployability-aware state-assisted baseline. Add images or replace ground-truth cues with estimates while measuring how success changes.

**Why this helps:** it isolates what an architectural/input change actually contributes.

**Verification:** report success and failures under comparable conditions. Neither the existence of a checkpoint nor a lower loss proves an architecture superior.

### Hurdle 11: critic loss becomes small but lifting stays at zero

**Related theory:** policy evaluation versus control, TD fixed points.

**What:** a critic can learn that unsuccessful behaviour earns a small nearly constant return.

**Why:** the target reflects data from the current policy. Accurately estimating a poor policy's return does not make that policy good. Constant targets also limit interpretation of explained variance.

**How to diagnose:** compare value/return means, target variance, TD residuals, actor changes, grasp/lift signals, and completed success.

**Solution:** look for useful experience and task incentives rather than driving value loss lower in isolation. Verify target masks and calculations, then investigate the actor's discovery and execution of successful behaviour.

**Why this helps:** it focuses effort on control improvement after prediction consistency is established.

**Verification:** success increases in completed held-lift evaluations. Lower critic error remains a supporting diagnostic.

### Hurdle 12: training looks random but physical exploration is weak

**Related theory:** stochastic policies, clipping, actuator dynamics.

**What:** raw actions have substantial noise, yet many physical targets saturate or move little.

**Why:** clipping collapses different samples to the same limit. Filtering and slew limits reduce step-to-step reference changes. Positive gripper actions all request open jaws. Noise alone does not imply coverage of meaningful grasp trajectories.

**How to diagnose:** compare raw samples, clipped commands, filtered targets, actual joint states, gripper closure frequencies, and successful transitions. Check all six action noise scales.

**Proposed solution:** tune exploration against observed physical movement and successful experience. Preserve needed physical bounds; investigate initialization or structured experience rather than removing stability limits indiscriminately.

**Why this helps:** the learner must explore distinct physical outcomes, not only distinct numbers before the action interface.

**Verification:** increased useful state/trajectory coverage and task success, with finite stable dynamics.

### Hurdle 13: success near the cube fails from home or with randomization

**Related theory:** curriculum, distribution shift, generalization.

**What:** a near-grasp policy may fail during approach, and a fixed-position policy may fail on moved cubes.

**Why:** each stage evaluates a different initial-condition distribution. Skills transfer only to the extent the representation and policy handle that difference.

**How to diagnose:** evaluate near-cube, home/fixed, and randomized conditions separately. Classify failure as approach, closure, lift, hold, or perception.

**Solution:** progress through the repository's evidence-gated `grasp → fixed → lift → place` sequence. Expand variation gradually and advance using the exact passing checkpoint.

**Why this helps:** later difficulty builds on measured competence rather than simultaneous discovery of every subskill.

**Verification:** each stage passes its own evaluation gate. The earliest easy stages test repeatability, not broad generalization.

### Hurdle 14: old checkpoints or gate files no longer fit the project

**Related theory:** reproducibility and distribution/interface consistency.

**What:** observation dimensions, CNN/MLP inputs, action mapping, geometry, or success criteria change while a saved model/gate remains.

**Why:** a policy learns an interface and task distribution. Even matching tensor dimensions do not establish semantic compatibility.

**How to diagnose:** compare checkpoint variant metadata, observation/action specification, normalization, configuration hashes, assets, and evaluation definitions. The saved fingerprints checked in section 25 differ from some current files.

**Solution:** use compatible checkpoints, preserve historical artifacts, and start fresh when the project's compatibility checks require it. Reevaluate after changes that affect evidence. Do not bypass rejected load checks merely to obtain playback.

**Why this helps:** it prevents mixing results from different tasks or command semantics.

**Verification:** the candidate loads through the intended interface and passes evaluation under explicitly recorded current conditions.

### Hurdle 15: simulation camera/GUI fails or GPU memory runs out

**Related theory:** infrastructure limits; this is not itself a Bellman-equation problem.

**What:** simulator startup or rendering fails before useful learning.

**Why:** possible causes include GPU memory contention, too many rendered environments, unavailable driver/device paths, or incompatible dependencies. A CUDA error alone is insufficient to identify which.

**How to diagnose:** read the first meaningful error, check GPU health with `nvidia-smi`, inspect resource use, and distinguish memory shortage from a driver problem. Compare a small headless startup with the intended camera-enabled configuration.

**Solution:** reduce environment/rendering load where appropriate, use the configured interpreter, and resolve resource contention or installation problems based on evidence. Do not kill another active training process without knowing its purpose and authorization.

**Why this helps:** it establishes a functioning data-collection system before tuning learning parameters.

**Verification:** successful current startup and short update run with the intended sensors. These checks still do not demonstrate a learned grasp.

### Hurdle 16: simulated success does not transfer to the real arm

**Related theory:** partial observation, calibration, dynamics mismatch, generalization.

**What:** simulation succeeds with perfect cube coordinates, but physical operation lacks those coordinates or has different timing and contacts.

**Why:** simulator truth is privileged; camera noise and systematic calibration errors change observations; real motor response changes transitions.

**How to diagnose:** replace one ideal input with a measured realistic estimate in simulation, perturb latency and calibration deliberately, and evaluate. Compare commanded and actual physical motion before enabling the whole task.

**Solution:** build and validate perception, measured joint/frame calibration, compatible preprocessing, realistic action/timing models, and bounded supervised hardware tests. Retrain or adapt to the resulting observation/dynamics distribution.

**Why this helps:** it removes concrete mismatches instead of treating real hardware as another perfect simulator environment.

**Verification:** separately report perception error, controller tracking, and supervised task success within the tested conditions. A state-assisted simulation checkpoint alone is insufficient.

</details>

## 27. Worked exercises with answers

### Start here: Practise until you can explain the answer

Use the exercises to practise reasoning, not just substituting numbers. Before calculating, say what the answer should mean.

For a return exercise, identify the reward sequence and how late each reward arrives. For a TD exercise, identify the reward, current prediction, next prediction, and whether the episode truly ends.

**Example:** if the task ends now, there is no future score within that attempt. The next episode's reset state must not contribute future value to the completed attempt.

For a practical exercise, predict the expected behaviour before running it. Compare your prediction with the result, then explain the difference.

**Write in your notebook:** A useful answer includes a number, its meaning, and the assumptions used to calculate it.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### Exercise 1: return changes with discount

Compare sequence A with rewards 0,0,10 and sequence B with rewards 2,2,2. Both end after three actions. Use E9.

| Discount | A return | B return | Larger return |
| ---: | ---: | ---: | --- |
| 0.5 | 2.5 | 3.5 | B |
| 0.9 | 8.1 | 5.42 | A |
| 0.99 | 9.801 | 5.9402 | A |

**Reason:** A concentrates its reward later. A stronger future discount reduces its relative appeal. This does not imply that the task should always use the largest possible discount; horizon and estimation behaviour matter.

### Exercise 2: one real reward-manager step

Suppose raw grasp score is 0.5 and raw dragging cost 0.1. Use actual weights 4 and -5 and actual policy interval 0.04 s. Before framework scaling the combined contribution is 1.5; afterward it is 0.06. A full task step includes additional terms too.

**Reason:** weights act on raw features, and the manager then applies time scaling. Log actual contributions rather than reasoning from raw weights alone.

### Exercise 3: terminal masking

Reward is 2, gamma 0.9, current value 4, and next-state estimate 5. Continuing TD residual is 2.5. True terminal residual is -2.

**Reason:** there is no future task return after a true terminal state. An automatic reset's observation belongs to another episode.

### Exercise 4: GAE calculation

Residuals are 1,2,-1; gamma is 0.99 and lambda 0.95. The delay weight is 0.9405. The first advantage estimate is about 1.99646. If old state value was 4, the constructed return target is 5.99646.

**Reason:** the second residual receives one delay factor and the third receives its square. These residuals belong to one episode segment.

### Exercise 5: PPO sign and clipping

With clip width 0.2, positive advantage 2 and ratio 1.5 give objective contribution 2.4. Negative advantage -2 and ratio 0.5 give -1.6.

**Reason:** the objective selects the minimum of two advantage-weighted branches. Taking the minimum means the negative-advantage case cannot be interpreted using only the positive case's intuition.

### Exercise 6: success threshold

45 successful episodes out of 50 give 90%, meeting the pipeline's success fraction threshold for that seed. If average tabletop slide is 1.2 cm, the gate still fails. If another required seed achieves only 88%, the overall gate fails.

**Reason:** the repository requires all configured conditions for each seed, not a favourable pooled average.

### Exercise 7: fair transition budgets

64 environments × 32 steps × 2000 iterations give 4,096,000 transitions. 32 environments at the same rollout length and iterations give half that experience.

**Reason:** faster wall-clock training or the same iteration count does not imply equal physical experience.

### Exercise 8: physical gripper versus command

The actor requests fully closed, but a cube prevents the measured gripper from closing beyond about 0.97 rad. Is this necessarily failure? No: physically held object size changes the attainable joint angle.

**Reason:** success thresholds must reflect actual grasp geometry, not an assumed empty-gripper closing angle.

### Exercise 9: reset versus randomization

You want a fixed cube position. Should its reset event be disabled? No. Restore deterministic position, orientation, and velocity with zero variation.

**Reason:** randomization selects the next start; reset establishes that start. Removing the latter lets previous episodes alter the next attempt.

### Exercise 10: identify the failing layer

Bounded random actions generate NaN robot links before any learning update. Which layer should be investigated first? Articulation/physics and action execution.

**Reason:** reproducing without optimization weakens the explanation that PPO gradients caused the original invalid state. It does not prove a unique physical cause, but guides the next test.

### Exercise 11: camera-only claim

An actor receives images and exact simulator cube-relative coordinates. Is it a camera-only policy? No. Test after replacing/removing privileged inputs and training for the resulting available information.

**Reason:** the numerical cue may carry the location information the policy would otherwise need to learn from images.

### Exercise 12: rewards and success

A critic loss is nearly zero and task success is 0%. Is this contradictory? No. The critic may accurately estimate the unsuccessful policy's return.

**Reason:** prediction quality and control quality are related but distinct.

### Small executable arithmetic lab

This code runs without simulator packages. It verifies arithmetic, not learned policy behaviour. The policy/environment examples above remain the practical tests of understanding.

```python
def discounted_return(rewards, gamma):
    # Accumulate backward using immediate reward plus discounted remaining return.
    result = 0.0
    for reward in reversed(rewards):
        result = reward + gamma * result
    return result

def td_residual(reward, current_value, next_value, gamma, terminal):
    continuing = 0.0 if terminal else 1.0
    return reward + gamma * continuing * next_value - current_value

def gae_for_one_episode_segment(residuals, gamma, lam):
    advantages = [0.0] * len(residuals)
    running = 0.0
    for index in reversed(range(len(residuals))):
        running = residuals[index] + gamma * lam * running
        advantages[index] = running
    return advantages

for gamma in (0.5, 0.9, 0.99):
    print(gamma, discounted_return([0, 0, 10], gamma),
          discounted_return([2, 2, 2], gamma))

print("Continuing TD:", td_residual(2, 4, 5, 0.9, False))
print("Terminal TD:", td_residual(2, 4, 5, 0.9, True))
print("GAE:", gae_for_one_episode_segment([1, 2, -1], 0.99, 0.95))
```

Variable guide: `rewards` and `residuals` are lists in chronological order. `gamma` is discount; `lam` is lambda. `result` stores return from the currently processed suffix. `current_value` and `next_value` are critic predictions; `terminal` indicates no continuation. `continuing` is the terminal mask. `running` stores the decayed residual sum; `index` identifies a list position. `reversed` visits the last item first. The recurrence in each helper implements E10, E24, or E29 respectively. The GAE helper assumes a single episode segment; a full training implementation must handle episode boundaries and appropriate final bootstraps.

</details>

## 28. Study sequence and experiment notebook

### Start here: How to write useful notes and run useful experiments

For each new idea, write five short items: meaning, why it is needed, a cube-lifting example, the equation in words, and one common mistake.

Spend the first reading pass on the simple explanations. Open the detailed notes for the topic you are currently studying. Return to a harder formula after working through a small example.

For an experiment, write the question before changing anything. Record the checkpoint, starting conditions, one deliberate change, training experience, test conditions, and completed success.

**Example question:** does the policy close the gripper when the cube is centered? A command/angle/offset timeline gives more useful evidence than watching only total reward.

**Write in your notebook:** Good notes help you explain an idea. Good experiments help you test an explanation.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 28.1 Where to continue your handwritten notes

Since you have already written about agent, environment, state, and actions, write these next in order:

1. Reward definition and transition timing.
2. Sparse, dense, terminal, intrinsic, and delayed rewards.
3. Weighted rewards and actual manager scaling.
4. Reward loopholes, progress shaping, and potential shaping.
5. Reward versus return, gamma, and physical time.
6. Policy, state value, action value, and advantage.
7. Bellman expectation and optimality, including one numerical example.
8. Monte Carlo versus TD and terminal masking.
9. Multi-step targets, traces, and GAE.
10. Function approximation, actor–critic, and PPO.
11. Robotics control, reset distributions, evaluation, and your hurdles.

For each displayed equation, copy its symbol guide first. Explain why an operator is present: why there is a sum, a discount power, a terminal mask, a gradient, or a minimum. Memorizing a formula without these explanations will make debugging harder.

### 28.2 Two-hour session structure

Use roughly 35 minutes reading/deriving, 60 minutes implementing or analysing one experiment, 15 minutes examining outcomes, and 10 minutes writing conclusions. Training can run between sessions; active study time should include prediction and interpretation rather than watching progress bars alone.

Work through the Sutton–Barto theoretical chapters while using the matching sections here. Small bandit/gridworld examples are useful because you can calculate what should happen before moving to a neural robot policy.

### 28.3 Experiment record template

```text
Question:
Hypothesis and why it is plausible:
Exact checkpoint and actor variant:
Curriculum and initial-state distribution:
Code/configuration/asset versions:
One deliberate change:
Training transition budget and seeds:
Evaluation conditions and complete episode count:
Expected outcome before running:
Observed reward components and task success:
Failure categories and representative video:
Alternative explanations:
Conclusion supported by the evidence:
Next experiment:
```

Examples of useful questions: Does deterministic gripper closure activate when aligned? Is the successful physical sequence actually rewarded more than hovering? Does a cube reset truly remove residual velocity? Does the camera estimator's error fit the grasp tolerance? Does a change increase success across the same test distribution?

### 28.4 Commands from your project

These are inspected command interfaces, provided for future practical sessions; they were not run during preparation of this guide. Full training runs consume substantial compute and should be deliberate experiments.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl

# Physical and numerical preflight, then inspect recorded stage status.
python scripts/train_pipeline.py --stage preflight
python scripts/train_pipeline.py --stage status

# Small current-interface startup/update check from the easier curriculum.
python scripts/train.py --task lift --curriculum grasp --policy state --headless --envs 4 --iterations 2 --device cuda:0

# Inspect a compatible candidate with matching curriculum and actor variant.
python scripts/play.py --task lift --curriculum grasp --policy state --checkpoint EXACT_STATE_CHECKPOINT --num_envs 1

# Direct evaluation of 50 complete episodes, writing a separate report.
python scripts/eval_pipeline.py --task lift --curriculum grasp --policy state --checkpoint EXACT_STATE_CHECKPOINT --episodes 50 --seed 101 --headless --output outputs/study_grasp_state_eval_101.json

# The pipeline's three-seed evaluation and mastery gate.
python scripts/train_pipeline.py --stage evaluate --evaluate-stage grasp --policy state --checkpoint EXACT_STATE_CHECKPOINT
```

`EXACT_STATE_CHECKPOINT` is a placeholder for an existing compatible checkpoint path. The direct evaluation report is distinct from pipeline evidence. The pipeline evaluation can update its gate/reports; use it when you deliberately want to assess that candidate through the official gate.

For CNN experiments, use `--policy vision` and a compatible CNN checkpoint. Do not infer semantic compatibility merely because a filename ends in `.pt`.

</details>

## 29. Source map and further reading

### Start here: Where to read next

The source map connects these notes to the books and your files. Use it when a particular question appears during training.

For rewards and return, read Sutton–Barto Chapter 3. For TD learning, read Chapter 6. For multi-step feedback, read Chapters 7 and 12. For actor–critic foundations, read Chapter 13. Study PPO after these ideas are familiar.

Use `env_cfg.py` to find active settings. Use `rewards.py` to see how raw reward scores are calculated. Use `observations.py` to see what information the learner receives. Use evaluation reports to find measured outcomes.

**Write in your notebook:** Read the theory that explains the question, then inspect the code that implements it. Check which version and experiment produced each result.

<details markdown="1">
<summary>Open the detailed explanation, equations, and project examples</summary>

### 29.1 Sutton–Barto chapter map

Numbers below are **printed book pages**, which differ from PDF page indices because of front matter.

| Chapter in your draft | Printed start | Core notebook material | Robotics relevance |
| --- | ---: | --- | --- |
| 1: The RL problem | 1 | Interaction, objectives, policy/value/model | Formulate lifting rather than merely choose an algorithm |
| 2: Multi-arm bandits | 27 | Exploration, averages, optimism, UCB, gradient bandits | Understand discovery before sequential grasp control |
| 3: Finite MDPs | 47 | Reward, returns, Markov property, values | Describe observations, hidden memory, time and success |
| 4: Dynamic programming | 79 | Policy evaluation/improvement and value iteration | Understand Bellman backups and model assumptions |
| 5: Monte Carlo | 99 | Complete sampled returns and off-policy estimation | Learn from full task attempts and compare sampling schemes |
| 6: TD learning | 127 | Bootstrapping, SARSA, Q-learning, maximization bias | Interpret critic errors and action-value learning |
| 7: Multi-step bootstrapping | 151 | n-step targets and horizon tradeoffs | Connect delayed lift rewards to earlier decisions |
| 8: Planning | 167 | Models, Dyna, model error, prioritized sweeping | Separate planning from model-free simulator training |
| 9: Prediction with approximation | 191 | Features, gradients, linear/neural prediction | Understand critic generalization and nonstationary targets |
| 10: Control with approximation | 229 | Approximate control, continuing-task considerations | Recognize task objective and representation choices |
| 11: Off-policy approximation | 243 | Divergence examples and deadly triad | Diagnose learning instability without conflating physics NaNs |
| 12: Eligibility traces | 251 | Lambda returns, traces, forward/backward views | Understand temporal credit and GAE's related construction |
| 13: Policy gradients | 265 | REINFORCE, baseline, actor–critic, continuous policies | Understand your actor and critic before PPO extensions |
| 14: Psychology | 281 | Conditioning, delayed reinforcement, habitual/goal-directed behaviour | Intuition for credit assignment and model-free/model-based distinctions |
| 15: Neuroscience | 319 | Reward, values, prediction-error hypothesis, actor–critic links | Distinguish a reward from a reward-prediction error |
| 16: Applications | 365 | Games, Acrobot, memory control, vision games, Go, soaring | Learn how task formulation and approximation differ across domains |
| 17: Frontiers | 403 | Connections and open questions in the draft | Understand that practical neural robotics still needs empirical evidence |

### 29.2 Additional book concepts to retain

**Importance sampling:** off-policy estimation compares a target policy with the policy that generated samples. Likelihood ratios correct weighting only when coverage/assumptions are suitable; products over long trajectories can have high variance. Replay alone is not the same as unbiased evaluation of any new policy.

**Gradient bandits:** learn preferences for actions rather than estimates of their mean reward, commonly using a softmax distribution and an average-reward baseline. This is a bridge to policy gradients, not your six-joint control algorithm.

**Continuing tasks:** some robots operate without natural episode boundaries. Average-reward formulations focus on long-run reward rate and differential values rather than the same finite-attempt return used here. Your current 6-second task is explicitly episodic.

**Feature representations:** polynomials, Fourier features, tile coding, and radial basis functions encode different generalization assumptions. Deep features are flexible but do not remove the need for suitable information and training coverage.

**Forward versus backward views:** a forward view specifies updates using future rewards/targets; a backward trace assigns current feedback to past feature activity. Their equivalence depends on the precise algorithm and assumptions. GAE's sampled weighted residuals are related to lambda-return reasoning; they should not be described as identical to every TD(lambda) implementation.

**Conditioning and neuroscience:** the book connects delayed reward learning and prediction errors with animal learning and dopamine research. These motivate intuitions, not literal claims that a neural network critic is a biological brain. An unexpected positive outcome can produce positive TD residual; an expected reward omitted can produce negative residual; a fully expected reward can have small residual.

**Habitual and goal-directed behaviour:** policies learned from repeated experience can act directly without online model search; model-based systems can reason through predicted outcomes. Both can be combined. Calling a learned policy “goal-directed” in everyday language does not establish that it performs model-based planning.

**Applications and frontiers:** task-specific successes rely on task representation, data, models/search where used, and evaluation. Acrobot illustrates underactuated control and delayed effects; TD-Gammon illustrates approximate values; game-playing examples illustrate combinations of value learning and planning. No example guarantees that a custom contact-rich SO-101 task converges under arbitrary rewards.

### 29.3 Robotics excerpt map

| Printed pages in supplied excerpt | Topic | Matching sections here |
| --- | --- | --- |
| 1–7 | Motivation and robot learning | 3, 19, 23 |
| 8–14 | MDP, reward, Bellman, exploration, HER | 4–13, 17 |
| 15–21 | Policy gradients, PPO, SAC | 15–17 |
| 22–28 | Simulation | 18–19 |
| 29–35 | Randomization, transfer, adaptation | 20–21 |
| 36–42 | Imitation, DAgger, data | 22 |
| 43–49 | ACT, diffusion, action multimodality | 23 |
| 50–56 | VLAs and fine-tuning | 23 |
| 57–63 | Locomotion, whole-body, industrial control | 18, 23 |
| 64–70 | Robustness, deployment, monitoring | 24, 26 |

PDF pages 27/28 duplicate printed page 25; PDF pages 55/56 duplicate printed page 52. Do not count those duplicated screenshots as extra covered chapters.

### 29.4 Primary supplementary references

- [PPO](https://arxiv.org/abs/1707.06347): recent rollout reuse and clipped surrogate optimization.
- [GAE](https://arxiv.org/abs/1506.02438): weighted residual advantage estimates.
- [SAC](https://arxiv.org/abs/1801.01290) and [SAC algorithms and applications](https://arxiv.org/abs/1812.05905): maximum-entropy off-policy learning.
- [Potential-based shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf): conditions for preserving optimal policies under reward shaping.
- [DAgger](https://arxiv.org/abs/1011.0686): learner-induced state distribution and expert relabeling.
- [ACT](https://arxiv.org/abs/2304.13705): action chunks and temporal ensembling.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137): conditional generative action-sequence learning.
- [OpenVLA](https://arxiv.org/abs/2406.09246) and [pi0](https://arxiv.org/abs/2410.24164): distinct VLA action-generation architectures.
- [MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) and [Genesis documentation](https://genesis-world.readthedocs.io/en/latest/): check current backend capabilities rather than copying broad simulator claims.

### 29.5 Project reading map

All relative paths below are in `/home/ai/jamshi/isaac_vision_rl`.

| Question | Read |
| --- | --- |
| What scene, actions, inputs, rewards, and endings are active? | `src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py` |
| What does a reward actually calculate? | `src/isaac_vision_rl/tasks/pickplace/mdp/rewards.py` |
| What information reaches actor and critic? | `src/isaac_vision_rl/tasks/pickplace/mdp/observations.py` |
| What are PPO and exploration settings? | `src/isaac_vision_rl/tasks/pickplace/config/so101/agents/rsl_rl_ppo_cfg.py` |
| How are exploration bounds and diagnostics implemented? | `src/isaac_vision_rl/utils/rl_agent.py` |
| How are commands smoothed and bounded? | `src/isaac_vision_rl/tasks/pickplace/mdp/actions.py` and `actions_cfg.py` |
| How do curricula and evidence gates work? | `src/isaac_vision_rl/utils/pipeline.py`, `scripts/train_pipeline.py` |
| How are complete episodes measured? | `scripts/eval_pipeline.py` |
| How are grasp geometry and resets verified? | `scripts/verify_grasp.py`, `scripts/check_resets.py`, `utils/grasp_geometry.py` |
| How is invalid physics diagnosed? | `scripts/check_stability.py`, `mdp/health.py` |
| Where do camera estimates come from? | `src/isaac_vision_rl/perception/cube.py`, `scripts/verify_camera_perception.py` |
| What operational limitations are documented? | `docs/agent_training.md`, `docs/training_pipeline.md` |

These notes add learning material only. The existing source, training settings, checkpoints, and gate files were not changed while preparing them.

</details>
