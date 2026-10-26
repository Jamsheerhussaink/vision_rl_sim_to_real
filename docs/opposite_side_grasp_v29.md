# Opposite-side grasp guidance (v29, lift-target update v30)

The old per-finger rewards measured pad-centre distance to the cube centre.
They could reward top approaches and continuously pay for remaining close.
The old `jaws_open` measurement also increased as the joint closed.

The current trainer now reuses `experiments/simple_rewards/finger_surfaces.py`:

- Each finger targets a different cube side, with assignments fixed during Grasp.
- Distances use inward collision-mesh patches and central side regions, following cube rotation.
- Each distance improvement pays independently; retreat is negative and waiting adds zero.
- First samples and stage changes establish baselines without a reward jump.
- Positive finger-distance progress is suppressed when the cube slides on the table faster than 0.01 m/s.
- Verified opposing enclosed contact stops finger-distance credit. Lift and Place receive no finger-distance credit.
- Opening feedback increases with jaw opening, requires the existing downward/overhead alignment, and runs only during Reach.
- Enclosure/contact shaping no longer encourages premature closing in Reach.

The reward names remain independent in the grid. Finger plots accumulate signed
progress per episode. This revision keeps the current five Grasp terms; it does
not add another penalty. Physics, RGB observations, safety terminations, and
checkpoint files are unchanged. A running Python process must restart to load it.

Stage/display follow-up:

- Reach now requires the cube between open fingers at the pinch pose, downward orientation and safe table clearance for 0.15 s.
- Opposing enclosed contact with closed jaws must persist for the configured grasp hold time. Its timer carries across Reach into Grasp, avoiding a second hold delay.
- Episode reset immediately publishes Stage 1 and zero chart baselines, before Isaac Lab rerenders the home pose. Other environments resetting do not reset environment zero's panel.
- The panel displays the episode step, current grasp validity and pending failure/success. A stage label describes the current attempt, not completed placement.

Lift-target update (v30): the default rise is now 10-12 cm above each episode's
initial cube height. The 0.5 s steady-hold requirement and actual-grasp check
remain. Launcher defaults, checkpoint metadata fallbacks and panel text agree.
The saved trial06/demo004 traces meet the new contact/jaw/speed/height conditions
for about five seconds. This replay uses their historical contact labels and
is not a current-policy evaluation. Offline box bounds give about 8.7 mm rim
clearance for an upright cube at the minimum target; a tilted cube can hang
lower and still needs visual carrying validation. See
`outputs/evaluation/lift_height_10_12cm_audit_20261022.json`.

Viewer alignment fix: Isaac Lab's `max_visible_envs=1` alone permits random world
selection, whereas the RGB stream and reward panel read environment zero. The
Newton launcher now explicitly selects `visible_env_indices=[0]` and disables
random selection for a one-robot view. This fixes a scene/panel identity mismatch;
it does not prove that a policy has passed a manipulation stage. The grid also
gets a fresh full-height initial size so an older saved window size cannot hide
the final row; it remains movable and resizable after opening.

Anti-pushing update (v31): while the cube is on the source table, sliding faster
than 0.01 m/s and not held by a valid grasp, positive task reward is suppressed.
Negative progress and existing dragging penalties remain. Reward baselines are
still updated, so stopping after a push cannot pay back withheld progress.
Verified airborne carrying remains rewarded. This adds no reward term and does
not change physics or safety thresholds. Focused tests establish the reward
logic; learned-policy improvement still requires recorded evaluation.

Approach/reward-routing audit (v32):

- The previous adapter exposed only columns 0-2 of five reward columns to PPO.
  This omitted opening/centering and grasp/dragging columns even though the
  panel displayed them. The three adapter slots now sum columns with stride
  three, including every independently graphed term exactly once.
- A bounded Newton evaluation compared 83 actual PPO rewards against graph
  totals: zero mismatches at a 1e-6 tolerance (maximum difference 3.2e-9).
  Its graph/task stages also matched. The learned policy still failed that
  attempt; this establishes routing accuracy, not manipulation success.
- The failed model150 evaluation descended off target and approached within
  0.51 mm of the table. New descent guidance requires <1.5 cm sideways target
  error and downward cosine >0.95, with overhead and pinch targets sharing the
  calibrated XY offset. Losing alignment returns reward guidance overhead;
  switching targets establishes a fresh baseline without reward jumps.
- The IK teacher's overhead gate is independent of the RL pinch-height Reach
  gate. Its settled position tolerance is 4 mm after a recorded stable 3.1 mm
  tracking error caused a timeout with the old 3 mm tolerance; joint-settling,
  contact and safety checks remain active.

Training was stopped with checkpoints preserved while these checks ran. See
`outputs/evaluation/approach_regression_model150_20261022.json` and
`outputs/evaluation/reward_adapter_verified_model150_20261022.json`.

Recorded scripted check after the teacher tolerance correction:
`outputs/evaluation/ik_aligned_descent_reference02_20261022.json` reports one
successful IK placement over 731 steps, stages 0/1/2/3 observed, about 10.9 cm
maximum cube rise, minimum hand/table clearance 5.5 mm, zero measured hand/table
force, and a completed return home. This is one nominal scripted trial with
safety terminations enabled, not a learned-policy success or robustness test.

Validation:

```bash
PYTHONPATH=src:.:tests /home/ai/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_opposite_side_grasp.py tests/test_simple_reward_experiment.py \
  -q -k 'not original_sources_are_untouched'
```

The excluded test fingerprints the old trainer and necessarily reports a change
when this requested update modifies it. Its manifest has not been rewritten.
Geometry/unit checks do not prove learned grasp improvement. Record a bounded
simulator evaluation before making that claim; do not run another simulator
alongside active training when resources are tight.

## v33: allow an imperfect two-finger hold to try Lift

Acquisition now requires both cube-filtered finger force magnitudes above
0.02 N, existing cube enclosure, jaw below 1.10 rad, and the existing safe
0.5-second dwell. Pad contact position and opposing force angles remain
strict diagnostics but no longer block Lift. After acquisition, both contacts
and nearby cube geometry remain required; the existing lost-contact timeout,
unsafe termination, lift height band and held-lift dwell remain unchanged.
This relaxes grasp recognition, not the final task success rule. Two contacts
can still reflect an insecure pinch; recorded lifting evaluation is required.

Verification: tests/test_practical_grasp_gate.py covers an imperfect two-contact
hold passing after dwell, one-contact/no-enclosure/open-jaw/unsafe failures,
dwell reset on contact loss and lost-contact failure after Lift. No new
simulator or learned-policy evaluation was run for v33 in this change.

## v34: separate Reach progress and visible attempt boundaries

Audit of the active v33 TensorBoard event at update 61: Reach fraction 0.9796,
Lift fraction 0.0156, Place fraction 0, unsafe-table termination metric 0.8277,
mean reward -0.6971. These running training metrics are not evaluation success.
The v33 process was left running; its loaded code does not change in place.

Confirmed bugs fixed: the Wrist downward slot previously used a combined
readiness delta, and Jaws open / Centered XY paid state-based rates for waiting.
Continuous Reach now uses independent signed changes in wrist quality, gated
jaw-opening quality, and XY centring quality. Reset/stage initialization pays
zero, stationary quality pays zero, improvement positive, regression negative.
Existing grasp rules, other stages, physics, safety limits and actor inputs
remain unchanged. Graphs mark attempt reset samples with New attempt lines;
penalties and task bonuses retain their real meanings, not invented negatives.

Focused verification: 22 tests passed across separate Reach progress, practical
grasp gate, opposite-side grasp, and graph views. No v34 simulator evaluation
has been run; no improved manipulation claim is made.
