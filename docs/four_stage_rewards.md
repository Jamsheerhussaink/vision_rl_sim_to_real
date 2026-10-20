# Four-stage reward laboratory (Reach distance and gripper readiness are separate)

Continuous approach scores (`--reward-style continuous`) are dense current-state
rewards: every step is scored, and the value changes smoothly with distance only.
During Reach, the distance target is the overhead point 8 cm above the cube center. During Grasp, it is the cube center.
The box position is never used as a Reach target. Getting closer earns a higher
distance score; getting farther lowers it. Wrist angle, jaw opening, and other
pose signals do not alter this distance curve. A separate Gripper Readiness
chart rewards a downward wrist and open jaws during Reach; in Grasp, it rewards
the wrist angle only so closing the jaws is not discouraged. During Reach the
distance target is the cube-derived overhead waypoint, so the box cannot attract
this reward. Both scores are multiplied by control-step duration and their
weights. Because they are dense state rewards, a stationary robot keeps
receiving its current per-step scores; stage transition and evaluation must
confirm actual grasping.
Revision v7 was introduced after model 300 completed zero of three fixed-scene
evaluations: it hovered about 14 cm from the overhead target with closed jaws.
Reward-ranking and static simulator tests do not establish learned task success.

This is an opt-in, separately versioned full task. Old rewards and checkpoints remain available through the legacy profile. Start the new profile with a fresh policy: its actor additionally observes the four-element task-phase indicator, and its new checkpoint schema is `four_stage_v2_bounded` (the value network also observes grasp-bonus history). It remains RGB plus joint/controller observations; the actor does not receive cube coordinates. Stage indicators are task context derived by the simulator, not a learned visual detector.

Every episode starts at the measured arm home pose with the gripper closed (0.0 rad). The policy must learn to open the jaws before grasping. The cube and box are fixed and mass/friction randomization is off in this initial laboratory profile. No approach-reset curriculum or automatic grasping is enabled. Manual, scripted and PPO modes share the environment configuration, action processing, stage machine, reward families, and safety terms.

## Smooth motion references

The four-stage profile uses arm reference speed <=0.5 rad/s and acceleration <=1.0 rad/s², wrist-roll speed <=0.2 rad/s, and gripper speed <=0.45 rad/s with acceleration <=0.9 rad/s². The existing arm low-pass filter remains active. The gripper now brakes toward its target instead of receiving an immediate position jump. At episode reset, its reference starts at measured joint position with zero reference velocity. Joint limits take precedence over smooth braking at hard endpoints. These filters are shared by manual control, training, scripted checks and policy evaluation.

Motion-profile metadata is `four_stage_smooth_v1`; old four-stage smoke results used the earlier controller and do not establish performance with the new one. Stronger smoothing changes policy dynamics and may require retraining. These are commanded-reference bounds, not guarantees on physical velocity, collision forces or zero vibration. No extra reward terms were added.

Recorded reversal diagnostic: `outputs/four_stage/smooth_controller.json` and MP4, 90 control steps. Repeated abrupt gripper target reversals produced peak reference speed 0.4500 rad/s and acceleration 0.90004 rad/s² (floating-point tolerance). Sixty targeted tests passed after this update. A stationary arm in this diagnostic does not establish full-motion tracking quality.

## Three terms per family, with overlapping activation

Reach is active in stages 1–2; positive grasp shaping in stages 2–3; the table-slide penalty remains active from stage 2 through placement; lift only in stage 3; placement only in stage 4. Each family has at most three terms, including its penalties. Reach now has two visible lines after merging its approach and alignment shaping. The placement stage no longer pays for maintaining a grasp, while the lost-grasp failure check still requires the cube to remain held until a valid box release.

Each row is one scalar term, including penalties. There are no other reward terms. Diagnostic measurements and failure conditions are not additional rewards.

| Stage | Term | Weight | Raw value |
|---|---|---:|---|
| Reach | Overhead approach (distance + alignment) | +0.5 | During Reach, first reward matching 8 cm clearance; then reward decreasing the full 3D distance to the pregrasp target. Alignment and opening feedback are combined at the equivalent of the previous +0.1 alignment weight. This penalizes climbing too high or moving away. In Grasp, reward progress down toward the cube |
| Reach | Table clearance | -1 | `clip((0.008-clearance)/0.008,0,1) * dt` |
| Grasp | Cube between fingers | +4 total | Only new episode-best enclosure improvement; no refill after release |
| Grasp | Both fingers touching cube | +8 total | New best fraction of 0.2 seconds of continuous valid contact; capped for entire episode |
| Grasp | Table sliding speed | -2 | `clip(horizontal cube speed/0.05,0,1) * dt`, while cube is on table |
| Lift | Supported lift progress | +12 | New-best progress toward a 0.15 m rise from the episode's starting cube height, gated by valid grasp; excess rise above 0.20 m is penalized |
| Lift | Held-lift dwell | +8 | New-best fraction of 0.5 seconds of valid elevated hold |
| Lift | Object speed | -1 | `clip(cube speed/0.25,0,1) * dt` |
| Place | Box approach progress | +10 | New-best increase of `clip(1-horizontal box distance/0.30,0,1)`, supported only |
| Place | Lowering progress | +6 | New-best increase in closeness to box floor height, supported and within 3.5 cm of box centre |
| Place | Settled release | +40 | One event after successful released/withdrawn placement dwell |

`dt` is seconds per control step (1/30 by default). Distances are metres, speeds metres/second, forces newtons, and joint angles radians. The panel shows reward contribution by term. The reward-manager adapter divides integrated/event rewards by dt because Isaac Lab multiplies by dt; it does not add rewards. Reach uses an ordered route: first the gripper must rise toward 8 cm above the cube and remain at that clearance for 0.1 s. Then the proximity reward uses the full 3D distance to the overhead pregrasp target, so climbing too high increases distance and gives negative progress. The baseline resets at the route switch to avoid a fake reward jump. If the gripper drops below 6 cm before leaving Reach, it returns to the raise subgoal. Moving away from the active goal produces negative signed progress, while standing still produces zero distance progress. In Grasp, the reach family remains active and rewards centered descent toward the cube. In the default bounded mode, grasp quality/contact earns only new episode-best progress, at most 12 combined points per episode. Lift and placement shaping remain new-best progress or capped dwell/event rewards. Holding or regrasping cannot refill those bonuses; reward alone does not establish task success. Stage changes reset their progress baselines; stages advance only once per episode. Touching/cycling cannot repeatedly earn those progress/milestone rewards.

Progress reward depends on how much the gripper moves toward the current subgoal during that control step, not how long it waits nearby. The first subgoal is vertical movement to the safe 8 cm clearance; the second is full 3D movement to the overhead target. This is intended to teach a visible raise-center-descend path while retaining the existing three-term Reach reward limit. Reward curves are diagnostic: judge success by observed alignment, two-finger grasp, held lift, and placement outcomes.

Enclosure progress is approach/closure shaping, **not proof of physical grasp**. Only the bilateral dwell and supported lift terms establish contact-dependent reward. Continuous safety/drag penalties are bounded per second. Stage weights are an initial interpretable design, not empirically optimized values.

## Transition and termination rules

- Reach → Grasp: open jaws >1.10 rad; tool within 2 cm of a point 8 cm above cube centre; downward alignment >0.8; clearance >=5 mm; continuously for 0.15 s. Reach rewards are reduced for side-level or off-centre approaches and strongest when the gripper is centered above the cube. The alignment reward also increases when the gripper opens, so a closed-start gripper has a learning signal to open before descending.
- Grasp → Lift: opposing cube-specific fingertip contacts, cube within 3 cm laterally/3.5 cm vertically, downward alignment >0.8, jaw <1.10 rad; continuously for 0.2 s.
- Lift → Place: same supported enclosure, cube rise 0.15–0.20 m above its episode-start height, cube speed <0.25 m/s; continuously for 0.5 s. This advances the stage, not the episode.
- Place success: full oriented cube containment on the box floor, linear speed <0.05 m/s and angular speed <0.5 rad/s; jaws >1.10 rad; no bilateral grasp; tool withdrawn >6.5 cm; continuously for 0.4 s. Containment uses the existing box-frame geometry implementation, not merely centre proximity.
- Broken dwell resets its timer. After grasping, loss of grasp for 0.2 s fails the episode unless releasing close to the box floor/interior. There is no backward cycling; retry requires an episode reset.
- Existing table guard remains: measured wrist/finger mesh clearance <3 mm or summed filtered table force >0.1 N fails. The 8 mm reward margin encourages distance before this threshold. Dropped cube below 0.35 m, invalid physics, robot dropping and camera-visibility failures remain enabled. Timeout is 45 seconds.
- Stage failure and successful placement are distinct from timeout. Incorrect release fails or times out; it cannot earn the placement event. Source collision-mesh clearance and control-step contact samples are simulation checks, not hardware interlocks or continuous-time safety guarantees.

## Manual controls: first try this

Automatic visible wrist diagnostic (one simulated environment, no learning or
physical motor commands): `python scripts/check_wrist_orientation.py --keep-open`.
It places the arm at four static poses with a shared tool-position target, holds
each pose, and compares the actual measured reward terms. These are diagnostic
placements, not a validated continuous motion trajectory. Results and camera
comparisons are saved to `outputs/four_stage/wrist_orientation_check.json` and
`.png`. Close Newton when finished; `--keep-open` leaves the final view visible.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/staged_task.py --mode manual --record-video --output outputs/four_stage/manual_session.json
```

One Newton environment opens, initially paused. In **SO-101 manual control and live rewards**:

- **Run/Pause** advances or pauses simulation.
- Hold a joint's **−/+** button to jog that joint. Release to hold. There are controls for all five arm joints and the gripper.
- **Jog speed** sets 0.05–0.5 rad/s; references are clamped to joint limits and pass through the same arm controller as training.
- **STOP / HOLD** freezes the current reference and pauses. No hardware is connected.
- **Manual inspection override:** all episode terminations, including success, table contact, visibility, dropped objects and timeout, are currently disabled in `--mode manual` only. One environment is enforced. Unsafe poses/lost contact do not permanently latch stage failure, so you can recover and keep inspecting rewards. Unsafe states do not complete milestones. **Reset to HOME** is explicit and clears episode return, stage memory and graph histories. Training, scripted diagnostics and policy evaluation retain their termination checks.
- GUI text shows current stage, completion condition, active reward families and their individual weighted terms, step total, episode return, finger forces, bilateral contact, object height/speed, clearance/table force, and sliding.
- Rewards are drawn by Newton's ImGui callback, not into the policy's RGB camera. Ordinary camera video does not include the controls. Session JSON records the numeric trace. Manual control does not update policy weights.
- A second **Reward charts | TensorBoard style** window provides a separate chart for each reward/penalty: simulation-step X axis, weighted-reward Y axis, grid, faint raw curve, bold EMA-smoothed curve, and a smoothing slider. Rewards are green, penalties red. Disable **Only active stage** to see all 12 charts. Disable **Follow latest steps** for native ImPlot pan/zoom. Histories retain up to 3,000 steps; inactive terms record zero. Reset clears histories. Smoothing is display-only and never affects reward or PPO.
- Newton uses clear flat fallback colors: original green table, yellow cube and dark robot, plus the requested darker brown box. These modify Newton's display model only; USD assets and the RGB camera materials are unchanged. A regression test prevents a tabletop primitive named `box` from being colored like the target box.

Use the camera preview for fine alignment; rotate the Newton scene as needed. Real motor execution is deliberately outside this tool.

## Train and evaluate

Fresh training, Newton displaying one of 16 environments:

```bash
python scripts/train.py --task place --policy rgb --curriculum place --reward-profile four-stage --live-reward-graphs --lr-schedule adaptive --cube-jitter 0 --envs 16 --iterations 2000 --device cuda:0 --viz newton --max_visible_envs 1
```

During training, `--live-reward-graphs` opens read-only TensorBoard-style charts for environment 0, the visible robot. It never exposes manual jogging or changes PPO actions. Curves persist across automatic episode resets. TensorBoard additionally receives `FourStageReward/<term>` scalar diagnostics averaged across all training environments; those batch averages differ from the visible robot's individual curves.

Headless: remove `--live-reward-graphs` and replace `--viz newton --max_visible_envs 1` with `--headless`. Do not run both simultaneously. New logs use `so101_pickplace_rgb_four_stage_v3_overlap`. Existing v1 experiments remain preserved. The actor input remains unchanged, but the critic gets three bonus-history values, so bounded mode uses checkpoint schema `four_stage_v2_bounded`. No legacy checkpoint should be supplied: legacy schemas are rejected rather than silently relabeled as four-stage policies.

Independent evaluation from home (replace CHECKPOINT with a new four-stage checkpoint):

```bash
python scripts/staged_task.py --mode evaluate --checkpoint CHECKPOINT --seed 101 --headless --max-steps 1350 --record-video --output outputs/four_stage/policy_evaluation.json
```

This is a one-episode diagnostic, not a multi-seed certification. Repeat with seeds 101/202/303 before judging reliability. The scene remains fixed; seeds alone do not demonstrate randomized-position generalization.

Scripted diagnostic, explicitly not learned behavior:

```bash
python scripts/staged_task.py --mode scripted --headless --max-steps 900 --record-video --output outputs/four_stage/scripted_demo.json
```

## Verification

```bash
PYTHONPATH=src:. python -m pytest tests/test_four_stage.py tests/test_reward_panel.py tests/test_newton_colors.py tests/test_hand_table.py tests/test_picklift_logic.py tests/test_pipeline.py tests/test_agent.py -q
```

Inspect videos alongside JSON: contact is not necessarily grasp; cube elevation is not necessarily held lift; high reward is not necessarily successful placement. A completed stage machine in synthetic tests validates logic, not robot motion. Report actual bounded simulator outcomes separately.

## Completed bounded checks, 17 October 2026

- 55 targeted tests passed. These include stage dwell and transitions, reward budget, repeated-reward prevention, one-finger rejection, lost contact, table safety, invalid/valid placement geometry, GUI jog/stop requests, signed independent histories, and object-color classification.
- Four-step headless hold/video check passed; GUI callbacks and a six-second stationary live-panel recording passed. Window screenshots were inspected, including the restored scene colors and individual ImPlot graphs.
- A two-environment/two-update fresh PPO run saved `logs/rsl_rl/so101_pickplace_rgb_four_stage_v1/2026-10-17_16-56-14/model_1.pt`. Checkpoint metadata records `four_stage_v1` and `four-stage`. A separate 30-step home-start inference loaded it and produced a readable video. It did not place the cube; two updates are only a workflow check.
- `outputs/four_stage/scripted_demo.json` and matching MP4: 537 scripted steps, all four stages visited, then `stage_failure` after loss of the cube during transport. Minimum sampled hand/table clearance 6.42 mm. This is a recorded failed scripted diagnostic, not a learned-policy demonstration or proof of successful physical placement.
- No long training or hardware motion was launched. Existing checkpoints were preserved. A complete physics placement and a trained successful full task remain unestablished.

Files implementing the new profile: `utils/four_stage.py`, `utils/four_stage_terms.py`, `utils/reward_panel.py`, `utils/newton_colors.py`, `scripts/staged_task.py`, plus opt-in wiring in `scripts/train.py`, `utils/pipeline.py`, `utils/rl_agent.py`, and `mdp/events.py`. Legacy behavior remains the default outside the opt-in profile.

Reach completion accepts either the safe open pregrasp pose or a valid enclosed, closed, opposing two-finger grasp, continuously for 0.15 seconds. This permits manual or policy direct grasps without requiring reopening. Stage 2 still requires its own 0.2-second grasp dwell; safety checks still block advancement even in manual inspection. The manual GUI shows pregrasp, enclosure, bilateral contact and unsafe flags.

## Standalone reward window with headless simulation

Use `--headless --reward-window` instead of Newton/live-reward-graphs to open the same graphical reward charts in a separate desktop window, without the simulator viewer. It displays environment 0 only even when PPO trains 16 environments. The desktop display must be available. Samples pass through a nonblocking pipe; a closed or slow chart window cannot block training. Closing this read-only window does not stop PPO; stop training in its terminal. The window closes when training exits.

```bash
python scripts/train.py --task place --policy rgb --curriculum place \
  --reward-profile four-stage --lr-schedule adaptive --cube-jitter 0 \
  --envs 16 --iterations 2000 --device cuda:0 --headless --reward-window
```

## Successful training videos

Add `--record-successes` to retain one RGB MP4 per successful full placement, across **all training environments**, not only environment 0. Camera frames are encoded at 640×360 at the control rate. Each episode is recorded to a temporary disk file; failed or unfinished attempts are deleted on reset/normal exit. Successful videos and JSON metadata are retained under `logs/rsl_rl/so101_pickplace_rgb_four_stage_v3_overlap/successful_videos/<launch timestamp>/`. They show stochastic training rollouts, not independent checkpoint evaluations. Recording every environment adds CPU/encoding/disk overhead. An abruptly killed process may leave hidden `.attempt*.mp4` files; these are not successful recordings.


## Prepared changes on October 19 (not applied to the running training)

Finger contact capacities increased from 64 to 1024 per prim. This addresses the observed truncation limit, but a simulator contact test is still required after current training stops; this edit alone does not prove the warning is eliminated. Contact force/opposition/inside-fingertip tests, physical geometry, camera, table safety and smooth control limits remain intact.

Default `--grasp-reward-mode bounded` uses v4 bounded rewards and a new critic schema. Start a **fresh** policy for it; do not directly resume a v3 checkpoint into this schema. Explicit `--grasp-reward-mode overlap` retains the old v3 reward behavior and critic shape for old checkpoints. Evaluation automatically reads the checkpoint metadata to select the old or new mode and goal. The old in-memory training continues using its previously loaded code/configuration.

Optional `--training-goal lift` ends successfully only after approach, grasp and a 0.5-second supported elevated hold. It never advances to Place. It retains home/closed-jaw starts and all production safety failures. `--training-goal place` remains the full task. Video JSON identifies `stable_held_lift` versus `four_stage_placement`; these are not interchangeable success criteria.

After current training has stopped, first record a sequential three-seed evaluation of the old checkpoint:

```bash
python scripts/evaluate_stage_batch.py \
  --checkpoint logs/rsl_rl/so101_pickplace_rgb_four_stage_v3_overlap/2026-10-19_09-42-12/model_1800.pt \
  --output outputs/four_stage/before_retuning_1800
```

The evaluator refuses to run alongside a detected training process, preserves existing videos and checkpoints, and records per-seed logs, videos, traces and a summary separating stage visits, goal success, table violations and sliding. Fixed-scene seeds are not a generalization test.

Prepared next **fresh full-task** command (not launched; placement retained per user request):

```bash
python scripts/train.py --task place --policy rgb --curriculum place \
  --reward-profile four-stage --grasp-reward-mode bounded --training-goal place \
  --lr-schedule adaptive --cube-jitter 0 --envs 16 --iterations 2000 \
  --device cuda:0 --headless --reward-window --record-successes
```

Before this long run, validate contact capacity and a bounded save/load/evaluation path when the GPU is free. The current CPU tests do not establish simulator grasp reliability.


## Relative lift-distance shaping and placement

Lift height is now measured as **how far the cube rises from its height at episode start**, instead of its absolute height from the environment origin. The defaults are a 0.15 m minimum rise and a 0.20 m upper safe limit; both can be changed with `--lift-rise-min` and `--lift-rise-max`. The cube must stay grasped, move slowly, and remain in that band for 0.5 seconds to leave Lift. Above 0.20 m, lift reward falls and a penalty applies; this discourages lifting higher than needed. The box's old absolute-height measurement is still used for lowering into the box.

The older checkpoint's recorded evaluation started with the cube centre near 0.426 m and later reached about 0.572 m, roughly a 0.146 m rise. It did pick and lift, but did not quite demonstrate the new 0.15 m target held steadily for 0.5 seconds. The checkpoint is still a useful starting policy; the small extra rise and stable hold must be verified in simulation before being treated as achieved.

In Place, box-approach credit is disabled while too high and excess height is penalized through the existing lowering term. Lowering below the carry band into the box is allowed. Settled/released/withdrawn placement remains the full-task success criterion. `--training-goal place` is the default and explicitly used in the prepared next command. It does not assert the task is learned or that a deadline guarantees success.

Graphs use beginner-friendly display names; stable internal metric keys are retained for log compatibility. Signed height/lowering graphs identify their too-high penalty as well as positive reward. The current already-running graph process retains its old labels until the next launch.

## Every stage success and major failure (next-launch recording)

`--record-successes` now enables both full successful-episode recordings and stage-event clips for **every training environment**. No stage or failure sampling cap is imposed. The currently running process keeps its old recorder until restarted.

Within the launch's video directory:

- `stage_success/reach/`: segment ending when Reach passes its safety/dwell check.
- `stage_success/grasp/`: segment ending when the opposing grasp passes its dwell check.
- `stage_success/lift/`: segment ending when the stable lift passes its height/speed/dwell check.
- `stage_success/place/`: segment ending after settled release and withdrawal.
- `failures/`: the stage segment leading to table contact, dropped cube, lost grasp, camera exit, robot drop, or episode ending before a milestone. JSON records the actual measured reason(s); interrupted recordings are explicitly labelled `recording_interrupted`, not physical failures or successes.
- Root `success*.mp4`: full successful episode for the selected lift or placement goal, with JSON metadata.

Each finalized clip has environment/episode/stage identifiers and simulation-step metadata, so a successful earlier stage remains available even if the episode later fails. Stage completion is a simulator condition, not proof of learned mastery. The ordinary full-episode temporary recorder still discards failed attempts, while the separate stage recorder retains the failure segment. This requires two encoders per environment and increases CPU/disk use. A hard kill may leave the currently open segments unplayable, but already finalized milestone/failure clips remain readable. No existing recordings or checkpoints are deleted.

## IK teacher and aligned demonstrations

`python scripts/check_ik_teacher.py` performs CPU-only nominal IK and sampled wrist/finger-versus-table clearance checks. It sends no motor commands and does not validate dynamics, box collisions or grasping. `outputs/four_stage/ik_teacher_preflight.json` records the checked joint targets.

With training stopped, run a bounded teacher attempt (omit `--headless` to watch Newton):

```bash
python scripts/staged_task.py --mode ik-teacher --headless --max-steps 1350 \
  --record-video --record-demo --output outputs/four_stage/ik_teacher_attempt_NEW.json
```

Use a new output name for each attempt. This teacher opens at home, waits for measured pregrasp, descends, waits for valid opposing contact, lifts to the target height, carries above the box rim, lowers, releases and withdraws. Phase timeouts and production safety terminations remain enabled. Position-priority IK is opt-in for this teacher; the existing solver defaults are preserved. A nominal fixed-finger bias centres the cube between the actual asymmetric jaws; its reliability must be checked in simulation.

The `_demo` folder stores compressed chunks of **pre-action** `policy` (29 values), `image` (3×128×128 float32) and normalized six-joint actions, with post-action reward/termination labels. These are actor inputs only; simulator cube positions used by the IK expert are not imitation-policy inputs. Chunking bounds memory. `metadata.json` starts ineligible and is eligible for cloning only if the whole task succeeds without teacher failure. Failed attempts remain diagnostic data, not automatically admitted to cloning. This is demonstration capture, not an implemented behavioural-cloning trainer or proof of a learned visual policy.

First measured teacher attempt (`ik_teacher_trial01`) reached the cube but timed out in Grasp with only the moving finger reporting force; the fixed finger force was zero. Its demonstration is ineligible. It was not presented as a successful grasp. A lower nominal grasp target was rejected by the offline 5 mm table-clearance guard; the revised teacher retains height and changes lateral alignment instead. Training was stopped gracefully at the user's explicit request before simulator trials.


Measured teacher outcomes: `ik_teacher_trial06` and `ik_teacher_live07` each completed the full `placed` condition in 815 simulation steps (same fixed scene/seed; not a generalization study). Camera video and aligned actor-only input/action datasets were recorded. Source-home, fingertip force, bilateral grasp, held lift, box release and withdrawal checks remained enabled. The initial release failure exposed that slow commanded jaw opening loses grasp contact before becoming fully open; bounded mode now permits that intentional opening only with whole-cube horizontal containment near the box floor. Strict placement success is unchanged; overlap mode keeps the legacy rule.

The old slide-height proxy also counted low cube motion on the box floor as tabletop dragging. Bounded mode now excludes the whole-cube box-floor region from both dragging penalty and integrated slide-distance metric. Earlier traces retain their original metrics, so do not compare old total slide distance directly with the corrected metric. This is a height/geometry proxy, not direct cube/table contact sensing. IK uses geometry/feedback, not reward maximization; its reward charts are diagnostic. The live teacher can use `--keep-open` to retain Newton and graphs after the attempt, with recordings finalized before the hold-open loop.

## Continuous state reward style (v12)

The user requested continuous current-state rewards, including descent, with only final successful placement as an instantaneous event. New training/manual/IK launches now default to `--reward-style continuous` in bounded mode. `--reward-style progress` preserves the earlier v5 new-best style. Legacy `--grasp-reward-mode overlap` forces its v3 progress-style compatibility path. Saved checkpoints record `so101_reward_style`; evaluation restores it, defaulting older checkpoints to progress. The critic shape remains compatible with the bounded schema; this is a changed reward objective, not reproducibility of an old run.

Revision v15 restores independent continuous distance reward: exp(-distance / 0.10), multiplied by timestep and weight. Reach targets the cube overhead waypoint; Grasp targets the cube center. Route and orientation do not suppress this distance score. Separate jaw-opening readiness still requires downward orientation and cube alignment. Positive grasp rewards stop in Place. This reward does not guarantee a top-down trajectory; evaluate recorded behavior.

Continuous weights are per second: Reach proximity +0.30 and readiness +0.10; Grasp enclosure +0.10 and valid two-finger contact +0.10 through the grasp and lift stages; Lift height quality +0.15 and safe held lift +0.15; Place box proximity +0.15 and lowering depth +0.15. Positive grasp shaping is disabled once placement begins; the table-slide penalty remains active. Final placement is still a one-time +40. Existing penalty rates remain -1 clearance, -2 sliding and -1 object speed, with excess height subtracted through height/lowering terms. Each scalar density is multiplied by control dt once. Three terms per family and the requested family activation windows remain intact.

Continuous descent is zero at carry height and increases toward the actual box-floor cube-centre height, gated by whole-cube horizontal box containment. It remains positive while safely maintaining depth, including during release. Box-proximity credit requires supported carry height or being over the box, so dragging toward the box on the table cannot earn that term. Unsafe/new-failure steps cannot earn positive bounded task reward; their negative penalties are retained.

Audit reproductions before these changes are in `outputs/four_stage/reward_audit_20261019.json`: old closed-jaw hover could collect 27 points without leaving Reach; dragging toward the box could earn +5 approach against -0.067 sliding in a synthetic step; an unsafe terminal lift tick could collect +5.71; sideways entry into the box radius could create +2 lowering credit with no descent (also observed at step 615 in the recorded teacher carry phase). Those examples diagnose logic, not measured frequencies of policy exploitation. v6 fixes the latter three mechanisms and reduces the hover rate, but continuous state rewards still reward waiting. Their maximum combined positive rate is 0.5/s, at most 22.5 over 45 s, below the +40 completion bonus. This undiscounted bound does not prove reward hacking is impossible; temporal discounting and policy evaluation still matter. No PPO discount parameters were changed.

The already-open viewer retains its old reward code until relaunched. No competing simulator was started for this change. CPU tests cover rate consistency, retained contact/depth rewards, monotonic descent, zero box credit for table dragging, safety masking and one-time final bonus.

## Post-attempt home return for IK demonstrations

IK-teacher attempts now run a separate guarded simulation return after normal success or a recoverable unsuccessful attempt. It opens while the cube is resting, moves vertically clear, travels to a reachable overhead home waypoint, folds to the initial measured arm home pose, then closes the gripper. Sampled source-mesh table clearance must remain at least 5 mm and measured guards stay active. Above-home uses a reachable 0.52 m tool waypoint; the box-clear waypoint is 0.55 m or the current higher tool position.

Recovery refuses unsafe/non-finite feedback, a cube above 0.445 m, or cube speed above 0.05 m/s (possible held/falling object). It does not automatically park after Ctrl+C or an exception. Safety-blocked cases hold instead of attempting an unchecked sweep. This is simulator-only recovery, not authorization for physical hardware.

The terminal task result and synchronized imitation samples stop at the original task boundary. Recovery uses the normal action filters and physics steps without restarting or stepping the terminated RL episode. It has no new task rewards and cannot turn a failed placement into a successful one. The rollout JSON includes a separate `return_home` report and the `_return_home.mp4` records recovery; original task video and teaching samples remain separate.


Revision v16: overlap checkpoint schema and reward style are independent. Explicit continuous style uses the existing continuous weights and distance curve with the v1 critic shape. Reach readiness combines 50% downward wrist quality, 25% gated jaw opening, and 25% height quality times downward wrist quality. Height quality peaks 8 cm above the cube and falls away on either side. This is soft orientation guidance, not a locked wrist or guaranteed top-down path. Stable cube rise of 15-20 cm advances to Place, where positive grasp shaping stops. Explicit progress remains available. Resuming old optimizer/critic state under changed rewards is fine-tuning, not exact replay of old training.


## Confirmed continuous reward specification (2026-10-20)

The user confirmed the following table as the desired specification. Graph labels match it verbatim. The continuous formulas and weights already implement these terms; this confirmation changes the display names, not their mathematical values. Use --reward-style continuous explicitly; progress is a different objective. Overlap selects compatibility with the older checkpoint observation schema and does not override the explicit continuous style.

| Reward or penalty | Weight |
| --- | ---: |
| Approach distance | +0.30 |
| Table clearance | -1.00 |
| Hand and gripper readiness | +0.10 |
| Cube enclosure | +0.10 |
| Valid two-finger grasp | +0.10 |
| Table dragging | -2.00 |
| Cube lift height | +0.15 |
| Stable held lift | +0.15 |
| Cube speed during Lift | -1.00 |
| Carry toward box | +0.15 |
| Lower into box | +0.15 |
| Successful placement | +40 |

Reach shaping continues through Grasp. Positive grasp shaping continues through Lift and stops in Place. Lift completion requires supported, slow cube rise of 15–20 cm for 0.5 seconds. Continuous state rewards accumulate at a steady rate while stationary. The height/wrist readiness cue is soft shaping and does not constrain joints or guarantee a top-down trajectory.


Revision v17: positive cube-enclosure and valid-grasp rewards are enabled in Reach, Grasp, and Lift (Stages 1–3), then disabled in Place (Stage 4). Table-dragging penalty remains enabled throughout. Existing stable 15–20 cm lift-rise transition (0.5 s, supported grasp, speed <0.25 m/s) is retained. Reach shaping still stops in Lift. Rewards on the transition tick belong to the stage before advancement; subsequent Place ticks have zero positive grasp reward.


### Reduced physics experiment (2026-10-20)

Four-stage configuration now explicitly uses gravity (0, 0, -8.829) m/s² instead of -9.81 and cube material static/dynamic friction 1.35/1.08 instead of 1.50/1.20 (10% reduction). Table and robot material settings, object mass, collisions, and reward weights are unchanged. Contact friction also depends on the other material and the combination rule. This configuration eases gravity loading but can increase slipping; it is less representative of Earth-gravity deployment. Evaluation loading the current configuration also uses these physics, so historical checkpoint comparisons must account for this change. It applies only to newly launched processes.


### Small task outcome values

Latest requested values supersede the historical +40 placement bonus: successful placement now pays +2 once. An unsuccessful stage failure or episode timeout pays -0.1 once, recorded as a negative value in the placement-outcome slot even if failure occurs earlier. Manual inspection disables this failure penalty. No penalty is added for ordinary unfinished intermediate steps. Reset clears the outcome-paid latch. Smaller success incentives can make prolonged dense state reward more attractive, so the previous completion-bonus dominance claim no longer applies.


### v19: distance-only approach (latest user request)

Hand and gripper readiness reward is removed: weight zero and raw value zero, hidden in charts and manual reward bars. Its tensor slot is reserved for checkpoint/manager compatibility and adds no reward. Approach uses only continuous distance: 0.30 * exp(-distance/0.10) * dt. During Reach the target is 8 cm above the cube; during Grasp it is the cube center. No separate height, wrist orientation or jaw-opening shaping remains. Stage-transition orientation/jaw/contact checks remain. Grasp/Lift/Place rewards, physics and small terminal outcomes are unchanged. Historical descriptions of readiness above are superseded.


### v20: separate reach and alignment (supersedes v19)

Reach reward stays 0.30 * exp(-distance/0.10) * dt. Separate alignment reward is 0.10 * exp(-(horizontal_offset/0.05)^2) * max(downward_cosine, 0) * dt. Both apply in Reach and Grasp, and stop in Lift. The old readiness slot stores alignment only for checkpoint compatibility, with its graph labeled Alignment (Centered + Pointing Down). No jaw-opening or hand-height shaping is included. Rewarding alignment is soft feedback rather than a hard wrist constraint.


### v21 anti-hover fine-tune

Continuous approach remains distance-based. A latched open-jaw overhead pose (pregrasp distance <2 cm, XY <2.5 cm, downward cosine >0.8, tool at least 6 cm above cube, jaws >1.10 rad) changes the target from overhead to cube center. The distance-progress baseline resets at that switch and at stage entry. At least 1 mm of new-best distance improvement resets the idle timer; backtracking/dithering cannot refill it. After 3 seconds without such progress, positive approach and alignment/open-jaw shaping decay exponentially with 2-second time constant, and a -0.05/s penalty is included in the Reach slot. Manual inspection does not receive the idle penalty. Alignment remains a distinct slot; it includes a +0.05/s maximum open-jaw bonus only when overhead alignment and downward orientation qualify in Reach. It stops encouraging opening in Grasp. The terminal placement bonus is +40 again, while unsuccessful ending remains -0.1 once. Grasp cutoff in Place, 15–20 cm stable lift, and reduced-gravity/friction experiment remain. Current task phase is still supplied to actor; the within-Reach latch and idle timers do not change checkpoint observation shapes. These changes require recorded evaluation; static reward-ranking tests do not prove learning.


### v22: jaw-opening progress fix

Evaluation of checkpoint 1050 reached within 0.56 cm of the overhead target but actual jaws never exceeded 0.82 rad; the unchanged opening gate is >1.10 rad. Anti-hover progress now includes cumulative new-best jaw opening within the overhead region (pregrasp distance <4 cm, XY <3.5 cm, downward cosine >0.8, tool >=6 cm above cube). Opening progress advances a nonrefillable episode best by at least 0.01 normalized opening. Distance baseline advances only for >=1 mm improvement, so smaller movements accumulate instead of continually resetting the baseline without timer credit. The gated open-jaw reward maximum increases from 0.05/s to 0.20/s, and safe overhead opening earns +1 once per episode in the alignment slot. Closing/reopening cannot refill this bonus. Unsafe/failure ticks suppress positive task rewards. Jaw thresholds, action mapping, physical contacts, no automatic grasping, existing lift/place rules and physics remain unchanged. These are shaping changes, not proof of successful learning.


### v23: stage semantics separate reaching from jaw opening

Reach completion now requires the safe aligned overhead pose (within 2 cm, downward cosine >0.8, >=5 mm clearance for 0.15 s), independent of jaw opening. Grasp handles opening before descent: both Reach and Grasp can earn the gated opening signal and once-only opening bonus. Until jaws exceed 1.10 rad at the overhead pose, continuous distance still targets overhead; the latch then switches it to cube-center descent and disables open-jaw shaping. Lift still requires enclosed opposing two-finger contact with closed jaws for 0.2 s. No iteration-based or forced stage promotion is introduced.


### v24: centered descent and near-cube closing

Checkpoint 1150 evaluation opened its jaws (93% of samples >1.10 rad) and pointed down, but stayed around 7 cm above cube center with no contact. After the open-overhead latch, continuous distance uses a 4 cm scale instead of 10 cm. A separate bounded new-best centered descent component is aggregated into Reach: at most +3 per episode, normalized from 8 cm above to cube-center height, gated by downward alignment and lateral centering; jaws must remain open until within the closing region. Repeated rises/descents cannot refill its best-so-far budget. Near the cube (XY <2.5 cm, absolute height offset <3.5 cm, cosine >0.8, Grasp stage and overhead latch), actual jaw closure adds at most +0.20/s to enclosure shaping. New-best jaw closing counts as progress for the idle timer. The Lift transition still requires real opposing enclosed two-finger contact; no scripted action, automatic attachment, relaxed contact gate or hardware movement is introduced. Checkpoint schemas are unchanged; internal reward memory is not added to actor inputs. Training and recorded evaluation must validate behavior.


### v25: use the successful IK teacher's pinch target

The descent target now matches IKTeacher._goal: cube center + fixed gripper_link local X axis times 0.016 m, plus 0.0095 m world Z. Overhead reaching still targets 8 cm above cube center. Closing shaping uses target-relative XY and height error (<2 cm each) instead of error from the cube center. Actual contact/enclosure checks remain unchanged. Reward-derived target coordinates do not enter actor observations or override actions. The live target rotates with the measured fixed-finger frame; teacher planning computes its bias from a solved neutral pose, so this is the same geometric offset rule rather than exact trajectory replay.

Offline audit outputs/evaluation/ik_target_geometry_audit.json checked all 19 recorded bilateral grasp samples across teacher_demo_004.json and ik_teacher_trial06.json. Every sample fits the closing region, with maximum target error below 0.40 cm. This validates target consistency with saved successful demonstrations; it does not establish that PPO has learned the motion or that dynamics still succeed. Fresh policy evaluation remains required. Diagnostic logs add target distance/XY/height errors, open-jaw fraction, and descent-unlocked fraction.
