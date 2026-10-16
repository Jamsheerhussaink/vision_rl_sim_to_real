# Script cleanup note — 2026-10-15

Supporting-tool paths below reflect the current folders. `demo_scene.py` and
other earlier demo/debug tools are now archived; see
[archive/README.md](../archive/README.md) before running historical demo commands.
`record_episodes.py` was removed in the earlier RGB cleanup; use
`eval_pipeline.py --record-video` for current evaluation recordings.

# SO-101 RGB training and sim-to-real handover

Prepared 2026-10-13, Asia/Kolkata. Read this before continuing in another chat.
Confirmed local facts were checked against source/configuration files and logs.
Hardware results are user-reported from a separate computer, not independently
observed by the assistant. Assumptions and unresolved items are marked explicitly.

## 1. Objective and working preferences

Train SO-101 to identify/approach, grasp and stably lift a 5 cm cube using RGB
plus joint readings; later learn randomized lifting and placement into a box.
The user explicitly rejected depth inputs. Keep depth out of the active actor
and deployment controller. Simulator ground truth may train the critic/rewards.
Use robotics-code-tutor: beginner explanations, explain changes before editing,
exact verification commands, headless and GUI options. Preserve checkpoints.
Keep only necessary previews; replace the current scene reference instead of
accumulating images. Report actual task success separately from reward/loss.
Hardware runs on another computer; do not imply local tools can control it.
Latest user authorization: start a bounded simulation training experiment even
though physical grasp preflight failed. This is diagnostic, not certified training.
Then user asked for this handover, focusing the next chat on training. Do not
start additional jobs during handover or blindly duplicate an existing run.

## 2. Project and important files

Simulation computer:
- Project: /home/ai/jamshi/isaac_vision_rl
- Interpreter: /home/ai/miniconda3/envs/env_isaaclab/bin/python
- Isaac Lab checkout: /home/ai/IsaacLab
- Git branch: isaaclab
- origin: https://github.com/Jamsheerhussaink/vision_rl_sim_to_real.git
- Many current source changes/deletions are local; do not assume cloud/GitHub
  has them. Check git status before publishing or cloud execution.

Entry points:
- scripts/train.py: PPO training; defaults --policy rgb.
- scripts/play.py: checkpoint playback; defaults --policy rgb.
- scripts/eval_pipeline.py: explicit success evaluation, video and trace recording.
- scripts/demo_scene.py: home/scripted 3-degree pan demo, not learned policy.
- scripts/capture_snapshot.py: actual task-camera reset image.
- scripts/checks/verify_grasp.py: scripted kinematics/contact grasp feasibility.
- scripts/checks/check_workcell.py, check_resets.py, check_stability.py: physical checks.
- scripts/hardware/realsense_rgb_preview.py: real RGB-only preview, no motors/policy.
- scripts/record_episodes.py: retained recording utility; audit variant support
  before assuming every older utility supports the new RGB mode.

Source:
- src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py: base task scene.
- src/isaac_vision_rl/utils/pipeline.py: active RGB overrides and curriculum/gates.
- .../config/so101/agents/rsl_rl_ppo_cfg.py: PPO/network configuration.
- .../mdp/observations.py: actor/critic observations.
- .../mdp/actions.py and actions_cfg.py: gripper and filtered arm commands.
- .../mdp/rewards.py: dense terms and strict held-lift success.
- src/isaac_vision_rl/utils/rl_agent.py: PPO diagnostics, noise adaptation,
  checkpoint compatibility metadata.
- src/isaac_vision_rl/utils/rgb_image.py: shared image preprocessing.
- src/isaac_vision_rl/utils/cube_material.py: cube/cardboard grain materials.
- src/isaac_vision_rl/utils/grasp_geometry.py: offline forward kinematics/IK.
- src/isaac_vision_rl/utils/calibration.py: existing encoder conversion; its
  zero/sign convention is NOT a verified mapping to the real robot model.

Required assets: usd/robot_grasp_calibrated.usda, robot.usd, table_real_rgb.usda,
table.usd, table_green_texture.png, cube_grain.png, box_grain.png, box_reference.usda,
box.usd and their referenced assets. Do not delete source assets/symlinks casually.
Current preview: outputs/rgb_scene_reference.png and matching .log.
Current camera assumption record: outputs/rgb_camera_approximation.json.
Documentation: README.md, docs/rgb_only_training.md, docs/cloud_handoff.md,
docs/project_cleanup.md. PPO study notes contain historical pipeline details.

## 3. Simulation changes already made

RGB-specific configuration is applied in configure_curriculum() for --policy rgb.
Legacy --policy vision/state retains depth-assisted baseline compatibility.
- Render only RGB; no distance_to_image_plane sensor output in RGB mode.
- Remove actor camera_cube and assist_state; remove critic assist_state.
- Disable CameraGripperAction capture assistance and automatic release.
- PPO controls jaw opening/closure; no depth latch/pause in active controller.
- Keep simulator state in critic, rewards and evaluation only.
- Camera reset rerender is enabled. Earlier RGB-D cache invalidation fix remains
  for legacy modes; active RGB has no cube-estimator cache dependency.
- RGB capture: 424x240, actor image 128x128. Bilinear resize to 128x72 with
  28 black rows above/below; normalize /255. No aspect-ratio stretching.
- Physics dt=1/240 s; decimation=4; policy/render interval=1/60 s.
- Arm target EMA alpha=1-0.8**(25/60), preserving approximate old time response.
- Target slew limit 1.5 rad/s; per-control-step maximum reference change .025 rad.
- Acceleration limiter disabled by default. Actual joint speed is not guaranteed
  by a target-slew limit; physical actuator response must be measured.
- Gripper mapping q=.6*a+.6 rad: -1 closed0, +1 open1.2; clipped to 0..1.2.

Current layout (scene metres, quaternions xyzw):
- Robot base (-.20,.10,.405), quaternion (0,0,0,1).
- Home radians: shoulder_pan .00359878, shoulder_lift -1.25, elbow1.60,
  wrist_flex .85, wrist_roll0, gripper0 (closed).
- Reset joint position/velocity noise is zero. Home pose was visually approximated,
  not obtained from measured real angles. PPO can open the initially closed jaw.
- Cube 5x5x5 cm, default centre (.05,.02,.426), rests near .425 m.
- Box default position (.05,.26,.401); table surface z=.40 m.
- Cube mass .055 kg; friction static1.5/dynamic1.2, restitution0.
- Robot fixed base; arm effort limit2.94, velocity1.5, stiffness80/damping6.
- Gripper effort limit .20, velocity .8, stiffness20/damping .5.
- Collisions/contact geometry were retained during visual changes; moving the
  base changes reach geometry and invalidates old physical evidence.
- Table darkened, grain added to cube/box, overhead lighting and camera-side
  fill added. Newton blue background replaced with dark green.

Camera:
position (-.039820052072371465,-.15365800654010944,.7)
rotation (.9292824074053241,-.08947971851577595,
          .03434804580329896,-.3567180945940328)
horizontal aperture24.88, focal12.44, vertical aperture24.88*240/424.
Nominal HFOV90 degrees, square-pixel VFOV59.02299711418064 degrees.
Height30 cm above tabletop, horizontal camera/base distance30 cm.
Pitch48 degrees down; aim yaw11 degrees from +Y toward +X.
This predicts top-centre table intersection .8972038310420628 m forward.
User estimated far visible distance75 cm and width120-130 cm; these are rough,
not simultaneously satisfied by the final visually adjusted nominal lens model.
User latest real photo is the reference; exact pose/optics are not calibrated.

User-supplied lens values NOT APPLIED:
fx386.4156, fy385.4689, ppx323.9681, ppy244.8502;
inverse_brown_conrady coefficients:
[-.05631551146507263,.06490031629800797,.000712135573849082,
 -.000649842550046742,-.02080184407532215].
User says desired RGB424x240/60FPS, but stream/profile origin of these intrinsics
is unverified; ppy244.85 exceeds image height240. Do not silently rescale them.
Real D455 not connected to simulation PC. Actual selected RGB FPS is unmeasured.

## 4. Current PPO configuration

Actor: 25 numerical values (six joint positions, six velocities, six previous
commands, six filtered targets, remaining time) plus RGB3x128x128.
CNN: channels[32,64,64], kernels[8,4,3], strides[4,2,1], no padding;
flattened image9216 + numerical25 =9241 inputs to MLP[256,128,64], ELU,
6 action means. Observation normalization enabled.
Critic:53 privileged numerical values, MLP[256,128,64], ELU, 1 scalar value.
No recurrent policy, SAC, HER replay, or RND configured.

PPO: Adam lr.0003 fixed, 5 epochs, 4 mini-batches, clip.2, clipped value loss,
value coefficient1, gradient norm1, desired_kl=None, action clip +/-1.
RGB60Hz gamma=.99**(25/60)=.9958211159986682;
lambda=.95**(25/60)=.9788545612213748.
Rollout round(32*60/25)=77 steps/environment (~1.283 seconds).
32 environments =>2464 transitions/update,616 per mini-batch,20 mini-batch
optimization steps/update.500 updates =>1,232,000 collected transitions.
Seed42, saves every50 iterations. Six-second episode =>360 policy steps.
Initial episode ages can be randomized by runner init_at_random_ep_len=True.
Gaussian initialstd.4, learned logstd constrained .1..8; targetstd.2.
Entropy initialcoefficient.01 adapts rate.05 within .001..05.
Live noise/entropy values must be read from actual logs, not assumed initial.

Rewards retained: reach-progress+3, grasp-progress+4, lift-progress+8,
held-lift completion+6000; premature-close-3, dragging-5, displacement-12,
table-collision-.5, action-rate-.02, joint-velocity-.01, joint-limits-.05.
Reward manager timestep scaling applies. A unit success signal at60Hz contributes
6000/60=100, versus240 under old25Hz. Progress/event scaling is not perfectly
invariant under frequency change; evaluate actual task success separately.
Fixed curriculum disables cube mass/friction randomization and reset jitter.
No cube/box jitter requested in the diagnostic fixed run.

Strict held lift: bilateral cube-specific contact >.02 N per finger, cube z>.47 m,
XY cube/tool separation<.035 m, Z separation<.045 m, jaw<1.10 rad,
cube speed<.25 m/s continuously for .5 seconds (30 control steps at60Hz).
Gate: each seed101/202/303,50 episodes each,success>=90%,mean table slide<=1cm.
Do not weaken success criteria to improve reported scores.

## 5. Training status at handover: inspect before restarting

Latest local disk run found:
logs/rsl_rl/so101_picklift_rgb/2026-10-13_11-10-02/
Saved params confirm32envs,500iterations,77steps,seed42,resumeFalse.
Log: outputs/training/rgb_fixed_20261013_110933.log.
Last observed line:
[8/500] 1.6% |473 sps|Rew:-10.04 (R:+0.00 L:+0.00 P:+0.00)
Loss:5.64e-01|Std:.40|S:.00|ETA:0:47:03.
Only model_0.pt was present in that run at inspection. A host process listing
found no matching python scripts/train.py process. No completion/exception line
explained the stop. Treat active state as needing recheck; do not invent a kill/OOM.
A500-update directory existing does not establish completed training.

Other recent runs:
2026-10-13_10-28-06 and2026-10-13_11-02-14:4envs,2iterations,model_0/model_1.
outputs/rgb_training_smoke.log last inspected had older11-02-14 progress and NUL
bytes despite model_1 existing; do not certify from that single log alone.
Source check suite:42 unittest tests passed on the latest pretraining check;
syntax/whitespace checks passed. These do not prove grasp mastery.

Completed historical RGB run:
logs/rsl_rl/so101_picklift_rgb/2026-10-12_12-38-22/model_499.pt.
It trained500updates at25Hz, square512capture, stretched128 input, older layout;
final logs showed no held-lift success. No three-seed RGB mastery evaluation.
Other60Hz RGB model_0/model_1 files are smoke-test checkpoints, not mastery.

Checkpoint metadata: RGB schema rgb_policy_gripper_v1;
so101_rgb_profile='424x240_letterbox128_v1'; so101_control_hz=60.
Current loader rejects earlier RGB image metadata/timing. DO NOT bypass checks
or assume image-profile metadata proves matched camera pose (it does not).
Legacy vision schema camera_grasp_assist_v7 uses RGB-D geometry and assistance.

## 6. Exact simulation commands

Activate and enter project:
```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
```
Short check:
```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --headless --envs 4 --iterations 2 --device cuda:0
```
Bounded fresh diagnostic training (only after checking no duplicate process):
```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --headless --envs 32 --iterations 500 --device cuda:0 --dashboard
```
GUI alternative, not simultaneous:
```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --viz newton --envs 32 --max_visible_envs 32 \
  --iterations 500 --device cuda:0 --dashboard
```
Evaluate an explicitly chosen new compatible checkpoint:
```bash
python scripts/eval_pipeline.py --task lift --curriculum fixed --policy rgb \
  --checkpoint REPLACE_WITH_ACTUAL_NEW_CHECKPOINT \
  --headless --episodes 50 --seed 101 --record-video \
  --output outputs/evaluations/rgb_fixed_seed101.json
```
Use --viz kit instead of --headless for GUI evaluation. Repeat with seeds202/303
and distinct output paths when assessing mastery. Do not name an uncreated
checkpoint as if it exists. Use actual run/model filenames.
Playback:
```bash
python scripts/play.py --task lift --curriculum fixed --policy rgb \
  --checkpoint REPLACE_WITH_ACTUAL_NEW_CHECKPOINT --num_envs 1 --viz newton
```
Scripted demo only:
```bash
python scripts/demo_scene.py --pan-demo --seconds 300
python scripts/demo_scene.py --headless --pan-demo --seconds 30
```
Snapshot:
```bash
SO101_POLICY=rgb SO101_CURRICULUM=fixed python scripts/capture_snapshot.py \
  --headless --width 424 --height 240 --output outputs/rgb_scene_reference.png
```
Physics grasp diagnostic:
```bash
SO101_POLICY=rgb SO101_CURRICULUM=fixed python scripts/checks/verify_grasp.py \
  --output outputs/pipeline/rgb_physical_grasp.json --device cuda:0
```
`--curriculum grasp` currently reads read_physical_report(), which validates
fingerprints from older physical_grasp.json/workcell evidence. Recent scene edits
make those stale. Do not bypass gates or promise the grasp curriculum launches;
refresh/adapt preflight evidence before using it.

## 7. Evaluations and unresolved simulation problem

Latest scripted report outputs/pipeline/rgb_physical_grasp.json (user scene
has subsequently been visually changed, so this report is historical/stale):
passed=False
fk_error_m=2.02469752734719e-07
minimum_hold_height_m=.4249999225139618
maximum_hold_distance_m=.060348670929670334
final_gripper_rad=9.995490472647361e-06.
A good IK/FK position calculation did NOT yield a physical held lift.
Log outputs/rgb_physical_grasp.log showed descent pushing the cube and empty
closure after attempted lift. Root cause is not established. Inspect actual vs
commanded joints, geometry, cube displacement/contact and timing, not only rewards.
The scripted verifier sets joint targets directly; it is not PPO evaluation.

Preserved legacy artifacts (NOT RGB-only):
artifacts/sim_to_real_candidates/best_observed_lift_3898/model_3898.pt
SHA256 e6a5acb47d27e5f5684649fb2e86a75609a73110c0fb1dcbbf6ccc3b4d2165f2
Current recorded baseline seed10147/50=.94,slide1.7132319019665454cm, after
camera freshness fix. It fails the <=1cm sliding criterion; no current3seed certificate.
Historical before-fix baseline142/150=.9466667 is not matched current certification.
artifacts/sim_to_real_candidates/experimental_latest_4397/model_4397.pt
SHA25672992abd02e12320a04e3d61d77d4b7c706407732c17804576ebba4bcfe515c4
Wrist-only before camera fix:seed10142/50=.84,20241/50=.82,30342/50=.84;
125/150=.8333333,mean slide3.20890432338179cm. Not promoted.
Same3898 wrist-only after-fix133/150=.8866667 failed gate. These controller/camera
conditions differ; do not compare as a controlled RGB study.
No verified placement or real learned grasp-and-lift success exists.

## 8. Real SO-101 and calibration (user-provided evidence)

Hardware computer: suhailck@suhailck, /home/suhailck/so101_pick_place,
conda/miniforge environment lerobot, Python3.12.
LeRobot0.6.2; installed under
/home/suhailck/miniforge3/envs/lerobot/lib/python3.12/site-packages;
editable checkout /home/suhailck/lerobot_ws/lerobot.
USB->SO101controller->motors. Current port/dev/ttyACM1; earlier/dev/ttyACM0.
User says same controller. Persistent /dev/serial/by-id path not provided.
Calibration ID my_awesome_follower_arm; remote path:
/home/suhailck/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json.
Local my_awesome_follower_arm.json matches byte-for-byte SHA256:
30cdcf43f2db004a9fe93472dd3527fbdff48fc44846663096fe15862435c8dd.

Calibration entries (all drive_mode0):
|Joint|ID|homing_offset|range_min|range_max|
|---|---:|---:|---:|---:|
|shoulder_pan|1|1934|766|3354|
|shoulder_lift|2|969|0|4093|
|elbow_flex|3|-303|4|4093|
|wrist_flex|4|-1567|33|2332|
|wrist_roll|5|-1583|0|4095|
|gripper|6|1463|2029|3545|
Motors FeetechSTS3215. Arm5joints DEGREES; gripper RANGE_0_100.
bus.connect(handshake:bool=True),disconnect(disable_torque:bool=True).
Robot config defaults use_degreesTrue, max_relative_targetNone,
disable_torque_on_disconnectTrue,P16,I0,D32,num_read_retries2.

Installed conversion code provided by user:
DEGREES: (raw-(range_min+range_max)/2)*360/(model_resolution-1).
Inverse: int(degrees*(model_resolution-1)/360+midpoint).
Gripper RANGE_0_100: bounded(raw-min)/(max-min)*100; inverse int(value/100*(max-min)+min).
4096 resolution => divisor4095, whereas older local calibration utility uses4096.
LeRobot degree midpoint is its coordinate convention; it is not automatically
simulation zero. Actual homing register is accounted for elsewhere; do not
blindly add homing_offset again. Sign/offset mapping to simulation remains unresolved.

robot.connect(calibrate=False) still calls configure(): torque-disabled context,
configure_motors(),writes Operating_Mode/P/I/D and gripper limits500/250/25.
Direct bus.connect(handshake=False) was used for register reads without configuration.
send_action()sync_writes Goal_Position and optionally limits relative target;
default max_relative_target=None does NOT bound new motor commands.

## 9. Exact hardware test outputs and what they establish

Initial raw home read (no movement):
```text
shoulder_pan1973;shoulder_lift2274;elbow_flex2030;
wrist_flex1947;wrist_roll2185;gripper2069
```
User stated arm at home and provided real-camera image. Conditional normalized
angles:pan-7.648°,lift20°,elbow-1.626°,wristflex67.209°,roll12.088°;
gripper2.639/100. This is a reference, not verified physical zero angles.
An interactive heredoc test failed EOFError at input() before target/torque
writes. Correct fix: save Python script and run it; heredocs consume stdin.

First pan trial, no settling pause:
START={pan:.967032967032967,lift:17.714285714285715,elbow:-1.7142857142857142,
wristflex:69.4065934065934,roll:11.56043956043956,gripper:.9894459102902375}
AT+3={pan:3.076923076923077,lift:17.36263736263736,elbow:-1.978021978021978,
wristflex:69.67032967032966,roll:11.736263736263735,gripper:.9894459102902375}
RETURN={pan:1.8461538461538463,lift:17.36263736263736,elbow:-1.978021978021978,
wristflex:69.75824175824175,roll:11.648351648351648,gripper:.9894459102902375}
Requested endpoint3.967032967°, measured change2.109890110°,return error.879120879°.

Pan trial with1second settling:
START={pan:.967032967032967,lift:18.593406593406595,elbow:-.13186813186813187,
wristflex:68.61538461538461,roll:1.5384615384615385,gripper:.9234828496042217}
END={pan:3.5164835164835164,lift:18.593406593406595,elbow:-.13186813186813187,
wristflex:68.52747252747253,roll:1.5384615384615385,gripper:.9234828496042217}
RETURN={pan:1.3186813186813187,lift:18.593406593406595,elbow:-.04395604395604396,
wristflex:68.61538461538461,roll:1.5384615384615385,gripper:.9234828496042217}
Measured change2.549450549°,endpoint error-.450549451°,return+.351648352°.
Other jointchanges<=.087912088°,gripper unchanged. User saw positive pan move
robot-right/toward-camera. This is not an exact image-left/right measurement.

Offline simulation FK comparison outputs/hardware_mapping/shoulder_pan_comparison.json:
+2.549° simpan =>tool displacement[-.2384650565,-6.2404699842,-.0000165596]mm;
image shift[+.3643041197,+2.6311313347]pixels;
camera-distancechange-4.1978784512mm. Direction is consistent with reported
camera-side movement; not a physical rollout or verified full calibration.

Old shoulder-lift file had incomplete substitutions. Two erroneous trials:
1) panstart1.934065934/lift20.087912088; printed endpoint3.934° and AT+2;
   lift remained20.087912088,measured movement0.
2) panstart1.846153846/lift20; printed endpoint3.846° and AT+2;
   lift remained20,return pan2.109890110,movement0.
Those do NOT establish failed motor direction; file calculation was wrong.

Corrected full single-joint file uses TEST_JOINT='shoulder_lift',CHANGE_DEGREES5,
RAMP_SECONDS3,RAMP_STEPS60,settle1s,tracking limit2° arm/2units gripper,
calibration-bound margin3°. It seeds fresh measured positions before enabling
all torque,ramps one joint,leaves torque unchanged on disconnect,NO automatic
return on error. Calibrated range checks do NOT check physical collisions.
Latest corrected result:
START={pan:2.10989010989011,lift:20.0,elbow:-.9230769230769231,
wristflex:68.35164835164835,roll:1.89010989010989,gripper:1.0554089709762533}
Requested lift endpoint25°. Calibrated range[-179.912,+179.912]°.
After MOVE:tracking error2.083° on shoulder_lift caused abort during ramp.
No endpoint/return readings; no positive/negative lift direction established.
Do NOT enlarge threshold or gains just to make test pass.

Latest read-only shoulder-lift diagnostic (user update during handover):
```text
Present_Position : 2274
Goal_Position : 2297
Torque_Enable : 1
Operating_Mode : 0
P_Coefficient : 16
I_Coefficient : 0
D_Coefficient : 32
```
23ticks=>23*360/4095=2.021978022° target-minus-current error.
Measured angle20°,target22.021978022°. Torque enabled,position mode;
remaining error confirmed. Cause unknown: lag/load/resistance/configuration
cannot be distinguished from these register reads. Do not modify gains blindly.
Arm may still hold its last targets; torque was intentionally NOT disabled.
No tests of elbow,wristflex,wristroll,gripper opening or verified motor bridge
have completed. Remote scripts are not automatically in this local repo.

## 10. Cleanup/cloud state

Recovery archive: /home/ai/Documents/ChatGPT/sim to real/project_cleanup_rgb_2026-10-12.
84items/~340MiB moved;36checkpoints hashed at that cleanup. More checkpoints
have been generated since;36is historical count, not current inventory.
No checkpoints deleted. Runtime grain textures and current reference retained.
Legacy helper dependencies retained; active RGB does not consume depth.
Private local Codex-history backup (~93.7MiB) exists outside public repo:
/home/ai/Documents/ChatGPT/sim to real/private_codex_history_2026-10-13/codex_history.tar.gz.
It is not a verified import into Cloud and may contain private pasted content.
Never publish it with project source. Auth files excluded.
User screenshot showed Cloud environment building; completion/GPU capability
not verified. Cloud has only pushed files, not all local source/history/USB access.

## 11. Agreed next steps: focus on training

1. Recheck processes and latest run log/checkpoints first; do not duplicate jobs.
2. Report diagnostic500run state honestly.8/500,S0is not mastery. Stopping cause
   unknown. The user explicitly authorized starting diagnostic simulation training.
3. If restarting, preserve existing run, use fresh compatible RGB config; don't
   resume older depth checkpoints or remove metadata checks. If resuming a new
   matching checkpoint, verify it exists and source/settings match.
4. Diagnose physical grasp failure in parallel with bounded diagnostic learning;
   no promise more PPO iterations will overcome a contact/control fault.
5. After bounded run,record50-episode fixed evaluations; inspect failure videos,
   actor gripper commands vs actual jaw angle,contact,dragging and held-lift checks.
6. If grasp fails,stop extending runs blindly: establish physical feasibility,
   then local grasp practice if its preflight evidence/gates are refreshed.
7. Once fixed lifting is reliable,randomize cube resets initially +/-1cm,then
   +/-2cm and wider VERIFIED reachable region; don't teleport during an episode.
8. Keep box fixed initially; later mild obstacle variation; randomize cube+box
   for placement only after randomized-lift gate passes.
9. Domain randomization around plausible real lighting/appearance/friction/mass/
   timing helps robustness; cannot substitute for joint/camera calibration.
10. Hardware checks can proceed separately; shoulder-lift tracking remains
    unresolved. Do not command real PPO until joint mapping,deployable input
    assembly and shadow inference have been verified. Never equate scripted
    movement/IK success with learned-policy success.

## 12. Suggested opening message for the next chat

Read docs/training_handover_2026-10-13.md and inspect current files before editing.
Focus on RGB-only SO-101 simulation training. Recheck latest training process/log
and compatible checkpoints. Preserve current scene and all checkpoints. The
scripted grasp test failed; the authorized500-update diagnostic run has only
8updates in its last observed log and no confirmed active process. Do not claim
mastery or real readiness. Explain causes/fixes simply, give headless/GUI commands,
and evaluate actual held lifts with recordings before progressing randomization.

## 13. Source snapshot fingerprints at handover

These identify inspected source/settings, not a commit or certification.
```json
{
  "src/isaac_vision_rl/utils/pipeline.py": "0ec41d5ec36637ea6c1f9489f2993173a74038e0a5a43749a92e12d77190e967",
  "src/isaac_vision_rl/utils/rl_agent.py": "0b19979ce2aa0a58aae0ded322a87181dc686d306ea8f6cb8516984a1f02996b",
  "src/isaac_vision_rl/tasks/pickplace/config/so101/env_cfg.py": "95826436fc146565f5a7041ccdb780b92c6e15a542b996333876b3274f24939a",
  "src/isaac_vision_rl/tasks/pickplace/config/so101/agents/rsl_rl_ppo_cfg.py": "ffcdb298806ff770aee20b4195f40f16aedd193c3938cde5558720e5485a75b4",
  "logs/rsl_rl/so101_picklift_rgb/2026-10-13_11-10-02/params/agent.yaml": "782e7cd34472918c4a983110f8679b1040fc9a0da9aaaabea85a0a34496f4312",
  "logs/rsl_rl/so101_picklift_rgb/2026-10-13_11-10-02/params/env.yaml": "892a473fbc351e43719589a699a968334a00761303565230a2c5646a049a79a4"
}
```
