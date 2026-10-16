# RGB-only PPO baseline

## Current camera framing (2026-10-13)

RGB task camera is now at (-0.08, -0.26, 0.75) m: 0.35 m above
the 0.40 m tabletop, lowered another 0.10 m. It aims toward (-0.08, 0.10, 0.25) m, using
OpenGL axes and XYZW quaternion (0.3069736622, 0, 0, 0.9517180101).
The Newton overview uses the same eye and aim point. This supersedes the
older mounting approximations below. Resolution, nominal 90-degree horizontal
field of view, image preprocessing and robot physics are preserved.
The lower view makes the cube larger than with the previous 0.65 m height. Start a fresh run
for this view; preserve older checkpoints and evaluate grasp/lift performance.

Current RGB layout: robot base (-0.28, -0.0363932023, 0.405) m,
30 cm horizontally from the camera and on the image's left side, facing +X.
Cube (-0.03, -0.1163932023, 0.426) m and box
(-0.03, 0.1236067977, 0.401) m are 25 cm forward of the base.
Their relative positions to the arm are preserved from the previous layout.
The 30 cm distance is measured horizontally, not along the camera/base 3D line.

## Revised approach reward (2026-10-13)

RGB approach progress now cancels RewardManager's timestep scaling, as grasp
and lift progress already do. Each new best 1 cm approach earns 0.6 with
weight 3, at both 25 and 60 Hz, rather than 0.01 at 60 Hz. Stationary poses,
backtracking and returning to a previous best earn no additional credit.
Only the RGB workflow is retained. Start a fresh run to compare
this reward profile; keep earlier checkpoints for comparison.

The corrected scripted grasp verifier starts with open fingers, retains the
RGB camera profile and checks bilateral contact during a full 0.5-second hold.
The one-environment report outputs/pipeline/rgb_grasp_open_start_v2.json passed
with minimum cube height 0.47169 m and maximum tool distance 0.01333 m.
This is scripted feasibility evidence, not PPO mastery or a curriculum gate.
The trained policy still needs independent completed-episode evaluation.

Use 32 environments on this machine. Concurrent 64-environment training and
evaluation exhausted system RAM and swap; evaluate after stopping training.

```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --headless --envs 32 --iterations 500 --device cuda:0 --dashboard
```

Newton GUI alternative:

```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --viz newton --envs 32 --max_visible_envs 4 \
  --iterations 500 --device cuda:0 --dashboard
```

## D455 timing update

New RGB runs default to 60 Hz control/rendering and 240 Hz physics (four
substeps). This targets the documented D455 USB3 RGB 424x240/60 FPS mode,
not a measured live camera rate. RGB capture resolution is now 424x240 as
specified by the user; shared actor_rgb preprocessing resizes to 128x72
and pads 28 rows above and below to produce 128x128 without stretching.
The real RGB preview uses the identical function. Viewpoint and lens projection remain
uncalibrated and are not yet matched to the D455. Previous RGB checkpoints
used a square 512x512 capture; these checkpoints are retained but rejected
by the new image-profile check. Use their saved source snapshots to reproduce
the old framing; start fresh for the new profile.

To preserve approximately the original physical-time learning/filter horizons:
gamma=0.99^(25/60)=0.995821, lambda=0.95^(25/60)=0.978855,
rollout length=77 steps (about 1.283 seconds), EMA alpha=1-0.8^(25/60).
The six-second deadline now corresponds to 360 policy steps. Reward rates
remain timestep-scaled; individual event/progress contributions consequently
change with timestep and must be evaluated, not assumed equivalent.

The first 500-update RGB run used 25 Hz and is preserved. Reproduce its
training/evaluation timing with `SO101_RGB_FPS=25` before the command.
New checkpoints record their frequency; loading mismatched timing is rejected.
Use fresh runs to compare frequencies. No real camera stream has been measured.
Reference: https://realsenseai.com/wp-content/uploads/2024/10/Intel-RealSense-D400-Series-Datasheet-October-2024.pdf

`--policy rgb` (the default in train/play/eval) uses RGB images plus joint positions,
joint velocities, previous actions, filtered targets and episode time. It renders
only RGB: no depth image, cube-coordinate cue, capture latch, automatic release,
or latch-dependent arm pause enters the actor/controller. Gripper actions map
-1 to closed, 0 to half open and +1 to fully open. The actor learns closure.

Simulator state remains available only for rewards, strict held-lift success and
the privileged training critic. This is asymmetric actor-critic training, not
privileged information fed to deployed actions. Existing dense shaping, penalties
and strict sustained bilateral-contact success are retained.

Only `--policy rgb` is supported. Depth-assisted checkpoints and utilities are removed from the active project.

The first bounded run starts from scratch, fixed cube and home pose, with 500
updates. Evaluate afterward; do not infer success from reward or smoke tests.
Use progressively wider cube randomization only after reliable fixed lifting.
No measured D455 pose/intrinsics are available yet: this run uses the existing
simulated camera. It is not yet matched to the lower real RGB viewpoint.

Headless:
```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --headless --envs 32 --iterations 500 --device cuda:0 --dashboard
```

GUI alternative (do not launch alongside the existing headless training):
```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --viz newton --envs 32 --max_visible_envs 32 \
  --iterations 500 --device cuda:0 --dashboard
```

Logs/checkpoints: `logs/rsl_rl/so101_picklift_rgb/<run>/`.
Evaluate an explicit new checkpoint with:
```bash
python scripts/eval_pipeline.py --task lift --curriculum fixed --policy rgb \
  --checkpoint logs/rsl_rl/so101_picklift_rgb/REPLACE_RUN/model_499.pt \
  --headless --episodes 50 --seed 101 --record-video \
  --output outputs/evaluations/rgb_fixed_seed101.json
```
For visual evaluation replace `--headless` with `--viz kit`.

## Real RGB preview when connected

Run `python scripts/hardware/realsense_rgb_preview.py` for capture and actor-image windows.
Add `--headless` for profile/intrinsics/timestamp-rate output without windows.
This uses only RGB and never commands motors or runs the policy. A disconnected
camera cannot be hardware-verified.

The camera retains its original horizontal aperture and pose; its vertical
aperture now explicitly follows 424:240 aspect ratio. This is a consistent
generic pinhole projection, not measured D455 optics. No tabletop or physics
geometry was moved merely to fit image dimensions.

## Real-photo visual approximation

### 2026-10-16 appearance update

The current appearance is configured in `configure_curriculum()` in
`src/isaac_vision_rl/utils/pipeline.py`, overriding the base scene defaults.
The D455 screenshot supplied on this date was used for a visual approximation:
stronger slightly cool room fill, smaller side lights producing white tabletop
reflections, warmer cardboard/cube colors, and table roughness reduced from
0.55 to 0.28. Camera framing, object poses, collision geometry, mass, friction,
and controllers were preserved from the working configuration at the start
of this update. Earlier layout descriptions below are historical.

The previous configuration and render are recoverable in
`outputs/appearance_backup_20261016/`. The fresh reference is
`outputs/rgb_scene_reference.png`. Reproduce it with:

```bash
PYTHONPATH=src:. /home/ai/miniconda3/envs/env_isaaclab/bin/python scripts/capture_snapshot.py --headless --output outputs/rgb_scene_reference.png
```

This is a lighting/material approximation from a screenshot, not calibrated
photometry. Cable geometry, cardboard creases, and real sensor noise remain
unmatched. Existing checkpoints require new evaluation under this appearance;
no training or task-success claim is implied by the rendered snapshot.

### Historical appearance notes

RGB scene camera is at (0.02,-0.27,0.78), aimed toward (0.02,0.10,0.45),
with focal length 20 and aspect-matched aperture. Robot base moves 7 cm right
to (-0.18,0.10,0.405); cube/box locations and all collision/mass/friction
settings are preserved. This changes relative reach geometry and needs a
new grasp feasibility evaluation before full training. RGB-only table material
uses table_real_rgb.usda with brighter green and reduced roughness; lighting
is brighter and cube colour warmer. outputs/rgb_scene_reference.png is the
verified reset view. This is a visual approximation, not measured calibration.
Real cables, cardboard wrinkles, and sensor noise are not reproduced exactly.

Only the current scene reference outputs/rgb_scene_reference.png and its render log are retained; superseded scene previews are removed. Training/evaluation records, checkpoints and runtime textures are retained.

RGB fixed/lift reset home angles (rad): shoulder pan 0.00359878, shoulder lift -1.25, elbow 1.60, wrist flex 0.85, wrist roll 0, gripper closed 0.0. These approximate the photo, not measured hardware home angles. Box grain is visual only; box_grain.png is a required runtime texture.

Robot base yaw is now zero (horizontal footprint in the task camera). Shoulder pan compensates the former -30 degree base yaw to preserve the arm world heading.

## Approximate measured camera mounting

Interpreted user measurements as 0.25 m horizontal camera-to-base-centre
distance and 0.35 m above the tabletop. With tabletop Z=0.40 m, camera
Z=0.75 m. Camera XY=(-0.06112074,-0.11992663), robot XY=(-0.18,0.10).
Aim point=(-0.04,0.12,0.45); focal length 17 is a visual framing approximation,
not measured D455 intrinsics. The supplied distances do not determine camera
yaw/pitch or lens projection; use physical calibration for real deployment.

Camera height subsequently reduced to 0.30 m above tabletop (world Z=0.70 m), retaining 0.25 m horizontal distance and re-aiming at the same workspace point.

## Nominal D455 approximation (no device calibration)

Horizontal FOV 90 degrees from D455 nominal RGB specifications; square-pixel
424x240 pinhole assumption gives vertical FOV 59.023 degrees. Focal length
12.44 and horizontal aperture 24.88 define this projection. Centred principal
point, no lens distortion; neither is factory calibration. Downward tilt
51.313 degrees makes the top-centre ray hit tabletop 0.75 m forward from
the point below the camera at 0.30 m height. Horizontal camera/base distance
remains 0.25 m. This model predicts about 1.406 m width at the far boundary,
not exactly the approximate 1.20-1.30 m width reported by the user; those
measurements and nominal FOV are not forced to match simultaneously.
Source: https://www.realsenseai.com/compare-depth-cameras/

Latest visual adjustment: robot base moved 2 cm left to (-0.20,0.10,0.405), camera held fixed. Camera/base horizontal distance consequently becomes about 26.01 cm rather than the earlier 25 cm target.

Latest mounting correction: camera is 0.30 m horizontally from the robot-base centre and 0.30 m above tabletop. Camera moved outward along the existing mounting direction; robot placement and optical orientation are preserved.

Latest real-photo angle refinement: pitch 48 degrees downward; aim yaw 11 degrees from scene +Y toward +X. Camera position and object placement preserved. This supersedes the earlier 75 cm far-boundary fit; the new nominal top-centre intersection is approximately 90 cm forward. Orientation is visually estimated, not measured.
