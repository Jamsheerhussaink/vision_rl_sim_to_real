# Project scripts

Run commands from the project root after `conda activate env_isaaclab`.

## Everyday work

| Script | Purpose |
| --- | --- |
| `train.py` | Train an RGB PPO policy |
| `play.py` | Watch or record a checkpoint |
| `eval_pipeline.py` | Measure complete episodes, held lifts, and table sliding |
| `train_live_session.py` | Run bounded training followed by evaluation and status output |
| `train_pipeline.py` | Run physical preflight and enforce staged evaluation gates |
| `capture_snapshot.py` | Capture the current simulated task camera |

```bash
# Headless bounded training and evaluation; simulation only.
python scripts/train_live_session.py --envs 4 --iterations 2 --episodes 2 --headless

# Watch an explicit checkpoint in the GUI.
python scripts/play.py --task lift --curriculum fixed --policy rgb --checkpoint PATH_TO_CHECKPOINT

# Capture the task camera without a GUI.
python scripts/capture_snapshot.py --headless
```

Use the main README for full training and evaluation commands. Run one simulator
at a time. Completing the bounded workflow does not establish grasp/lift success.

## Supporting tools

| Folder | Scripts | Purpose |
| --- | --- | --- |
| `checks/` | `check_workcell.py`, `check_resets.py`, `verify_grasp.py`, `check_stability.py` | Collision, reset, scripted grasp feasibility, and finite-state checks |
| `checks/` | `check_gripper_contact.py`, `check_gripper_surface.py` | CAD/contact inspection and offline trace comparison |
| `checks/` | `list_envs.py` | List registered environments |
| `hardware/` | `realsense_rgb_preview.py` | Real camera preview without motor commands |
| `hardware/` | `camera_placement_guide.py` | Live RGB with simulation ghost for manual visual alignment |
| `hardware/` | `hardware_bridge.py`, `check_hardware_signs.py`, `probe_connection.py` | Motor bridge, joint mapping, and read-only connection checks |
| `hardware/` | `mirror_evaluation.py` | Mirror simulation poses on hardware; uses simulated camera observations |
| `hardware/` | `real_camera_policy.py` | Standalone policy from real RGB and encoder feedback; shadow mode by default |
| `hardware/` | `return_policy_home.py` | Plan a bounded nearby return to checkpoint home; execution is explicit |
| `hardware/` | `return_photo_home.py` | Plan or execute a bounded return to the measured physical home; no checkpoint required |
| `assets/` | `build_robot_asset.py`, `build_table_texture.py` | Rebuild robot/table assets when needed |

For example, the camera preview is now:

```bash
python scripts/hardware/realsense_rgb_preview.py --headless
```

For a manual placement guide, close the other camera preview and run:

```bash
python scripts/hardware/camera_placement_guide.py
```

Cyan outlines and the transparent ghost come from
`outputs/rgb_scene_reference.png`. Match camera framing first, then the robot
base/footprint and objects; the arm outline also depends on its joint pose.
This is an image alignment guide, not measured physical localization.
Press `g` to toggle the ghost or `q` to close. No motors are commanded.

Hardware tools can move motors depending on their options. For bridge startup,
see [the startup guide](../docs/bridge_startup_checks.md).

`scripts/demo_mirror_motion.py` remains for the measured-home and table-clearance
workflow described in [the clearance guide](../docs/photo_home_and_clearance.md).
The old archive and superseded `compare_pick_place.py`, `park_home.py`, and
`stream_to_robot.py` were removed in the [core cleanup](../docs/project_cleanup.md).
