# Project cleanup and retained evidence

Cleanup completed on 10 October 2026. Obsolete files were moved to a recoverable
archive, not permanently erased. This reduces clutter in the project; it does
not reclaim the archive's disk space.

Archive:
`/home/ai/Documents/ChatGPT/sim to real/project_cleanup_2026-10-10`

Its `manifest.json` lists every source, destination, reason and size, along with
the SHA-256 hashes of the protected checkpoints. All 263 planned moves completed.

## What remains and why

- `logs/rsl_rl/so101_picklift_vision/2026-10-10_09-44-58/`: completed baseline
  training run, configuration/provenance and TensorBoard history.
- `model_3898.pt`: the evaluated baseline and starting point for fine-tuning.
- `model_3850.pt`: the preceding checkpoint, retained for recovery/comparison.
- `outputs/evaluations/lift_3898*`: current baseline, full-smooth, wrist-only and
  hold-condition results, videos and traces. These are evidence needed to judge
  future improvements; a high training reward cannot replace them.
- `outputs/perception/`: RGB-D calibration and camera validation/reference data.
- `outputs/pipeline/physical_grasp.json`, `workcell_physics.json`, `reset_check.json`
  and `stability_after_final.json`: physical verification evidence. Fingerprinted
  reports can become stale after code/asset changes and must then be regenerated.
- The open/closed/raised grasp reference images and approved tabletop previews.
- All active package code, tests, teaching documents, robot/table/box/cube source
  assets and `env_final_sim.png`.

The log folder decreased from approximately 21 GB to 60 MB. Outputs decreased
from about 290 MB to 160 MB. The external archive contains about 21 GB.

## Which scripts to use

| Task | Maintained script |
|---|---|
| Train manually | `scripts/train.py` |
| Run staged training/preflight | `scripts/train_pipeline.py` |
| Evaluate and measure strict success | `scripts/eval_pipeline.py` |
| Watch a checkpoint | `scripts/play.py` |
| Record successful and failed episodes | `scripts/record_episodes.py` |
| Verify workcell physics | `scripts/check_workcell.py` |
| Verify reset/stability | `scripts/check_resets.py`, `scripts/check_stability.py` |
| Verify physical grasp | `scripts/verify_grasp.py` |
| Verify RGB-D perception | `scripts/verify_camera_perception.py`, `scripts/check_randomized_camera.py` |
| Capture the current scene | `scripts/capture_snapshot.py` |
| Build assets/texture | `scripts/build_robot_asset.py`, `scripts/build_table_texture.py` |
| Prepare future real-camera work | `scripts/calibrate_camera_points.py`, `scripts/realsense_perception.py` |

`scripts/test_physics.py` was archived because it uses the older launcher/data
interfaces and overlaps the maintained workcell checks. `verify_sim_vs_real.py`
constructs an older alternate scene instead of the current environment. Its only
remaining script reference was inside `upload_to_notion.py`, an optional older
documentation-upload utility; that utility was archived with it.

The archived material also includes obsolete state/placement/vision training runs,
intermediate baseline checkpoints, old previews and rollout experiments,
conversation history, temporary validation dependencies, Python bytecode and tool
caches, and the unused decompiled robot export. No Git metadata was removed.
`.waylog/` is now ignored as local tooling history.

## Recovering a file

The archive preserves project-relative paths. For example, the old optional
Notion uploader is at:

`/home/ai/Documents/ChatGPT/sim to real/project_cleanup_2026-10-10/scripts/upload_to_notion.py`

Check the original destination before restoring it so newer work is not
overwritten. Prefer restoring only the required file/run rather than returning
all historical artifacts to the project.

## Retention for future runs

1. Preserve active run directories and checkpoints while training is running.
2. After evaluation, retain the chosen checkpoint, a recovery checkpoint and
   the run's configuration/TensorBoard provenance.
3. Retain strict evaluation JSON and representative success/failure recordings
   for every controller/model comparison still under investigation.
4. Archive obsolete smoke runs and intermediate models only after checking that
   current commands and evidence gates do not depend on them.
5. Do not delete this archive merely to tidy the project again. Permanent removal
   is a separate decision because it erases the only retained copies of old work.

Post-cleanup validation: 37 unittest cases, eight additional function checks
(including registration), and syntax parsing of all retained Python source files
passed. Both protected checkpoint hashes match the pre-cleanup inventory.
These checks verify project integrity; they do not prove real-robot readiness.
