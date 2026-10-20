# Core project cleanup — 2026-10-17

Removed 478 files and two obsolete script aliases (391.67 MiB), including all 69 existing videos, bulk evaluation
traces, duplicate screenshots, texture/appearance backups, obsolete diagnostics,
unused CAD/source exports, duplicate imported models, caches, and the earlier
script archive. This cleanup deletes those files; it does not create recovery
copies. Git history was not rewritten.

Removed superseded scripts: `scripts/hardware/compare_pick_place.py`,
`scripts/hardware/park_home.py`, and `scripts/hardware/stream_to_robot.py`.
Removed the older cloud handoff, training handover, and pan-investigation notes.

Retained:

- Training, playback, completed-episode evaluation, staged preflight, snapshots,
  physical checks, asset builders, and their tests.
- Current real-camera policy, mock/default evaluation mirror, hardware bridge,
  read-only probe, camera preview/alignment, joint mapping, and checked home tools.
- All 89 model/checkpoint/export files, their run settings, training logs, and
  saved provenance. Checkpoint deletion requires a separate explicit instruction.
- Calibration, measured-home records, current camera reference images,
  runtime textures, source robot USD for the asset builder, and all imported
  payload layers required by the scene and saved checkpoint configurations.
- Compact evaluation/real-policy summaries, physical-grasp/workcell reports,
  and the latest audit changes/evidence. A retained report does not imply that
  the current configuration passes a gate or that the robot learned the task.
- Current hardware/training guides and the PPO and robotics study notes.

Only required payload files remain under `urdf/`; the original URDF conversion
sources and unused CAD meshes were removed after resolving USD dependencies.
The older `robot_grasp_calibrated.usda` is retained because a saved checkpoint
configuration references it. The unused `robot_grasp_surface_v2.usda` is removed.

Older references to recordings in retained guides are historical; those videos
and per-frame traces were deleted by this cleanup. New runs can still record
videos using their existing options. No recording defaults or policy behavior
were changed.

The exact removed paths, sizes, checksums, and protected-file checksums are in
[core_cleanup_manifest_20261017.json](core_cleanup_manifest_20261017.json).
See [the script guide](../scripts/README.md) for current commands.

Verification commands (from the project root):

```bash
conda activate env_isaaclab
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider
```

The UDP loopback test requires permission to open a localhost socket. Verification
also checks retained file hashes, USD dependencies, Python syntax, the evidence
fingerprint, and the absence of video files. No simulator rollout, training,
asset rebuilding, or real motor command is part of this cleanup.

Verification completed: 105 tests passed; all 89 model/checkpoint/export files
were unchanged, required USD dependencies remained available, and no videos
remained. Concurrent work updated source/tests and physical-check reports and
created new grasp-audit images during cleanup; those changes were preserved.
Caches can also regenerate while other Python processes are running. The test
result describes the suite at the time it ran, not a lock on concurrent edits.
