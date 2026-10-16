# Earlier scripts

`scripts_20261015.tar.gz` preserves seven scripts removed from the active script
list on 2026-10-15. These are earlier demos and troubleshooting tools, not the
current RGB training/evaluation workflow:

- `align_wrist_roll.py`: manual wrist zero alignment and calibration editing.
- `park_home.py`: fixed encoder-tick parking routine.
- `demo_mirror_motion.py`: scripted simulated/real arm motion demo.
- `demo_scene.py`: scene viewer and scripted pan demo.
- `test_pan_motion.py`: diagnostic from the shoulder-pan investigation.
- `check_sim_real_mirroring.py`: earlier combined mirroring diagnostic.
- `compare_sim_real_video.py`: scripted camera/motion comparison experiment.

The archive contains the exact original files under their original `scripts/`
paths. `scripts_20261015_manifest.json` records SHA-256 checksums for all 27
original scripts and the 14 supporting-tool moves. Archived files have been
verified against their original bytes.

To inspect the archive from the repository root:

```bash
tar -tzf archive/scripts_20261015.tar.gz
```

To recover one tool at its original path without overwriting an existing file:

```bash
tar -xkzf archive/scripts_20261015.tar.gz scripts/test_pan_motion.py
```

Review recovered hardware routines before use; some move motors or edit
calibration. Archiving does not certify their behavior. Older handover/debugging
documents retain historical commands for these archived tools.
