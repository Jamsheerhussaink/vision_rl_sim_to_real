# Project cleanup

## Script organization — 2026-10-15

Reduced the top-level script list from 27 to six everyday entry points. Moved
seven checks into `scripts/checks/`, five hardware tools into `scripts/hardware/`,
and two asset builders into `scripts/assets/`. Updated pipeline calls, evidence
fingerprint paths, the bridge test import, and documentation for those moves.

Seven earlier demo/troubleshooting scripts are stored byte-for-byte in
`archive/scripts_20261015.tar.gz`; recovery instructions and the complete list
are in [archive/README.md](../archive/README.md). The manifest there records
original checksums and relocated paths. Checkpoints, evaluation outputs,
calibration, scene assets, and training logic were not changed by this step.

Saved curriculum gates fingerprint source files and script paths. Existing gates
can therefore become stale after this reorganization; reevaluate a checkpoint
before advancing a gated stage. Recorded reports remain preserved.

See [scripts/README.md](../scripts/README.md) for current commands and locations.

Validation: all 60 existing tests passed (59 inside the sandbox; the localhost
UDP loopback test passed separately with socket access). All 13 checked CLI
entry points accepted `--help`, active Python files parsed, the evidence
fingerprint resolved every file, and archived contents matched their checksums.
No training, simulation rollout, asset rebuild, or physical motor command was run.

Repeat the tests from the project root in the Isaac Lab environment:

```bash
PYTHONPATH=src:. python -m pytest -q
```

## RGB-only cleanup — 2026-10-14

Active training, playback, evaluation and hardware-streaming entry points accept
only `--policy rgb`. The actor receives RGB plus joint/controller readings;
simulator object state remains available to rewards, the critic and evaluation.

Removed from the active project:
- RGB-D cube estimator and perception package.
- Camera-grasp latch, automatic jaw override and capture-based arm hold.
- Depth-camera verification and camera-assisted pick/lift scripts.
- Depth/state-only tests and obsolete depth diagnostics in evaluation traces.
- Duplicate legacy episode recorder; eval_pipeline.py records RGB evaluation MP4s.
- Unused SKRL agent configuration; RSL-RL PPO remains registered.
- logs/rsl_rl/so101_picklift_vision, including its 15 depth-assisted checkpoints.
- Legacy depth model/evaluation artifacts and copied source snapshots.

Retained:
- train.py (including rgb64 preset), play.py and eval_pipeline.py.
- RGB staged pipeline with success gates and physical preflight checks.
- Snapshot/demo, workcell/reset/stability/grasp checks and asset builders.
- RealSense RGB preview, hardware bridge/client, joint-calibration and tracking checks.
- RGB checkpoints, tuned rewards, current camera/scene, runtime textures and assets.
- Tests covering PPO/GAE, RGB input, contact/lift logic, physics, curriculum gates,
  controller filtering and hardware bridge behavior.

The cleanup manifest is outputs/rgb_cleanup_20261014.json. Recovery archives are
outside the project in /tmp/so101_rgb_cleanup_20261014/recovery.tar.gz and
legacy_outputs.tar.gz. /tmp is temporary storage, not a durable backup. The
recovery archive also includes pre-cleanup source, tests, scripts and docs.

All 51 retained tests passed, including a localhost UDP test with no robot motor
commands. The latest RGB checkpoint is being evaluated after cleanup to verify
checkpoint/controller compatibility. Test passes alone are not task mastery.

Training examples (run only one simulator at a time):

```bash
python scripts/train.py --profile rgb64 --headless --device cuda:0 --dashboard
python scripts/train.py --profile rgb64 --viz newton --max_visible_envs 4 --device cuda:0 --dashboard
```

For interrupted runs, use ordinary train.py flags with an explicit checkpoint;
the rgb64 preset intentionally starts fresh. Evaluate separately from training
because concurrent simulators previously exhausted system RAM and swap.
