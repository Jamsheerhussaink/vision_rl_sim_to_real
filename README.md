# SO-101 RGB-only pick-and-lift

Active goal: learn approach, gripper closure and stable held lift from RGB images
plus joint/controller readings. Depth and simulator cube coordinates do not enter
the actor. Simulator state is used for the training critic, rewards and evaluation.

## Current scene

RGB 424x240 at a nominal 60 Hz; physics 240 Hz. Aspect-preserving 128x128 actor
input. Camera is 30 cm above the tabletop with a nominal D455 pinhole projection,
not measured calibration. The current approved preview is
[outputs/rgb_scene_reference.png](outputs/rgb_scene_reference.png).

## Training

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --headless --envs 32 --iterations 500 --device cuda:0 --dashboard
```

GUI alternative; run only one training process:

```bash
python scripts/train.py --task lift --curriculum fixed --policy rgb \
  --viz newton --envs 32 --max_visible_envs 32 \
  --iterations 500 --device cuda:0 --dashboard
```

Old RGB checkpoints have different timing/image profiles. Preserve them; start
fresh for this profile. No successful lift is certified by a smoke test.

## Evaluate an explicit new checkpoint

```bash
python scripts/eval_pipeline.py --task lift --curriculum fixed --policy rgb \
  --checkpoint PATH_TO_NEW_CHECKPOINT --headless --episodes 50 --seed 101 \
  --record-video --output outputs/evaluations/rgb_fixed_seed101.json
```

For visual evaluation replace `--headless` with `--viz kit`. Evaluate fixed lifting
before widening cube randomization. Placement follows reliable randomized lifts.

## Real-camera preview, when connected

`python scripts/hardware/realsense_rgb_preview.py` opens raw RGB and actor-image windows.
Add `--headless` to print profile/intrinsics/frame-rate information without windows.
This script does not move motors or run the policy. Real deployment still requires
verified joint mapping, observations, timing and a robot control bridge.

## Mirror evaluation movements on hardware

Use [scripts/hardware/mirror_evaluation.py](scripts/hardware/mirror_evaluation.py) to run an explicit
checkpoint from the simulated RGB camera and copy its joint positions to SO-101.
It defaults to mock motors; `--execute` enables real movement. Commands, startup
requirements, tracking limits and shutdown behavior are documented in
[docs/mirror_evaluation.md](docs/mirror_evaluation.md).

## Retained project tools

The six everyday entry points stay in `scripts/`: `train.py`, `play.py`,
`eval_pipeline.py`, `train_live_session.py`, `train_pipeline.py`, and
`capture_snapshot.py`. Supporting tools live in `scripts/checks/`,
`scripts/hardware/`, and `scripts/assets/`.

See [scripts/README.md](scripts/README.md) for the full tool list and commands.
Earlier demo and troubleshooting scripts are stored in a recoverable
[archive](archive/README.md).

Required runtime textures are in usd/. Checkpoints and curated model artifacts
are preserved. See [docs/project_cleanup.md](docs/project_cleanup.md) for recovery.
[docs/ppo_pipeline_explained.md](docs/ppo_pipeline_explained.md) and the RL study notes
include historical pipeline details; the RGB guide describes the active variant.
