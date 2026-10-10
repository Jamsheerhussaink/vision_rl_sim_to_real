# Evaluated SO-101 vision lift checkpoint

`model_3898.pt` is the completed baseline, not the currently running wrist-only
fine-tuning result. Model metadata and its SHA-256 are in `metadata.json`.
Summary evaluation reports are in `artifacts/evaluations/`.

Use the matching code in this commit. The actor uses global rendered RGB-D cube
geometry, RGB images and robot/controller measurements. Closure is assisted by a
camera-capture latch; this is not evidence of fully learned jaw control.

The three-seed baseline comparison recorded 142/150 strict held-lift successes
(94.7%). Wrist-only control recorded 133/150 (88.7%) and reduced wrist rotation.
Repeat evaluations varied despite identical seeds. This checkpoint is a simulation
baseline and is not certified for placement or execution on a real robot.

From the project root in the existing Isaac Lab Python environment:

```bash
python scripts/play.py --task lift --curriculum lift --policy vision \
  --checkpoint artifacts/models/so101_lift_vision_3898/model_3898.pt \
  --num_envs 1 --cube-jitter 0.01 --viz kit
```

Recorded headless evaluation (choose a new output filename for each run):

```bash
python scripts/eval_pipeline.py --task lift --curriculum lift --policy vision \
  --checkpoint artifacts/models/so101_lift_vision_3898/model_3898.pt \
  --episodes 50 --seed 101 --cube-jitter 0.01 --headless --record-video \
  --output outputs/evaluations/published_baseline_seed101.json
```

For visible evaluation replace `--headless` with `--viz kit`. For wrist-only
evaluation add `--wrist-only-control`. Do not combine it with `--smooth-control`.

Historical log paths in project study notes describe the local development
machine; those generated logs are not included in this repository snapshot.
