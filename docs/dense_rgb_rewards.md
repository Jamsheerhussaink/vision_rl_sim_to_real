# Dense RGB lift rewards

The RGB configuration now adds bounded continuous feedback, alongside existing new-best progress bonuses:

| Term | Weight | Feedback |
|---|---:|---|
| dense_reach | 2 | Smooth cube-to-tool distance, 10 cm scale |
| dense_alignment | 3 | Lateral/vertical alignment (3.5/4.5 cm scales), downward tool orientation |
| approach_open | 0.5 | Open jaws during approach, fades near grasp alignment |
| stable_support | 10 | Both fingers contact cube, alignment, elevation and low cube speed |

Table displacement weight changes from -12 to -3 for RGB only. Drag, table collision, premature closure and action safety penalties remain. Strict two-sided sustained lift termination remains unchanged. Geometry is used for training reward only, not actor observations. Legacy vision/state configurations retain their existing rewards.

Isaac Lab multiplies reward rates by the control time step. These bounded dense terms accumulate with physical time, not simply with frame count. Existing progress bonuses retain their original scaling.

Proximity shaping can reward hovering: monitor independently measured held-lift success and compare it with proximity reward. The support term must never be interpreted as successful lifting by itself. Placement is still gated on randomized lift mastery.

Validation: 58 unit tests pass. A simulation-only smoke evaluation uses the existing model; it validates reward integration, not learning improvement. Existing checkpoints remain preserved. New training with these rewards is required before measuring improvement.

## Fresh 64-environment preset (2026-10-14)

`--profile rgb64` selects RGB fixed-position lift training from scratch,
64 environments and 2000 PPO updates. Explicit `--envs` and `--iterations`
override the resource/budget defaults. The profile rejects checkpoint resumes
and incompatible task/policy/curriculum settings. Runs are separated under
`logs/rsl_rl/so101_picklift_rgb_tuned64/`. Current scene and tuned rewards are
used; this is a new training preset, not a new actor architecture or automatic
reward-weight optimizer. Cube coordinates remain outside the actor.

```bash
python scripts/train.py --profile rgb64 --headless --device cuda:0 --dashboard
```

Newton GUI alternative:

```bash
python scripts/train.py --profile rgb64 --viz newton --max_visible_envs 4 \
  --device cuda:0 --dashboard
```

64 x 77 = 4928 transitions/update; 2000 updates collect 9,856,000 transitions.
The four-update headless resource check completed with peak measured GPU usage
98%, peak VRAM 8974 MiB and minimum available host RAM 4081 MiB. These samples
are in outputs/rgb64_resource_samples.json; they do not guarantee later memory
peaks or GUI memory usage. Leave headroom for temporary allocations and desktop
applications. Do not run evaluation concurrently. A free TensorBoard port from
6006 through 6015 is selected to preserve existing dashboards; read the printed URL.

Run the full experiment in the user's terminal. Checkpoint success must be
measured with completed held-lift evaluations; the short check did not train
or certify a successful lift policy.
