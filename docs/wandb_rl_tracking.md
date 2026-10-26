# W&B for SO-101 training review

The new simple-reward and lesson-curriculum launchers currently save TensorBoard event files locally. Those event files already include the stage fractions, individual reward terms, bilateral finger contact, table contact and clearance, cube sliding, PPO losses, and action noise. The W&B option sends those same scalar metrics and run configuration to a W&B project so the W&B MCP can read and compare runs.

W&B logging is opt-in. It does not change the policy, reward values, stage gates, physics, or checkpoint selection. The metrics-only writer leaves git diffs, checkpoints, and videos on this computer. W&B still receives scalar metrics and the run configuration.

The older `scripts/mcp_server.py` reward editor uses a separate legacy weight list. The simple-reward experiment keeps its weights in `src/isaac_vision_rl/experiments/simple_rewards/stages.py`, so do not use that old editor to tune this curriculum. Use the W&B MCP to inspect and compare runs; change reward weights in the experiment source only after reviewing the evidence.

## One-time account setup

Create or copy an API key from <https://wandb.ai/authorize>. Keep the key private. In the terminal that will launch both Codex and training, enter it without putting the key in shell history:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
printf '\n'
```

The W&B SDK and `uv` are installed in `env_isaaclab`. For a fresh environment, install them with `python -m pip install wandb uv` after activating that environment.

The Codex MCP server uses this environment variable. Fully restart Codex from that environment after setting it so the local W&B MCP can inherit the key. The MCP is configured for `models-only` and `read-only`; it can inspect W&B runs but cannot modify them.

## Start training with W&B

Use `--wandb` and optionally select a project name. For example:

```bash
cd /home/ai/jamshi/isaac_vision_rl
PYTHONPATH=src:. /home/ai/miniconda3/envs/env_isaaclab/bin/python scripts/train_curriculum.py \
  --lesson approach --task place --policy rgb --curriculum place \
  --reward-profile four-stage --reward-style continuous \
  --grasp-reward-mode overlap --training-goal place \
  --lift-rise-min 0.10 --lift-rise-max 0.12 --lift-hold-seconds 0 \
  --lr-schedule adaptive --cube-jitter 0 \
  --envs 16 --iterations 500 --device cuda:0 \
  --headless --wandb --wandb-project so101-isaaclab-rl
```

For Newton and the live reward graph, replace `--headless` with:

```bash
--viz newton --gui --max_visible_envs 1 --live-reward-graphs
```

Use the same W&B flags with later lessons (`--lesson grasp`, `lift`, or `place`). Existing TensorBoard runs are not copied automatically; this sends data only from a new run launched with `--wandb`.

## Ask the MCP to inspect a run

After Codex restarts with the API key available, ask it to read the latest run in `so101-isaaclab-rl` and compare `FourStage/REACH_fraction`, `FourStage/GRASP_fraction`, `Grasp/bilateral_contact_fraction`, `Episode_Termination/unsafe_table_contact`, `FourStage/slide_distance_m`, `Loss/action_saturation`, and `Loss/explained_variance`. Stage fractions are training diagnostics, not independent evaluation success rates. Use recorded evaluation episodes and videos before saying grasping or placement improved.
