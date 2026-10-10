# isaac_vision_rl

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

This `isaaclab` branch contains the Isaac Lab SO-101 project. The evaluated
simulation baseline is included at
`artifacts/models/so101_lift_vision_3898/model_3898.pt`; see its
[model guide](artifacts/models/so101_lift_vision_3898/README.md) for playback,
evaluation commands, metadata and limitations. Evaluation summary reports are in
`artifacts/evaluations/`. Active training runs, full logs, videos and local tooling
history are excluded from this published snapshot.

See [project cleanup and retained evidence](docs/project_cleanup.md) for the
current checkpoints, maintained scripts and recovery archive.

The registered tasks are:

- `IsaacVisionRl-PickLift-SO101`
- `IsaacVisionRl-Pickplace-SO101`

## SO-101 lift training in the existing Conda environment

Use the [evidence-gated training pipeline](docs/training_pipeline.md) for new runs.
The latest close-when-centred controller and saved rollout evidence are described in
[camera grasp assistance](docs/camera_grasp_assist.md).
For the current contact-based reward correction and ±1 cm cube randomization,
follow [contact and randomized training](docs/contact_randomized_training.md).
The current simulation-only camera workflow is described in
[global camera training](docs/global_camera_training.md). The actor gets cube
geometry from rendered RGB-D, rather than the simulator's cube pose. Start fresh:
older state-assisted checkpoints are incompatible with these observations.
Start with `python scripts/train_pipeline.py --stage preflight`; placement is blocked
until randomized lift evaluation passes.

The workcell uses `env_final_sim.png` as its appearance reference: a mottled green
tabletop, black SO-101, yellow cube, brown box and three overhead area lights with ambient fill.
The box has a physical floor and walls in both lift and placement stages; it is
positioned behind the initial cube workspace during lift training. The robot resets to a folded home pose with open jaws and the cube visible.
Home angles in radians: shoulder pan -0.5, shoulder lift -1.05, elbow 1.35,
wrist flex 1.10, wrist roll 0.0, and gripper 1.20.
To capture the nominal lift scene without a GUI:

```bash
python scripts/capture_snapshot.py --task lift --headless --output outputs/lift_scene.png
```

This preview uses a larger image for inspection; policy RGB stays at 128 x 128.
The task camera and GUI use the original tabletop-only framing, with the green
surface filling the background and no room background or table edges in view.
GUI resets select `/World/envs/env_0/Camera`, the actual task sensor, rather than
an approximate perspective view. The physical camera housing is hidden to keep
it from obstructing the optical view. Policy RGB resolution stays at 128 x 128.
Scene appearance, camera and ready-pose changes require a fresh training run.

See [the complete agent and training guide](docs/agent_training.md) for the MDP,
actor/critic, dense rewards, entropy exploration, TD/GAE, metrics and evaluation.

The corrected action mapping spans the arm's soft joint limits with filtered position targets.
The gripper action maps `-1` to closed while both `0` and `+1` target fully open, so the
zero-mean initial policy does not close before reaching the cube. Premature closure is penalized
until the cube is centered between the fingers. The first lift milestone requires the cube
center above 0.47 m, near a closed gripper, moving slowly, for at least 0.5 seconds with cube-specific contact on both gripper bodies.
Inspect playback as well as the contact and held-lift metrics.

Approach training guides the tool directly toward the cube, without an overhead
waypoint or staged descent. Only new closest-distance progress earns reach feedback;
stationary proximity does not earn a separate reward. Premature closure and cube dragging remain penalized, and the gripper
starts open. Start fresh after changing this approach objective.

The first milestone is a camera-driven lift policy: the global RGB-D camera estimates
cube position, upright orientation and visibility. The actor receives these estimates
and robot proprioception. Simulator cube state is used only by training rewards,
the privileged critic and independent success scoring. Missing detections have an
explicit visibility flag; there is no simulator-position fallback.

Start a **fresh run** after this change. Old checkpoints use different observations,
action mapping and noise parameters. The agent rejects incompatible checkpoints.
Keep them for comparison; new checkpoints can be resumed or transferred between stages.

```bash
conda activate env_isaaclab
cd /home/ai/jamshi/isaac_vision_rl

# Short startup and training check without a window
python scripts/train.py --task lift --headless --envs 4 --iterations 2 --device cuda:0

# Fresh training run without a window, with TensorBoard
python scripts/train.py --task lift --headless --envs 64 --iterations 2000 --device cuda:0 --dashboard

# Alternative: train with the simulator window (start with fewer environments)
python scripts/train.py --task lift --viz kit --envs 4 --iterations 2000 --device cuda:0
```

Logs and checkpoints are saved to `logs/rsl_rl/so101_picklift_vision/<timestamp>/`.
At `http://localhost:6006`, inspect `Episode_Reward/grasp_cube`,
`Episode_Reward/lift_cube`, and `Episode_Termination/lift_success` alongside total reward.
The installed termination manager reports mean activation across environments; use this
as a training diagnostic and measure episode success separately in playback. A larger
total reward alone does not establish successful lifting.

After training, replace the path below with a checkpoint from the **new** run:

```bash
python scripts/play.py --task lift --checkpoint logs/rsl_rl/so101_picklift_vision/NEW_RUN_TIMESTAMP/model_1999.pt --num_envs 1
```

Confirm that the fingers lift and hold the cube before proceeding to placement or real-robot work.

Quantitative evaluation counts complete episodes and saves success, reward, peak lift
and tabletop sliding measurements. Replace `NEW_RUN_TIMESTAMP` with the new run:

```bash
python scripts/eval_pipeline.py --task lift --checkpoint logs/rsl_rl/so101_picklift_vision/NEW_RUN_TIMESTAMP/model_1999.pt --episodes 50 --headless
```

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project was generated from an Isaac Lab source checkout. Its `pyproject.toml` uses editable relative paths to that
checkout, so regenerate the project or update `[tool.uv.sources]` if either directory moves.

Optional backends are available through extras. Pass the extra to each `uv run` command that needs it:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx
```

The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes. You can also select `ovrtx` independently when using
Newton physics with the OVRTX renderer. Commit `pyproject.toml` and `uv.lock` so collaborators use the same environment.

## Run the generated tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run python scripts/list_envs.py --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `list_envs.py`.

## Project structure

Task families live in `src/isaac_vision_rl/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/so101`. Add another robot by
creating a sibling of `config/so101`; add another task by creating a sibling task-family directory.

## Development

Run the generated registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and generates a git-ignored `pyrightconfig.json`. The generated
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
