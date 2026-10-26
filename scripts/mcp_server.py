#!/usr/bin/env python3
"""
Isaac Vision RL - Custom MCP Server
Exposes specialized tools for inspecting RL training, evaluation metrics,
robot arm joint configs, and hardware/GPU status to Antigravity.
"""

import os
import sys
import glob
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

# Initialize MCP Server
server = MCPServer("isaac-vision-rl-assistant")

# Resolve project workspace directory
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/home/ai/jamshi/isaac_vision_rl")).resolve()


def _get_workspace_path(*subpaths: str) -> Path:
    return WORKSPACE_ROOT.joinpath(*subpaths)


@server.tool()
def get_system_hardware_status() -> str:
    """Inspects GPU status (VRAM, utilization), CPU/memory load, connected RealSense cameras,
    serial robot controllers (/dev/ttyUSB*, /dev/ttyACM*), and active training processes."""
    output = []
    output.append("=== Hardware & System Status ===")

    # 1. GPU via nvidia-smi
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3
        ).strip()
        lines = smi_out.splitlines()
        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, total_mem, used_mem, util, temp = parts[:5]
                output.append(f"GPU {idx} ({name}): Util {util}%, Memory {used_mem}MB / {total_mem}MB, Temp {temp}°C")
    except Exception as e:
        output.append(f"GPU: Unable to query nvidia-smi ({e})")

    # 2. Memory / System Load
    try:
        load1, load5, load15 = os.getloadavg()
        output.append(f"CPU Load Avg: {load1:.2f}, {load5:.2f}, {load15:.2f}")
    except Exception:
        pass

    # 3. Serial / Robot bus ports
    serial_devices = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if serial_devices:
        output.append(f"Serial Robot Bus: Connected at {', '.join(sorted(serial_devices))}")
    else:
        output.append("Serial Robot Bus: No /dev/ttyUSB* or /dev/ttyACM* devices detected")

    # 4. USB Devices (RealSense / Controllers)
    try:
        lsusb = subprocess.check_output(["lsusb"], text=True, timeout=3)
        realsense_lines = [l for l in lsusb.splitlines() if any(k in l.lower() for k in ["intel", "realsense", "depth"])]
        if realsense_lines:
            output.append("Cameras detected:")
            for l in realsense_lines:
                output.append(f"  - {l.strip()}")
        else:
            output.append("RealSense Camera: No Intel RealSense USB device found in lsusb")
    except Exception as e:
        output.append(f"USB: {e}")

    # 5. Running RL/Isaac Sim processes
    try:
        ps_out = subprocess.check_output(["ps", "-eo", "pid,user,%cpu,%mem,cmd"], text=True, timeout=3)
        rl_procs = []
        for line in ps_out.splitlines():
            cmd_lower = line.lower()
            if any(k in cmd_lower for k in ["train.py", "eval_pipeline.py", "isaaclab", "play.py", "train_live_session.py"]):
                if "grep" not in cmd_lower and "ps -eo" not in cmd_lower:
                    rl_procs.append(line.strip())
        if rl_procs:
            output.append("\nActive Simulation / Training Processes:")
            for p in rl_procs[:10]:
                output.append(f"  {p}")
        else:
            output.append("\nActive Simulation Processes: None currently running")
    except Exception as e:
        output.append(f"Processes: {e}")

    return "\n".join(output)


@server.tool()
def list_training_runs(limit: int = 10) -> str:
    """Lists the most recent training experiment runs in logs/rsl_rl, including checkpoint counts and dates."""
    runs_dir = _get_workspace_path("logs", "rsl_rl")
    if not runs_dir.exists():
        return f"Runs directory not found at {runs_dir}"

    experiment_dirs = []
    for entry in runs_dir.iterdir():
        if entry.is_dir():
            # Check for subdirectories (timestamps) or direct models
            models = list(entry.glob("**/model_*.pt"))
            mtime = entry.stat().st_mtime
            experiment_dirs.append((mtime, entry.name, len(models), entry))

    experiment_dirs.sort(key=lambda x: x[0], reverse=True)
    if not experiment_dirs:
        return "No training runs found in logs/rsl_rl."

    results = [f"=== Recent Training Runs in {runs_dir} (showing {min(limit, len(experiment_dirs))}) ==="]
    for mtime, name, num_models, path in experiment_dirs[:limit]:
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        # Find latest checkpoint
        models = sorted(list(path.glob("**/model_*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        latest_model = models[0].name if models else "None"
        results.append(f"• {name}")
        results.append(f"  - Last Modified: {date_str}")
        results.append(f"  - Checkpoints: {num_models} saved (Latest: {latest_model})")
        results.append(f"  - Path: {path.relative_to(WORKSPACE_ROOT)}")

    return "\n".join(results)


@server.tool()
def get_run_details(run_name: str = "latest") -> str:
    """Gets detailed info on a specific run in logs/rsl_rl (or the latest run if run_name is 'latest').
    Returns run configuration, available checkpoints, and parameter files."""
    runs_dir = _get_workspace_path("logs", "rsl_rl")
    if not runs_dir.exists():
        return f"Directory not found: {runs_dir}"

    target_dir: Optional[Path] = None
    if run_name.lower() in ("latest", "", "recent"):
        subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        if not subdirs:
            return "No runs found."
        target_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
    else:
        # Search for matching name
        matches = [p for p in runs_dir.glob(f"*{run_name}*") if p.is_dir()]
        if not matches:
            return f"Run '{run_name}' not found in {runs_dir}."
        target_dir = max(matches, key=lambda p: p.stat().st_mtime)

    lines = [f"=== Run Details: {target_dir.name} ==="]
    lines.append(f"Full Path: {target_dir}")

    # Look for run.json
    run_json_files = list(target_dir.glob("**/run.json"))
    if run_json_files:
        try:
            with open(run_json_files[0]) as f:
                data = json.load(f)
                lines.append(f"\nRun Metadata (from {run_json_files[0].relative_to(target_dir)}):")
                lines.append(json.dumps(data, indent=2))
        except Exception as e:
            lines.append(f"Error reading run.json: {e}")

    # Look for model checkpoints
    models = sorted(list(target_dir.glob("**/model_*.pt")), key=lambda p: p.stat().st_mtime)
    if models:
        lines.append(f"\nSaved Checkpoints ({len(models)} total):")
        for m in models[-10:]:  # Show last 10
            size_mb = m.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%H:%M:%S")
            lines.append(f"  • {m.relative_to(target_dir)} ({size_mb:.1f} MB, {mtime})")
    else:
        lines.append("\nNo .pt model checkpoints found in this run.")

    # Look for TensorBoard event files
    tfevents = list(target_dir.glob("**/events.out.tfevents.*"))
    if tfevents:
        lines.append(f"\nTensorBoard Event Files:")
        for ev in tfevents:
            size_kb = ev.stat().st_size / 1024
            lines.append(f"  • {ev.relative_to(target_dir)} ({size_kb:.1f} KB)")

    return "\n".join(lines)


@server.tool()
def list_evaluations(limit: int = 15) -> str:
    """Lists recent evaluation result summaries from outputs/evaluations with success rates,
    mean rewards, peak lift measurements, and steps."""
    eval_dir = _get_workspace_path("outputs", "evaluations")
    if not eval_dir.exists():
        return f"Evaluations directory not found: {eval_dir}"

    eval_files = [
        f for f in eval_dir.glob("*.json")
        if not f.name.endswith(".trace.json") and not f.name.endswith(".provenance.json")
    ]
    eval_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not eval_files:
        return "No evaluation JSON files found in outputs/evaluations/."

    lines = [f"=== Recent Evaluations (showing {min(limit, len(eval_files))} files) ==="]
    lines.append(f"{'Filename':<35} | {'Success%':<8} | {'Mean Rew':<9} | {'Peak Lift':<10} | {'Steps':<6} | Checkpoint")
    lines.append("-" * 105)

    for fpath in eval_files[:limit]:
        try:
            with open(fpath) as f:
                data = json.load(f)
            success_rate = data.get("success_rate")
            success_str = f"{success_rate*100:.1f}%" if success_rate is not None else "N/A"
            mean_rew = data.get("mean_reward")
            rew_str = f"{mean_rew:.2f}" if mean_rew is not None else "N/A"
            peak_lift = data.get("mean_peak_lift_cm")
            lift_str = f"{peak_lift:.2f} cm" if peak_lift is not None else "N/A"
            steps = data.get("mean_steps")
            step_str = f"{steps:.1f}" if steps is not None else "N/A"
            checkpoint = data.get("checkpoint", "unknown")
            ckpt_name = Path(checkpoint).name if checkpoint else "unknown"

            lines.append(f"{fpath.name[:35]:<35} | {success_str:<8} | {rew_str:<9} | {lift_str:<10} | {step_str:<6} | {ckpt_name}")
        except Exception:
            lines.append(f"{fpath.name[:35]:<35} | (error parsing JSON)")

    return "\n".join(lines)


@server.tool()
def get_evaluation_detail(filename: str) -> str:
    """Reads a specific evaluation JSON from outputs/evaluations/ and provides the full breakdown
    including per-episode rewards, closest tool distances, and checks for videos/traces."""
    eval_dir = _get_workspace_path("outputs", "evaluations")
    fpath = eval_dir / filename
    if not fpath.exists():
        matches = list(eval_dir.glob(f"*{filename}*"))
        if matches:
            fpath = matches[0]
        else:
            return f"File '{filename}' not found in {eval_dir}."

    try:
        with open(fpath) as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading {fpath}: {e}"

    lines = [f"=== Evaluation Detail: {fpath.name} ==="]
    lines.append(f"Task: {data.get('task')}")
    lines.append(f"Curriculum: {data.get('curriculum')}")
    lines.append(f"Policy: {data.get('policy')}")
    lines.append(f"Checkpoint: {data.get('checkpoint')}")
    lines.append(f"Robot Asset: {data.get('robot_asset')}")
    lines.append(f"Seed: {data.get('seed')} | Episodes: {data.get('episodes')}")
    lines.append(f"Success Rate: {data.get('success_rate') * 100 if data.get('success_rate') is not None else 'N/A'}%")
    lines.append(f"Mean Reward: {data.get('mean_reward')}")
    lines.append(f"Mean Peak Lift: {data.get('mean_peak_lift_cm')} cm")
    lines.append(f"Mean Table Slide: {data.get('mean_table_slide_cm')} cm")

    # Check companion video / logs
    base_name = fpath.stem
    mp4_file = eval_dir / f"{base_name}.mp4"
    log_file = eval_dir / f"{base_name}.log"
    trace_file = eval_dir / f"{base_name}.trace.json"

    lines.append("\nCompanion Assets:")
    lines.append(f"  • Video (.mp4): {'Available (' + str(round(mp4_file.stat().st_size / (1024*1024), 2)) + ' MB)' if mp4_file.exists() else 'Not found'}")
    lines.append(f"  • Log (.log): {'Available' if log_file.exists() else 'Not found'}")
    lines.append(f"  • Trace (.trace.json): {'Available' if trace_file.exists() else 'Not found'}")

    episodes = data.get("episode_results", [])
    if episodes:
        lines.append(f"\nEpisode Breakdown ({len(episodes)} episodes):")
        for i, ep in enumerate(episodes[:10]):
            success_flag = "✓ PASS" if ep.get("success") else "✗ FAIL"
            lines.append(
                f"  Ep {i+1:02d} [{success_flag}]: Rew={ep.get('reward', 0):.2f}, "
                f"PeakLift={ep.get('peak_lift_cm', 0):.2f}cm, "
                f"ClosestDist={ep.get('closest_tool_distance_cm', 0):.2f}cm"
            )
        if len(episodes) > 10:
            lines.append(f"  ... ({len(episodes) - 10} more episodes omitted)")

    return "\n".join(lines)


@server.tool()
def get_robot_arm_calibration() -> str:
    """Reads my_awesome_follower_arm.json and returns the calibrated servo IDs,
    homing offsets, tick ranges, and zero positions for each joint of the SO-101 robot arm."""
    cfg_path = _get_workspace_path("my_awesome_follower_arm.json")
    if not cfg_path.exists():
        return f"Configuration file not found: {cfg_path}"

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)

        lines = ["=== SO-101 Follower Arm Calibration ==="]
        lines.append(f"{'Joint Name':<16} | {'ID':<3} | {'Homing Offset':<14} | {'Range (Min - Max)':<18} | {'Zero Ticks'}")
        lines.append("-" * 75)

        for joint, p in cfg.items():
            jid = p.get("id", "?")
            offset = p.get("homing_offset", 0)
            rmin = p.get("range_min", 0)
            rmax = p.get("range_max", 0)
            zticks = p.get("zero_ticks", "-")
            lines.append(f"{joint:<16} | {jid:<3} | {offset:<14} | {rmin} - {rmax:<11} | {zticks}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading calibration file: {e}"


@server.tool()
def get_project_dashboard() -> str:
    """Quick high-level dashboard combining Git status, latest training run,
    latest evaluation, and hardware availability in a single call."""
    lines = ["=== Isaac Vision RL Project Dashboard ==="]

    # Git info
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=WORKSPACE_ROOT, text=True, timeout=2).strip()
        last_commit = subprocess.check_output(["git", "log", "-1", "--oneline"], cwd=WORKSPACE_ROOT, text=True, timeout=2).strip()
        lines.append(f"Git Branch: {branch} (Latest: {last_commit})")
    except Exception:
        lines.append("Git Branch: N/A")

    # Latest Run
    runs_dir = _get_workspace_path("logs", "rsl_rl")
    if runs_dir.exists():
        subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        if subdirs:
            latest_run = max(subdirs, key=lambda p: p.stat().st_mtime)
            models = list(latest_run.glob("**/model_*.pt"))
            lines.append(f"Latest Run: {latest_run.name} ({len(models)} checkpoints)")

    # Latest Eval
    eval_dir = _get_workspace_path("outputs", "evaluations")
    if eval_dir.exists():
        eval_files = [f for f in eval_dir.glob("*.json") if not f.name.endswith(".trace.json")]
        if eval_files:
            latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_eval) as f:
                    edata = json.load(f)
                srate = edata.get("success_rate", 0) * 100
                mrew = edata.get("mean_reward", 0)
                lines.append(f"Latest Eval: {latest_eval.name} -> Success Rate: {srate:.1f}%, Mean Reward: {mrew:.2f}")
            except Exception:
                lines.append(f"Latest Eval: {latest_eval.name}")

    # Hardware check
    serial_devices = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    lines.append(f"Robot Arm Port: {'Connected (' + ', '.join(serial_devices) + ')' if serial_devices else 'Disconnected'}")

    return "\n".join(lines)


DEFAULT_REWARD_TERMS = {
    "reach_progress": {"base_weight": 3.0, "category": "approach", "desc": "Continuous progress credit toward the cube"},
    "premature_close": {"base_weight": -3.0, "category": "penalty", "desc": "Penalty for closing fingers before centering"},
    "grasp_approach": {"base_weight": 2.0, "category": "grasp", "desc": "Positioning fingers around cube for grasp"},
    "grasp_close": {"base_weight": 3.0, "category": "grasp", "desc": "Closing gripper fingers when aligned"},
    "contact_grade": {"base_weight": 2.0, "category": "grasp", "desc": "Graded contact between finger pads and cube"},
    "grasp_cube": {"base_weight": 4.0, "category": "grasp", "desc": "Finger closure when cube is centered"},
    "lift_cube": {"base_weight": 8.0, "category": "lift", "desc": "Elevation progress and height holding"},
    "lift_completed": {"base_weight": 6000.0, "category": "lift", "desc": "One-time bonus on reaching target height"},
    "drag_penalty": {"base_weight": -5.0, "category": "penalty", "desc": "Penalty for dragging cube across table"},
    "displacement_penalty": {"base_weight": -12.0, "category": "penalty", "desc": "Penalty for horizontal cube displacement"},
    "table_collision": {"base_weight": -5.0, "category": "penalty", "desc": "Penalty for pressing into tabletop"},
    "action_rate": {"base_weight": -0.06, "category": "regularization", "desc": "Action jerk/smoothness penalty"},
    "joint_vel": {"base_weight": -0.01, "category": "regularization", "desc": "Joint velocity penalty"},
    "joint_limits": {"base_weight": -0.05, "category": "regularization", "desc": "Joint limit clipping penalty"},
    "pinch_force": {"base_weight": 4.0, "category": "grasp", "desc": "Bonus for maximum pinch force on pads"},
    "camera_boundary": {"base_weight": -2.0, "category": "penalty", "desc": "Repulsion from camera FOV edges"},
    "open_jaws": {"base_weight": 0.05, "category": "approach", "desc": "Encourages opening gripper fingers fully before descent"},
    "jaws_open": {"base_weight": 0.05, "category": "approach", "desc": "Encourages opening gripper fingers fully before descent"},
    "point_down": {"base_weight": 0.35, "category": "approach", "desc": "Reward for orienting the wrist/fingers vertically downwards"},
    "place_cube": {"base_weight": 12.0, "category": "place", "desc": "Transporting cube to target box after lift"},
    "release_cube": {"base_weight": 12.0, "category": "place", "desc": "Opening gripper inside target box"},
    "place_completed": {"base_weight": 6000.0, "category": "place", "desc": "One-time placement success bonus"}
}


def _get_reward_overrides() -> dict:
    override_file = _get_workspace_path("reward_overrides.json")
    if override_file.exists():
        try:
            with open(override_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@server.tool()
def get_reward_weights() -> str:
    """Inspects all RL reward terms, their category, baseline weights, and any active MCP overrides."""
    overrides = _get_reward_overrides()
    lines = ["=== RL Reward Configuration ==="]
    if overrides:
        lines.append(f"Active Overrides ({len(overrides)} tuned terms in reward_overrides.json):")
        for k, v in overrides.items():
            lines.append(f"  • {k}: {v}")
        lines.append("")
    else:
        lines.append("Status: Using baseline weights (no overrides active).\n")

    lines.append(f"{'Term Name':<22} | {'Category':<14} | {'Current Wt':<10} | {'Base Wt':<8} | Description")
    lines.append("-" * 90)

    for term, meta in DEFAULT_REWARD_TERMS.items():
        base_wt = meta["base_weight"]
        current_wt = overrides.get(term, base_wt)
        status_flag = f"*{current_wt:<9}" if term in overrides else f" {current_wt:<9}"
        lines.append(f"{term:<22} | {meta['category']:<14} | {status_flag} | {base_wt:<8} | {meta['desc']}")

    return "\n".join(lines)


@server.tool()
def set_reward_weights(updates_json: str) -> str:
    """Updates one or more reward term weights in reward_overrides.json.
    Example updates_json: '{\"reach_progress\": 5.0, \"grasp_cube\": 6.0, \"drag_penalty\": -8.0}'"""
    try:
        new_updates = json.loads(updates_json)
        if not isinstance(new_updates, dict):
            return "Error: updates_json must be a JSON object mapping term names to numerical weights."
    except Exception as e:
        return f"JSON parse error: {e}"

    overrides = _get_reward_overrides()
    applied = []
    ignored = []

    for k, v in new_updates.items():
        try:
            v_float = float(v)
            overrides[k] = v_float
            applied.append(f"{k} -> {v_float}")
        except ValueError:
            ignored.append(f"{k} (invalid value '{v}')")

    override_file = _get_workspace_path("reward_overrides.json")
    try:
        with open(override_file, "w") as f:
            json.dump(overrides, f, indent=2)
    except Exception as e:
        return f"Failed to save {override_file}: {e}"

    res = ["=== Reward Weights Updated ==="]
    res.append(f"Saved to: {override_file.name}")
    if applied:
        res.append("Applied updates:")
        for a in applied:
            res.append(f"  ✓ {a}")
    if ignored:
        res.append("Ignored:")
        for ig in ignored:
            res.append(f"  ✗ {ig}")

    res.append("\nThese weights will be automatically loaded on your next Isaac Lab training or evaluation run.")
    return "\n".join(res)


@server.tool()
def reset_reward_weights() -> str:
    """Removes all custom reward overrides in reward_overrides.json and restores baseline default weights."""
    override_file = _get_workspace_path("reward_overrides.json")
    if override_file.exists():
        try:
            override_file.unlink()
            return f"Reset complete: {override_file.name} deleted. All reward weights restored to baseline."
        except Exception as e:
            return f"Error deleting {override_file}: {e}"
    return "Already at baseline. No reward_overrides.json file was active."


@server.tool()
def diagnose_live_training(run_name: str = "latest") -> str:
    """Reads real-time data from your SO-101 Reward Observatory window and training logs.
    Analyzes sideways alignment, wrist angle, gate status, stage completion fractions,
    and recommends reward adjustments."""
    lines = []

    # 1. Check for live real-time snapshot from SO-101 Reward Observatory window
    obs_file = _get_workspace_path("logs", "live_observatory.json")
    if obs_file.exists():
        try:
            import time
            if time.time() - obs_file.stat().st_mtime < 180:  # Active within 3 mins
                with open(obs_file) as f:
                    obs = json.load(f)
                lines.append("=== Live SO-101 Reward Observatory Telemetry ===")
                st_names = ["01 REACH", "02 GRASP", "03 LIFT", "04 PLACE"]
                cur_stage = obs.get("stage", 0)
                st_label = st_names[cur_stage] if cur_stage < len(st_names) else f"Stage {cur_stage}"
                lines.append(f"Active Stage: {st_label} | Episode Step: {obs.get('episode_step', '?')}")

                if "approach_checks" in obs:
                    ac = obs["approach_checks"]
                    gate_status = "✓ UNLOCKED (Allowed)" if ac.get("descent_allowed") else "✗ LOCKED (Align Above Cube First)"
                    lines.append(f"Descent Gate: {gate_status}")
                    lines.append(f"  • Sideways Error: {ac.get('xy_error_cm', 0):.2f} cm (need <{ac.get('xy_tolerance_cm', 2.0):.1f} cm)")
                    lines.append(f"  • Wrist Down: {ac.get('downward_cosine', 0):.2f} (need >{ac.get('downward_cosine_min', 0.90):.2f})")
                    lines.append(f"  • Table Clearance: {ac.get('clearance_mm', 0):.1f} mm")

                if "grasp_checks" in obs:
                    gc = obs["grasp_checks"]
                    lines.append("\nGrasp Diagnostics:")
                    lines.append(f"  • Pad Touches: Fixed={'✓' if gc.get('fixed_pad') else '✗'}, Moving={'✓' if gc.get('moving_pad') else '✗'}")
                    lines.append(f"  • Jaws Ready: {'✓' if gc.get('jaw_ready') else '✗'} ({gc.get('jaw_rad', 0):.2f} rad)")
                    lines.append(f"  • Enclosed: {'✓' if gc.get('enclosed') else '✗'}")

                lines.append("")
        except Exception:
            pass

    runs_dir = _get_workspace_path("logs", "rsl_rl")
    if not runs_dir.exists():
        return "\n".join(lines) if lines else f"Directory not found: {runs_dir}"

    target_dir: Optional[Path] = None
    if run_name.lower() in ("latest", "", "recent"):
        subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        if not subdirs:
            return "No runs found in logs/rsl_rl."
        target_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
    else:
        matches = [p for p in runs_dir.glob(f"*{run_name}*") if p.is_dir()]
        if not matches:
            return f"Run '{run_name}' not found in {runs_dir}."
        target_dir = max(matches, key=lambda p: p.stat().st_mtime)

    # Find tfevents files
    event_files = sorted(target_dir.glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return f"No TensorBoard event files found in {target_dir.name} yet."

    latest_event = event_files[-1]

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(latest_event))
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
    except Exception as e:
        return f"Error loading event file {latest_event.name}: {e}"

    def get_latest_scalar(tag: str) -> Optional[tuple]:
        if tag in scalars:
            evs = ea.Scalars(tag)
            if evs:
                return evs[-1].step, evs[-1].value
        return None

    def get_first_scalar(tag: str) -> Optional[float]:
        if tag in scalars:
            evs = ea.Scalars(tag)
            if evs:
                return evs[0].value
        return None

    step_info = get_latest_scalar("Train/mean_reward")
    current_step = step_info[0] if step_info else "?"
    current_rew = step_info[1] if step_info else 0.0
    initial_rew = get_first_scalar("Train/mean_reward")

    lines = [f"=== Live Training Diagnostic: {target_dir.name} ==="]
    lines.append(f"Current Update / Step: {current_step}")
    if initial_rew is not None:
        lines.append(f"Mean Reward: {current_rew:.3f} (started at {initial_rew:.3f})")
    else:
        lines.append(f"Mean Reward: {current_rew:.3f}")

    # Key Stage Metrics
    reach_info = get_latest_scalar("FourStage/REACH_fraction")
    grasp_info = get_latest_scalar("FourStage/GRASP_fraction")
    lift_info = get_latest_scalar("FourStage/LIFT_fraction")
    place_info = get_latest_scalar("FourStage/PLACE_fraction")

    lines.append("\nStage Progress Rates:")
    reach_pct = (reach_info[1] * 100) if reach_info else 0.0
    grasp_pct = (grasp_info[1] * 100) if grasp_info else 0.0
    lift_pct = (lift_info[1] * 100) if lift_info else 0.0
    place_pct = (place_info[1] * 100) if place_info else 0.0

    lines.append(f"  • Pregrasp Reach: {reach_pct:.1f}%")
    lines.append(f"  • Grasp Closure:  {grasp_pct:.1f}%")
    lines.append(f"  • Lift Success:   {lift_pct:.1f}%")
    lines.append(f"  • Place Success:  {place_pct:.1f}%")

    # Safety & Terminations
    table_contact = get_latest_scalar("Safety/hand_table_contact_fraction")
    table_drag = get_latest_scalar("FourStageReward/table_dragging")
    timeout_rate = get_latest_scalar("Episode_Termination/time_out")

    lines.append("\nSafety & Terminations:")
    lines.append(f"  • Table Collisions: {(table_contact[1] * 100) if table_contact else 0.0:.2f}% of steps")
    lines.append(f"  • Drag Penalty Active: {'Yes' if table_drag and table_drag[1] < -0.1 else 'Low / None'}")
    lines.append(f"  • Timeouts: {(timeout_rate[1] * 100) if timeout_rate else 0.0:.1f}% of episodes")

    # Automated Diagnosis & Recommendation
    issues = []
    recommendations = []

    if reach_pct > 70.0 and grasp_pct < 10.0:
        issues.append("BOTTLENECK DETECTED AT GRASP: Robot reaches the pregrasp point successfully, but fails to close fingers around the cube.")
        recommendations.append("Increase 'grasp_approach' (e.g. to 4.0) and 'grasp_cube' (e.g. to 6.0).")
        recommendations.append("Slightly ease 'premature_close' (e.g. from -3.0 to -1.5) so the robot isn't overly penalized for closing early.")
    elif grasp_pct > 50.0 and lift_pct < 10.0:
        issues.append("BOTTLENECK DETECTED AT LIFT: Robot grasps the cube, but fails to lift it vertically off the table.")
        recommendations.append("Increase 'lift_cube' weight (e.g. from 8.0 to 14.0).")
        recommendations.append("Increase 'pinch_force' weight (e.g. from 4.0 to 7.0) to prevent the cube from slipping during upward acceleration.")
    elif table_contact and table_contact[1] > 0.05:
        issues.append("HIGH TABLE SCRAPING: Gripper is pressing too hard into the tabletop.")
        recommendations.append("Increase 'table_collision' penalty (e.g. from -5.0 to -10.0).")
    else:
        issues.append("Training progressing steadily across stages.")

    lines.append("\n🔍 Issue Diagnosis:")
    for iss in issues:
        lines.append(f"  ⚠ {iss}")

    if recommendations:
        lines.append("\n💡 Recommended Fixes for Next Training Run:")
        for rec in recommendations:
            lines.append(f"  👉 {rec}")
        lines.append("\nYou can apply these directly by asking: 'Apply the recommended grasp tuning'")

    return "\n".join(lines)


if __name__ == "__main__":
    # Start the MCP server using standard I/O (JSON-RPC)
    server.run("stdio")

