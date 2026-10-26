#!/usr/bin/env python3
"""Automated Multi-Stage Curriculum Runner for SO-101 Vision RL.

Monitors the currently active lesson (e.g. approach), and automatically chains
forward through all subsequent curriculum stages:
  1. approach  (reaches pregrasp above cube with open jaws)
  2. grasp     (closes jaws and secures bilateral contact)
  3. lift      (lifts cube stably off tabletop)
  4. place     (transports cube into box and releases)

Supports fast headless execution across parallel environments.
"""

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/home/ai/miniconda3/envs/env_isaaclab/bin/python"
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = sys.executable

LOG_DIR = ROOT_DIR / "logs"
STATUS_FILE = LOG_DIR / "curriculum_status.json"
RUNNER_LOG = LOG_DIR / "auto_curriculum.log"

STAGES = ["approach", "grasp", "lift", "place"]
STAGE_CONFIGS = {
    "approach": {"envs": 16, "iterations": 500},
    "grasp":    {"envs": 16, "iterations": 200},  # Grasp fine-tuning with closing reward and without table-abort
    "lift":     {"envs": 16, "iterations": 400},
    "place":    {"envs": 16, "iterations": 500},
}


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [AutoCurriculum] {msg}"
    print(formatted, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNNER_LOG, "a") as f:
        f.write(formatted + "\n")


def update_status(stage: str, state: str, details: dict = None):
    data = {
        "current_stage": stage,
        "state": state,
        "updated_at": datetime.now().isoformat(),
        "details": details or {},
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_latest_checkpoint(stage: str) -> str | None:
    exp_dir = LOG_DIR / "rsl_rl" / f"so101_pickplace_rgb_simple_curriculum_v1_{stage}"
    if not exp_dir.exists():
        return None
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    latest_run = max(runs, key=lambda p: p.stat().st_mtime)
    ckpts = list(latest_run.glob("model_*.pt"))
    if not ckpts:
        return None
    # Sort numerically by model index
    def ckpt_index(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return 0
    best_ckpt = max(ckpts, key=ckpt_index)
    return str(best_ckpt)


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_running_curriculum_pid() -> tuple[int | None, str | None]:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if "train_curriculum.py" in line and "grep" not in line:
                parts = line.split()
                pid = int(parts[1])
                stage = "approach"
                for s in STAGES:
                    if f"--lesson {s}" in line:
                        stage = s
                        break
                return pid, stage
    except Exception as e:
        log(f"Error checking processes: {e}")
    return None, None


def run_stage(stage: str, checkpoint: str | None = None, headless: bool = True, reward_window: bool = False) -> str:
    cfg = STAGE_CONFIGS[stage]
    cmd = [
        PYTHON_BIN,
        str(ROOT_DIR / "scripts/train_curriculum.py"),
        "--lesson", stage,
        "--task", "place",
        "--curriculum", "place",
        "--reward-profile", "four-stage",
        "--reward-style", "continuous",
        "--envs", str(cfg["envs"]),
        "--iterations", str(cfg["iterations"]),
        "--device", "cuda:0",
        "--lift-rise-min", "0.10",
        "--lift-rise-max", "0.12",
    ]
    if reward_window:
        cmd.append("--reward-window")

    if headless:
        cmd.append("--headless")
    else:
        cmd += ["--viz", "newton", "--max_visible_envs", "1", "--live-reward-graphs"]

    if checkpoint:
        cmd += ["--checkpoint", checkpoint, "--actor-warmstart"]

    log(f"Launching Stage: {stage.upper()} (Iterations: {cfg['iterations']}, Envs: {cfg['envs']})")
    log(f"Command: {' '.join(cmd)}")
    update_status(stage, "starting", {"checkpoint": checkpoint, "command": cmd})

    env = os.environ.copy()
    env["SO101_TABLE_DRAG_DEADBAND"] = "0.003"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    update_status(stage, "running", {"pid": proc.pid, "checkpoint": checkpoint})

    stage_log_path = LOG_DIR / f"curriculum_{stage}.log"
    with open(stage_log_path, "w") as stage_log:
        for line in proc.stdout:
            stage_log.write(line)
            stage_log.flush()
            if "Iteration" in line or "Update" in line or "step" in line.lower() or "[" in line:
                stripped = line.strip()
                if stripped.startswith("[") and ("/" in stripped or "%" in stripped):
                    print(f"[{stage}] {stripped}", flush=True)
                    update_status(stage, "running", {
                        "pid": proc.pid,
                        "checkpoint": checkpoint,
                        "progress": stripped,
                    })

    proc.wait()
    if proc.returncode != 0:
        log(f"Warning: Stage {stage} finished with exit code {proc.returncode}")
        update_status(stage, "error", {"exit_code": proc.returncode})
    else:
        log(f"Stage {stage.upper()} completed successfully!")
        update_status(stage, "completed")

    latest_ckpt = get_latest_checkpoint(stage)
    log(f"Stage {stage} produced checkpoint: {latest_ckpt}")
    return latest_ckpt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous multi-stage curriculum runner")
    parser.add_argument("--start-stage", choices=STAGES, default=None, help="Force start from a specific stage")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint to warmstart from")
    parser.add_argument("--gui", action="store_true", help="Run with Newton GUI viewer (default is headless)")
    args = parser.parse_args()

    headless = not args.gui
    log(f"=== SO-101 Autonomous Curriculum Pipeline Started (Headless: {headless}) ===")
    
    # Check if a lesson is currently running
    pid, current_stage = find_running_curriculum_pid()
    if pid:
        log(f"Detected currently active run for lesson '{current_stage}' (PID: {pid}). Waiting for completion...")
        update_status(current_stage, "monitoring_active_pid", {"pid": pid})
        while is_pid_running(pid):
            time.sleep(10)
        log(f"Active run for lesson '{current_stage}' (PID: {pid}) has finished.")
        start_idx = STAGES.index(current_stage) + 1
        last_ckpt = get_latest_checkpoint(current_stage)
    elif args.start_stage:
        start_idx = STAGES.index(args.start_stage)
        last_ckpt = args.checkpoint
        if not last_ckpt and start_idx > 0:
            last_ckpt = get_latest_checkpoint(STAGES[start_idx - 1])
    else:
        log("No currently running lesson detected. Checking latest available checkpoints...")
        start_idx = 0
        last_ckpt = None
        for i, s in enumerate(STAGES):
            ckpt = get_latest_checkpoint(s)
            if ckpt:
                start_idx = i + 1
                last_ckpt = ckpt

    log(f"Starting pipeline from stage index {start_idx} (Next: {STAGES[start_idx] if start_idx < len(STAGES) else 'Done'})")
    log(f"Initial warmstart checkpoint: {last_ckpt}")

    for idx in range(start_idx, len(STAGES)):
        stage_name = STAGES[idx]
        log(f"\n=======================================================")
        log(f" ADVANCING TO LESSON {idx+1}/{len(STAGES)}: {stage_name.upper()}")
        log(f"=======================================================")
        
        last_ckpt = run_stage(stage_name, checkpoint=last_ckpt, headless=headless)
        if not last_ckpt:
            log(f"Error: No checkpoint found after {stage_name}! Halting pipeline.")
            update_status(stage_name, "failed_no_checkpoint")
            sys.exit(1)
        
        log(f"Successfully finalized {stage_name}. Next stage will use: {last_ckpt}")
        log("Allowing 10s cooldown for GPU memory release before next stage...")
        time.sleep(10)

    log("\n🎉 ALL CURRICULUM STAGES (approach -> grasp -> lift -> place) HAVE COMPLETED SUCCESSFULLY! 🎉")
    update_status("all", "complete", {"final_checkpoint": last_ckpt})


if __name__ == "__main__":
    main()

