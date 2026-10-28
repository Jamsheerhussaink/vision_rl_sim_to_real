#!/usr/bin/env python3
"""Evaluate an RGB checkpoint using the sim camera and mirror its poses on SO-101.

Defaults to mock motors. --execute opens serial and moves the real robot.
One episode only: simulator reset poses are never forwarded. Normal completion
returns to the initial pose and leaves torque enabled to hold it.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/hardware"))

from isaac_vision_rl.utils.calibration import CALIB_FILE_PATH, JOINT_ORDER, FollowerArmCalibration
from isaac_vision_rl.utils.evaluation_mirror import MirrorWorker, mirror_reference
from isaac_vision_rl.utils.motor_limits import GRIPPER_MIN_CLOSE_RAD


def extract_checkpoint_metadata(path: Path | str) -> dict:
    """Read metadata dictionary from a PyTorch .pt checkpoint zip without importing torch or initializing CUDA."""
    import io
    import pickle
    import zipfile

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return lambda *args, **kwargs: None
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return None

    try:
        with zipfile.ZipFile(str(path), "r") as z:
            for name in z.namelist():
                if name.endswith("data.pkl"):
                    data = z.read(name)
                    obj = SafeUnpickler(io.BytesIO(data)).load()
                    if isinstance(obj, dict):
                        return {
                            k: v
                            for k, v in obj.items()
                            if not k.startswith("model") and not k.endswith("state_dict")
                        }
    except Exception as exc:
        print(f"[WARN] Failed to read checkpoint metadata safely: {exc}")
    return {}


def find_latest_checkpoint(task: str = "lift", policy: str = "rgb") -> str:
    """Find the best / most recent checkpoint compatible with the given task and policy."""
    import glob
    folder = f"so101_pick{task}_{policy}"
    patterns = [
        str(ROOT / f"logs/rsl_rl/{folder}/*/model_*.pt"),
        str(ROOT / "logs/rsl_rl/so101_picklift_rgb/*/model_*.pt"),
        str(ROOT / "logs/rsl_rl/so101_pickplace_rgb/*/model_*.pt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in logs/rsl_rl for task {task}")
    def sort_key(path_str):
        stem = Path(path_str).stem
        iteration = int(stem.split("_")[-1]) if stem.startswith("model_") and stem.split("_")[-1].isdigit() else 0
        return (iteration, os.path.getmtime(path_str))
    files.sort(key=sort_key)
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint (.pt); defaults to latest")
    parser.add_argument("--task", choices=["lift", "place"], default="lift")
    parser.add_argument("--curriculum", choices=["grasp", "fixed", "lift", "place"])
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to mirror (default: 1, 0 for continuous)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--viz", choices=["kit", "newton"], default="newton", help="Visualizer backend (default: newton)")
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    parser.add_argument("--box-jitter", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="Enable real serial motor movement")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--calib", default=CALIB_FILE_PATH)
    parser.add_argument("--speed", type=float, default=600.0, help="Maximum encoder ticks/second per joint (default: 600)")
    parser.add_argument("--gripper-speed", type=float, default=800.0,
                        help="Gripper speed limit in ticks/second (default: 800)")
    parser.add_argument("--allow-holding", action="store_true",
                        help="Adopt an already powered stationary arm at its measured pose")
    parser.add_argument("--release-on-exit", action="store_true",
                        help="Disable torque after stopping; support the arm before using this")
    parser.add_argument("--watchdog", type=float, default=1.0, help="Maximum age of a sim target in seconds (default: 1.0)")
    parser.add_argument("--tracking-limit", type=int, default=350, help="Maximum measured error from sent command, ticks (default: 350)")
    parser.add_argument("--alignment-tolerance", type=int, default=12,
                        help="Startup error tolerance per joint in ticks, held for 0.2 seconds (default: 12)")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional shorter run; 0 means full episodes")
    parser.add_argument("--gripper-min-rad", type=float, default=GRIPPER_MIN_CLOSE_RAD,
                        help="Never command the real jaw below this (0 closed, 1.2 open); stops a cube-blocked jaw "
                             "from being squeezed into servo overload")
    parser.add_argument("--record-video", action="store_true",
                        help="Record sim RGB, real RealSense RGB and wall-clock comparison videos")
    parser.add_argument("--output", default=None, help="JSON report (default unique timestamped path)")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.task, "rgb")
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not all(0 <= v <= 0.04 for v in (args.cube_jitter, args.box_jitter)):
        parser.error("Jitter must be between 0 and 0.04 metres")
    if args.max_steps < 0 or not all(0 < v < float("inf") for v in
                                    (args.speed, args.watchdog, args.tracking_limit, args.baud, args.alignment_tolerance, args.gripper_speed)):
        parser.error("Rates and limits must be finite and positive; max-steps must be nonnegative")

    meta = extract_checkpoint_metadata(checkpoint)
    saved_hz = meta.get("so101_control_hz")
    if saved_hz:
        os.environ["SO101_RGB_FPS"] = str(saved_hz)
        print(f"[CHECKPOINT] Auto-configured control frequency to {saved_hz} Hz from checkpoint metadata.")
    else:
        os.environ.setdefault("SO101_RGB_FPS", "60")

    profile = meta.get("so101_rgb_profile")
    if profile:
        print(f"[CHECKPOINT] Detected profile '{profile}' (iter {meta.get('iter', 'unknown')}).")
        if profile.startswith("424x240"):
            os.environ.setdefault("SO101_CAMERA_WIDTH", "424")
            os.environ.setdefault("SO101_CAMERA_HEIGHT", "240")
        elif profile.startswith("1280x720"):
            os.environ.setdefault("SO101_CAMERA_WIDTH", "1280")
            os.environ.setdefault("SO101_CAMERA_HEIGHT", "720")

    os.environ["SO101_POLICY"] = "rgb"
    os.environ["SO101_CURRICULUM"] = args.curriculum or ("place" if args.task == "place" else "lift")
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg
    from hardware_bridge import FeetechServoBus, MockServoBus

    cfg = PickLiftEnvCfg() if args.task == "lift" else PickplaceEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 1, args.seed, args.device
    viz_choice = None if args.headless else args.viz
    if viz_choice == "newton":
        cfg.sim.default_visualizer_cfg.streaming_view = True
        cfg.sim.default_visualizer_cfg.streaming_envs = [0]
    agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    agent.device, agent.seed = args.device, args.seed
    task_id = "IsaacVisionRl-PickLift-SO101" if args.task == "lift" else "IsaacVisionRl-Pickplace-SO101"
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    output = Path(args.output or f"outputs/mirror/{time.time_ns()}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint), "task": args.task, "seed": args.seed,
              "curriculum": os.environ["SO101_CURRICULUM"], "execute": args.execute,
              "camera": "simulation RGB", "joint_order": JOINT_ORDER, "trace": [],
              "calibration": str(Path(args.calib).resolve()), "speed_ticks_s": args.speed,
              "gripper_speed_ticks_s": args.gripper_speed, "gripper_source": "processed_policy_target",
              "release_on_exit": args.release_on_exit, "allow_holding": args.allow_holding,
              "alignment_tolerance_ticks": args.alignment_tolerance,
              "alignment_settle_s": 0.2, "watchdog_s": args.watchdog, "tracking_limit_ticks": args.tracking_limit,
              "cube_jitter_m": args.cube_jitter, "box_jitter_m": args.box_jitter}
    worker = None
    recorder = None
    try:
        viz_choice = None if args.headless else args.viz
        with launch_simulation(cfg, {"device": args.device, "visualizer": viz_choice}):
            env = RslRlVecEnvWrapper(gym.make(task_id, cfg=cfg), clip_actions=agent.clip_actions)
            try:
                runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
                runner.load(str(checkpoint))
                policy = runner.get_inference_policy()
                base = env.unwrapped
                robot = base.scene["robot"]
                indices = [robot.find_joints(name)[0][0] for name in JOINT_ORDER]

                def angles():
                    return robot.data.joint_pos.torch[0, indices].cpu().tolist()

                def sim_rgb():
                    rgb = base.scene["camera"].data.output["rgb"]
                    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                    return rgb[0, :, :, :3].cpu().numpy()

                obs = env.get_observations()
                # Warm actor inference before enabling motors; do not step/reset the scene.
                with torch.inference_mode():
                    policy(obs)
                calib = FollowerArmCalibration(args.calib)
                from isaac_vision_rl.utils.evaluation_mirror import checked_ticks
                # Sim starts the jaw wider (1.55) than the real jaw can open; clip only the gripper.
                def clipped_start():
                    q = angles()
                    q[5] = float(min(q[5], calib.get_joint_limits_rad()[1][5]))
                    return q

                start = clipped_start()
                checked_ticks(calib, start)  # Reject unreachable first pose before opening serial.
                if stop:
                    report["stop_reason"] = "interrupted_before_motor_start"
                    return
                if args.record_video:
                    from isaac_vision_rl.utils.mirror_video import MirrorVideoRecorder
                    video_folder = output.with_suffix("").with_name(output.stem + "_videos")
                    recorder = MirrorVideoRecorder(video_folder, sim_rgb())
                    recorder.start()  # Camera/encoder must work before opening the motor bus.
                    report["video_folder"] = str(video_folder)
                bus = FeetechServoBus(args.port, args.baud, calib) if args.execute else MockServoBus(calib)
                worker = MirrorWorker(bus, calib, start, rate_hz=1 / base.step_dt,
                                      speed=args.speed, watchdog=args.watchdog, tracking_limit=args.tracking_limit,
                                      alignment_tolerance=args.alignment_tolerance,
                                      gripper_speed=args.gripper_speed, allow_holding=args.allow_holding,
                                      release_on_exit=args.release_on_exit,
                                      close_gripper_on_home=True)
                print(f"{'REAL MOTORS' if args.execute else 'MOCK MOTORS'}: approaching simulation start pose.")
                print("Normal completion returns home, then holds with torque enabled. "
                      "Interrupts/faults skip home; keep a physical power cutoff accessible.")
                if args.release_on_exit:
                    print("--release-on-exit: torque will be disabled on shutdown; support the arm.")
                worker.start(cancelled=lambda: stop)
                ep_target = args.episodes
                episodes_completed = 0
                report["episodes_completed"] = 0
                report["step_dt_s"] = base.step_dt
                report["stop_reason"] = "interrupted"

                target_str = f"{ep_target} episode{'s' if ep_target != 1 else ''}" if ep_target > 0 else "continuous evaluation"
                print(f"Mirroring evaluation lively in real time at nominal {1 / base.step_dt:.1f} Hz ({target_str}).")

                with torch.inference_mode():
                    while not stop and (ep_target == 0 or episodes_completed < ep_target):
                        ep_idx = episodes_completed + 1
                        total_str = f"/{ep_target}" if ep_target > 0 else " (continuous)"
                        print(f"\n{'='*25} EPISODE {ep_idx}{total_str} {'='*25}", flush=True)
                        deadline = time.monotonic()
                        steps = 0
                        while not stop:
                            # Copy visible pose BEFORE env.step: step may auto-reset on done.
                            if recorder is not None:
                                recorder.publish(sim_rgb())
                            sim_q = angles()
                            gripper_target = (float(base.action_manager.get_term("gripper_action").processed_actions[0, 0])
                                              if steps > 0 else sim_q[5])
                            q = mirror_reference(sim_q, gripper_target, calib=calib)
                            q[5] = max(q[5], args.gripper_min_rad)
                            feedback = worker.publish(q)
                            report["trace"].append({"episode": ep_idx, "step": steps, "sim_time_s": steps * base.step_dt,
                                                    "wall_time": time.time(), "sim_joint_rad": sim_q,
                                                    "hardware_reference_rad": q,
                                                    "gripper_policy_target_rad": gripper_target, **feedback})
                            action = policy(obs)
                            obs, _, dones, _ = env.step(action)
                            worker.check()
                            steps += 1

                            if steps % 15 == 0:
                                sys.stdout.write(
                                    f"\r[MIRROR] Ep {ep_idx} | Step {steps:3d} ({steps * base.step_dt:.2f}s) | "
                                    f"Sim Grip {sim_q[5]:.2f} | Cmd Grip {gripper_target:.2f} rad   "
                                )
                                sys.stdout.flush()

                            if bool(dones[0].item()):
                                sys.stdout.write("\n")
                                print(f"[SIM] Episode {ep_idx} completed at step {steps} ({steps * base.step_dt:.2f}s).", flush=True)
                                report["stop_reason"] = "episode_ended"
                                break
                            if args.max_steps and steps >= args.max_steps:
                                sys.stdout.write("\n")
                                print(f"[SIM] Episode {ep_idx} reached max steps {args.max_steps}.", flush=True)
                                report["stop_reason"] = "max_steps"
                                break
                            deadline += base.step_dt
                            remaining = deadline - time.monotonic()
                            if remaining > 0:
                                time.sleep(remaining)
                            else:
                                # Never send a burst of commands to catch up after slow rendering.
                                deadline = time.monotonic()

                        episodes_completed += 1
                        report["episodes_completed"] = episodes_completed
                        report["steps"] = len(report["trace"])

                        if not stop and (ep_target == 0 or episodes_completed < ep_target):
                            print(f"[RESET] Repositioning arm to home pose for Episode {episodes_completed + 1}...", flush=True)
                            if recorder is not None:
                                recorder.publish(phase="return_home (between episodes)")
                            worker.return_home(cancelled=lambda: stop)
                            time.sleep(1.0)
                            checked_ticks(calib, clipped_start())

                if not stop:
                    print("\n[HOME] Evaluation completed. Returning to initial simulation pose...", flush=True)
                    if recorder is not None:
                        recorder.publish(phase="return_home (sim frame held)")
                    worker.return_home(cancelled=lambda: stop)
                    report["returned_home"] = True
                    print("[HOME] Reached home. Leaving torque enabled to hold position."
                          if not args.release_on_exit else "[HOME] Reached home. Releasing torque as requested.", flush=True)
            except BaseException as exc:
                report["error"] = str(exc)
                raise
            finally:
                try:
                    if worker is not None:
                        with worker.lock:
                            report["last_motor_feedback"] = dict(worker.feedback)
                        try:
                            worker.close()
                        finally:
                            report["motor_shutdown_state"] = worker.shutdown_state
                        worker = None
                except BaseException as exc:
                    report["error"] = str(exc)
                    raise
                finally:
                    if recorder is not None:
                        try:
                            recorder.close()
                        except Exception as exc:
                            report["recording_error"] = str(exc)
                        report["video_frames"] = recorder.frames
                    # Kit shutdown can terminate Python; persist before closing it.
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"Mirror report: {output}", flush=True)
                    env.close()
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Mirror report: {output}")


if __name__ == "__main__":
    main()
