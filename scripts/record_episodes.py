#!/usr/bin/env python3
"""
Record successful and failed grasp-and-lift episodes:
- Evaluates a trained SO-101 RL policy without changing its weights
- Records only strict environment successes as successful episodes
- Records a bounded set of failed episodes, classified by the observed failure
- Captures high-resolution photos at critical milestones:
    1. photo_1_approach.png    (Pre-grasp alignment above the cube)
    2. photo_2_grasp.png       (Fingers firmly closed with perfect contact)
    3. photo_3_lifting.png     (Cube ascending vertically off the tabletop)
    4. photo_4_peak_height.png (Cube stabilized at peak lift clearance)
- Stores videos, milestone images, metrics, and a failure summary in a dedicated folder
"""

import argparse
import glob
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def find_latest_checkpoint(task: str = "lift") -> str:
    patterns = [
        str(ROOT / f"logs/rsl_rl/so101_pick{task}_vision/*/model_*.pt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError("No checkpoint found in logs/rsl_rl")
    files.sort(key=os.path.getmtime)
    return files[-1]


def optimization_report(failure_counts: dict[str, int]) -> dict[str, object]:
    """Translate recorded failure categories into the next controlled experiment."""
    guidance = {
        "did_not_reach_cube": (
            "Inspect camera visibility and arm approach; keep the grasp stage near the cube "
            "before increasing reset range."
        ),
        "gripper_never_closed": "Inspect the camera capture latch, gripper action mapping, and close threshold.",
        "missed_or_unstable_grasp": "Inspect fingertip-to-cube alignment, closing timing, and contact geometry.",
        "dragged_or_pushed_cube": (
            "Increase emphasis on vertical lift after contact and inspect the drag/displacement penalties."
        ),
        "partial_or_unheld_lift": "Inspect grip force, finger contact, and arm motion immediately after lift-off.",
    }
    ordered = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "failure_counts": failure_counts,
        "dominant_failure": ordered[0][0] if ordered else None,
        "recommended_next_check": guidance.get(ordered[0][0]) if ordered else None,
        "note": (
            "This report diagnoses rollout failures. It does not alter PPO weights or reward settings automatically."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, choices=["lift", "place"], help="Task to evaluate")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint (.pt). Defaults to latest.")
    parser.add_argument("--output-dir", default="outputs/recorded_episodes", help="Output directory")
    parser.add_argument("--target-successes", type=int, default=3, help="Number of successful episodes to record")
    parser.add_argument(
        "--target-failures", type=int, default=10, help="Maximum failed episodes to save, classified by failure mode"
    )
    parser.add_argument("--max-attempts", type=int, default=50, help="Maximum evaluation episodes to attempt")
    parser.add_argument("--seed", type=int, default=101, help="Evaluation random seed")
    parser.add_argument("--device", default="cuda:0", help="Simulation device")
    parser.add_argument("--curriculum", default=None, choices=["grasp", "fixed", "lift", "place"])
    parser.add_argument("--policy", default="vision", choices=["vision", "state"])
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY reset jitter in metres")
    args = parser.parse_args()
    if args.target_successes < 0 or args.target_failures < 0 or args.max_attempts < 1:
        parser.error("targets must be non-negative and --max-attempts must be positive")

    task = args.task or ("place" if args.curriculum == "place" else "lift")
    args.curriculum = args.curriculum or task
    if (task == "lift" and args.curriculum == "place") or (task == "place" and args.curriculum != "place"):
        parser.error("Use a lift curriculum for --task lift, or --curriculum place for --task place")
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum

    checkpoint_path = args.checkpoint or find_latest_checkpoint(task)
    checkpoint = Path(checkpoint_path).resolve()
    print(f"[INFO] Using policy checkpoint: {checkpoint} (Task: {task}, Curriculum: {args.curriculum})")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg

    is_place = task == "place"
    task_id = "IsaacVisionRl-Pickplace-SO101" if is_place else "IsaacVisionRl-PickLift-SO101"
    success_term_name = "placed" if is_place else "lift_success"
    cfg = PickplaceEnvCfg() if is_place else PickLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.compute_final_obs = True

    agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    agent.device = args.device
    agent.seed = args.seed

    recorded_successes = []
    recorded_failures = []
    failure_counts: dict[str, int] = {}
    success_count = 0
    failure_count = 0
    attempt_count = 0

    with launch_simulation(cfg, {"device": args.device}):
        env = RslRlVecEnvWrapper(cast("ManagerBasedRLEnv", gym.make(task_id, cfg=cfg)), clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy()
            base = cast("ManagerBasedRLEnv", env.unwrapped)

            obs = env.get_observations()
            manager = base.observation_manager
            task_index = manager.active_terms["critic"].index("task_state")
            task_start = sum(math.prod(dim) for dim in manager.group_obs_term_dim["critic"][:task_index])
            previous = obs["critic"][0, task_start : task_start + 3].clone()
            initial_dist = float(torch.norm(previous - obs["critic"][0, task_start + 6 : task_start + 9]))

            # Episode frame buffers
            frame_buffer = []
            info_buffer = []
            reward_sum = 0.0
            steps = 0
            peak_elevation = 0.0
            drag_distance = 0.0
            closest_dist = initial_dist

            print(
                "[INFO] Beginning evaluation. "
                f"Target successes: {args.target_successes}; failures to diagnose: {args.target_failures}."
            )

            with torch.inference_mode():
                while (
                    (success_count < args.target_successes or failure_count < args.target_failures)
                    and attempt_count < args.max_attempts
                ):
                    action = policy(obs)

                    # Extract global camera RGB
                    image_raw = base.scene["camera"].data.output["rgb"]
                    image_t = image_raw.torch if hasattr(image_raw, "torch") else image_raw
                    rgb = image_t[0, :, :, :3].cpu().numpy().astype(np.uint8)

                    # Query robot & cube state
                    robot = base.scene["robot"]
                    gripper_idx = robot.find_joints("gripper")[0][0]
                    actual_gripper = float(robot.data.joint_pos.torch[0, gripper_idx])
                    cube_data = base.scene["cube"].data.root_pos_w
                    cube_pos = (cube_data.torch if hasattr(cube_data, "torch") else cube_data)[0].cpu().numpy()
                    cube_z = float(cube_pos[2])

                    frame_buffer.append(rgb.copy())
                    info_buffer.append({
                        "step": steps,
                        "cube_z": cube_z,
                        "cube_pos": cube_pos.tolist(),
                        "gripper_actual": actual_gripper,
                    })

                    obs, reward, done, extras = env.step(action)
                    terminal = bool(done[0].item())

                    measured = extras["final_obs"] if terminal else obs
                    curr_pos = measured["critic"][0, task_start : task_start + 3]
                    curr_tcp = measured["critic"][0, task_start + 6 : task_start + 9]
                    closest_dist = min(closest_dist, float(torch.norm(curr_pos - curr_tcp)))
                    peak_elevation = max(peak_elevation, float(curr_pos[2]) - 0.425)

                    if float(previous[2]) < 0.435 and float(curr_pos[2]) < 0.435:
                        drag_distance += float(torch.norm(curr_pos[:2] - previous[:2]))

                    reward_sum += float(reward[0])
                    steps += 1

                    if terminal:
                        attempt_count += 1
                        term_manager = base.termination_manager
                        term_name = success_term_name
                        # Do not infer success from height alone: a pushed cube can briefly rise.
                        # The environment term requires the full sustained-contact success definition.
                        is_success = bool(term_manager.get_term(term_name)[0])
                        gripper_closed = any(item["gripper_actual"] < 1.10 for item in info_buffer)
                        if is_success:
                            outcome = "strict_success"
                        elif peak_elevation >= 0.02:
                            outcome = "partial_or_unheld_lift"
                        elif drag_distance >= 0.01:
                            outcome = "dragged_or_pushed_cube"
                        elif closest_dist > 0.06:
                            outcome = "did_not_reach_cube"
                        elif not gripper_closed:
                            outcome = "gripper_never_closed"
                        else:
                            outcome = "missed_or_unstable_grasp"

                        print(
                            f"  Attempt {attempt_count}: Success={is_success} ({outcome}), "
                            f"PeakLift={peak_elevation*100:.1f}cm, Slide={drag_distance*100:.1f}cm, "
                            f"Steps={steps}, Rew={reward_sum:.1f}"
                        )

                        should_record = is_success or failure_count < args.target_failures
                        if should_record:
                            category = "successful" if is_success else "failures"
                            if is_success:
                                success_count += 1
                                episode_number = success_count
                            else:
                                failure_count += 1
                                failure_counts[outcome] = failure_counts.get(outcome, 0) + 1
                                episode_number = failure_count
                            ep_dir = out_dir / category / f"episode_{episode_number:03d}_{outcome}"
                            ep_dir.mkdir(parents=True, exist_ok=True)

                            # Save full episode MP4
                            video_name = "episode_pick_place.mp4" if is_place else "episode_grasp_lift.mp4"
                            video_path = ep_dir / video_name
                            writer = cv2.VideoWriter(
                                str(video_path),
                                cv2.VideoWriter.fourcc(*"mp4v"),
                                25,
                                (cfg.scene.camera.width, cfg.scene.camera.height),
                            )
                            for idx, f in enumerate(frame_buffer):
                                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                                cv2.putText(
                                    bgr,
                                    f"{outcome} | Step {idx:03d} | Cube Z: {info_buffer[idx]['cube_z']:.3f}m",
                                    (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 255),
                                    2,
                                )
                                writer.write(bgr)
                            writer.release()
                            print(f"  -> Video saved: {video_path}")

                            # Find milestone frames
                            # 1. Approach: frame 0
                            step_approach = 0
                            # Candidate closure frame; angle alone does not prove contact.
                            step_grasp = min(len(frame_buffer) - 1, 20)
                            for i, inf in enumerate(info_buffer):
                                if inf["gripper_actual"] < 1.10:
                                    step_grasp = i
                                    break
                            # 3. Lifting / Transport: when cube rises > 1 cm above table (0.435m)
                            step_lifting = min(len(frame_buffer) - 1, step_grasp + 10)
                            for i, inf in enumerate(info_buffer):
                                if inf["cube_z"] > 0.435:
                                    step_lifting = i
                                    break
                            # 4. Peak height / Placed
                            max_z = max(inf["cube_z"] for inf in info_buffer)
                            step_peak = next(i for i, inf in enumerate(info_buffer) if inf["cube_z"] == max_z)
                            step_final = len(frame_buffer) - 1

                            # Save milestone photos
                            if is_place:
                                milestones = [
                                    (
                                        "photo_1_approach.png",
                                        step_approach,
                                        "Phase 1: Pre-Grasp Approach from Home Pose",
                                    ),
                                    ("photo_2_grasp.png", step_grasp, "Phase 2: Closed Grasp (Zero Air Gap)"),
                                    ("photo_3_transport.png", step_peak, "Phase 3: Transport to Box"),
                                    ("photo_4_placed.png", step_final, "Phase 4: Released and Settled in Box"),
                                ]
                            else:
                                milestones = [
                                    ("photo_1_approach.png", step_approach, "Phase 1: Pre-Grasp Approach"),
                                    ("photo_2_grasp.png", step_grasp, "Phase 2: Closed Grasp with Contact"),
                                    ("photo_3_lifting.png", step_lifting, "Phase 3: Vertical Lift Ascent"),
                                    ("photo_4_peak_height.png", step_peak, "Phase 4: High-Clearance Peak Hold"),
                                ]
                            photo_records = []
                            for filename, step_num, label in milestones:
                                p_path = ep_dir / filename
                                frame_rgb = frame_buffer[step_num]
                                Image.fromarray(frame_rgb).save(p_path)
                                photo_records.append({
                                    "file": filename,
                                    "step": step_num,
                                    "phase": label,
                                    "cube_z_m": info_buffer[step_num]["cube_z"],
                                    "gripper_rad": info_buffer[step_num]["gripper_actual"],
                                })
                                cube_z = info_buffer[step_num]["cube_z"]
                                print(f"  -> Photo saved: {p_path.name} (step {step_num}, cube_z={cube_z:.3f}m)")

                            ep_meta = {
                                "attempt": attempt_count,
                                "episode_number": episode_number,
                                "outcome": outcome,
                                "strict_success": is_success,
                                "total_steps": steps,
                                "total_reward": reward_sum,
                                "peak_lift_cm": peak_elevation * 100,
                                "table_slide_cm": drag_distance * 100,
                                "initial_tool_distance_cm": initial_dist * 100,
                                "closest_tool_distance_cm": closest_dist * 100,
                                "video_file": video_path.name,
                                "photos": photo_records,
                            }
                            (ep_dir / "metrics.json").write_text(json.dumps(ep_meta, indent=2))
                            if is_success:
                                recorded_successes.append(ep_meta)
                            else:
                                recorded_failures.append(ep_meta)

                        # Reset buffers for next attempt
                        frame_buffer.clear()
                        info_buffer.clear()
                        reward_sum = 0.0
                        steps = 0
                        peak_elevation = 0.0
                        drag_distance = 0.0
                        previous = obs["critic"][0, task_start : task_start + 3].clone()
                        initial_dist = float(torch.norm(previous - obs["critic"][0, task_start + 6 : task_start + 9]))
                        closest_dist = initial_dist
                    else:
                        # Accumulate tabletop movement between consecutive frames, not from the episode start.
                        previous = curr_pos.clone()

            # Write master summary
            summary = {
                "checkpoint": str(checkpoint),
                "task": task,
                "curriculum": args.curriculum,
                "strict_success_term": success_term_name,
                "total_attempts": attempt_count,
                "recorded_successful_episodes": len(recorded_successes),
                "recorded_failed_episodes": len(recorded_failures),
                "failure_counts": failure_counts,
                "successful_episodes": recorded_successes,
                "failed_episodes": recorded_failures,
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            (out_dir / "optimization_report.json").write_text(json.dumps(optimization_report(failure_counts), indent=2))
            print(
                f"\n[INFO] Saved {len(recorded_successes)} strict successes and "
                f"{len(recorded_failures)} diagnosed failures into {out_dir}"
            )

        finally:
            env.close()

if __name__ == "__main__":
    main()
