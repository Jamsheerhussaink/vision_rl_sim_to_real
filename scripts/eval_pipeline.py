#!/usr/bin/env python3
"""Evaluate complete deterministic episodes and write measured results to JSON."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FOUR_STAGE_NAMES = ("REACH", "GRASP", "LIFT", "PLACE")


def active_four_stage(log):
    """Read the active stage from the reward manager's diagnostic log, if present."""
    if not isinstance(log, dict):
        return None
    for index, name in enumerate(FOUR_STAGE_NAMES):
        try:
            if float(log.get(f"FourStage/{name}_fraction", 0.0)) >= 0.5:
                return name
        except (TypeError, ValueError):
            continue
    return None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def lift_condition_diagnostics(height, xy_separation, z_separation, jaw_angle, speed, forces,
                               minimal_height=0.47):
    """Read-only comparison with LiftSuccess; does not advance its hold counter."""
    conditions = {
        "bilateral_contact": len(forces) == 2 and min(forces.values()) > 0.02,
        "height": height > minimal_height,
        "xy_separation": xy_separation < 0.035,
        "z_separation": z_separation < 0.045,
        "jaw_angle": jaw_angle < 1.10,
        "cube_speed": speed < 0.25,
    }
    return {
        **conditions,
        "all_conditions": all(conditions.values()),
        "all_except_jaw_angle": all(value for key, value in conditions.items() if key != "jaw_angle"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "place"], default="lift")
    parser.add_argument("--checkpoint", required=True, help="Explicit checkpoint from the new agent")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--viz", choices=["kit", "newton"])
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--trace-only", action="store_true", help="Save per-step diagnostics without recording video")
    parser.add_argument("--record-video", action="store_true", help="Save global camera MP4 and per-step gripper trace")
    parser.add_argument("--contact-overlay", action="store_true", help="Overlay collision pad boxes on video only; does not change policy inputs or physics")
    parser.add_argument(
        "--sample-actions", action="store_true", help="Evaluate exploratory actions rather than actor mean"
    )
    parser.add_argument("--curriculum", choices=["grasp", "fixed", "lift", "place"], default=None)
    parser.add_argument(
        "--policy",
        choices=["rgb"],
        default="rgb",
        help="RGB images and joint/controller readings",
    )
    parser.add_argument("--cube-jitter", type=float, default=0.0, help="Cube XY reset jitter in metres, 0 to 0.04")
    parser.add_argument("--box-jitter", type=float, default=0.0, help="Box XY reset jitter in metres, 0 to 0.04")
    args = parser.parse_args()
    if args.contact_overlay and not args.record_video:
        parser.error("--contact-overlay requires --record-video")
    if not 0 <= args.cube_jitter <= 0.04:
        parser.error("--cube-jitter must be between 0 and 0.04 metres")
    if not 0 <= args.box_jitter <= 0.04:
        parser.error("--box-jitter must be between 0 and 0.04 metres")
    os.environ["SO101_CUBE_JITTER"] = str(args.cube_jitter)
    os.environ["SO101_BOX_JITTER"] = str(args.box_jitter)
    os.environ["SO101_POLICY"] = args.policy
    os.environ["SO101_CURRICULUM"] = args.curriculum or ("place" if args.task == "place" else "lift")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils.dict import class_to_dict
    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg, PickplaceEnvCfg

    cfg = PickLiftEnvCfg() if args.task == "lift" else PickplaceEnvCfg()
    cfg.scene.num_envs = 1
    if args.contact_overlay:
        cfg.scene.fixed_finger_contact.track_contact_points = True
        cfg.scene.moving_finger_contact.track_contact_points = True
        cfg.scene.fixed_finger_contact.max_contact_data_count_per_prim = 64
        cfg.scene.moving_finger_contact.max_contact_data_count_per_prim = 64
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.viz == "newton":
        cfg.sim.default_visualizer_cfg.streaming_view = True
        cfg.sim.default_visualizer_cfg.streaming_envs = [0]
        cfg.sim.default_visualizer_cfg.visible_env_indices = [0]
        cfg.sim.default_visualizer_cfg.randomly_sample_visible_envs = False
    cfg.compute_final_obs = True
    agent = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    agent.device = args.device
    agent.seed = args.seed
    task_id = "IsaacVisionRl-PickLift-SO101" if args.task == "lift" else "IsaacVisionRl-Pickplace-SO101"
    success_term = "lift_success" if args.task == "lift" else "placed"
    results = []
    video = None
    trace = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, {"device": args.device, "visualizer": args.viz}):
        env = RslRlVecEnvWrapper(cast("ManagerBasedRLEnv", gym.make(task_id, cfg=cfg)), clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(env, class_to_dict(agent), log_dir=None, device=args.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy()
            base = cast("ManagerBasedRLEnv", env.unwrapped)
            viewer_binding=[
                {"type":type(v).__name__,"visible_env_ids":v.get_visualized_env_ids()}
                for v in base.sim.visualizers if hasattr(v,'get_visualized_env_ids')
            ]
            print(f'[VIEWER CHECK] reward environment=0; visualizers={viewer_binding}',flush=True)
            if args.record_video:
                import cv2
                import numpy as np

                video = cv2.VideoWriter(
                    str(output.with_suffix(".mp4")),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    1 / base.step_dt,
                    (cfg.scene.camera.width, cfg.scene.camera.height),
                )
                if not video.isOpened():
                    raise RuntimeError("Could not open MP4 video encoder")
            obs = env.get_observations()
            reward_sum, steps, peak, drag = 0.0, 0, 0.0, 0.0
            stage_step_counts = {name: 0 for name in FOUR_STAGE_NAMES}
            observed_stages = []
            manager = cast("ManagerBasedRLEnv", env.unwrapped).observation_manager
            task_index = manager.active_terms["critic"].index("task_state")
            task_start = sum(math.prod(dim) for dim in manager.group_obs_term_dim["critic"][:task_index])
            robot = base.scene["robot"]
            overlay = None
            if args.contact_overlay:
                from isaac_vision_rl.utils.contact_overlay import ContactOverlay
                overlay = ContactOverlay(cfg.scene.robot.spawn.usd_path, robot)
            tool_id = robot.find_bodies("gripper_frame_link")[0][0]
            wrist_id = robot.find_joints("wrist_roll")[0][0]
            previous = obs["critic"][0, task_start : task_start + 3].clone()
            initial_distance = float(torch.norm(previous - obs["critic"][0, task_start + 6 : task_start + 9]))
            closest_distance, visible_steps = initial_distance, 0.0
            from isaac_vision_rl.utils.hand_table import hand_table_metrics
            minimum_clearance, peak_table_force = float("inf"), 0.0
            with torch.inference_mode():
                while len(results) < args.episodes:
                    clearance, table_force = hand_table_metrics(base)
                    minimum_clearance = min(minimum_clearance, float(clearance[0]))
                    peak_table_force = max(peak_table_force, float(table_force[0]))
                    action = runner.alg.actor(obs, stochastic_output=True) if args.sample_actions else policy(obs)
                    if video is not None or args.trace_only:
                        image = base.scene["camera"].data.output["rgb"]
                        image = image.torch if hasattr(image, "torch") else image
                        rgb = image[0, :, :, :3].cpu().numpy().astype(np.uint8) if video is not None else None
                        robot = base.scene["robot"]
                        joint = robot.find_joints("gripper")[0][0]
                        actual = float(robot.data.joint_pos.torch[0, joint])
                        raw = float(action[0, -1])
                        gripper_term = base.action_manager.get_term("gripper_action")
                        target = float(gripper_term.processed_actions[0, 0])
                        forces = {}
                        for name in ("fixed_finger_contact", "moving_finger_contact"):
                            matrix = base.scene[name].data.normal_force_matrix_w
                            if matrix is not None:
                                matrix = matrix.torch if hasattr(matrix, "torch") else matrix
                                forces[name] = float(torch.linalg.vector_norm(matrix[0], dim=-1).max())
                        cube = base.scene["cube"]
                        cube_pos = cube.data.root_pos_w.torch[0]
                        tool_pos = robot.data.body_pos_w.torch[0, tool_id]
                        offset = cube_pos - tool_pos
                        cube_speed = float(torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch[0]))
                        height = float(cube_pos[2])
                        xy_separation = float(torch.linalg.vector_norm(offset[:2]))
                        z_separation = abs(float(offset[2]))
                        held_steps = None
                        hold_conditions = None
                        if args.task == "lift":
                            term = base.termination_manager.get_term_cfg("lift_success")
                            held_steps = int(term.func._held_steps[0])
                            hold_conditions = lift_condition_diagnostics(
                                height, xy_separation, z_separation, actual, cube_speed, forces,
                                minimal_height=term.params.get("minimal_height", 0.47),
                            )
                        trace.append(
                            {
                                "episode": len(results),
                                "step": steps,
                                "gripper_action": raw,
                                "gripper_target_rad": target,
                                "gripper_actual_rad": actual,
                                "wrist_roll_rad": float(robot.data.joint_pos.torch[0, wrist_id]),
                                "wrist_roll_velocity_rad_s": float(robot.data.joint_vel.torch[0, wrist_id]),
                                "tool_quaternion_xyzw": robot.data.body_quat_w.torch[0, tool_id].cpu().tolist(),
                                "tool_angular_velocity_rad_s": robot.data.body_ang_vel_w.torch[0, tool_id].cpu().tolist(),
                                "cube_contact_normal_force_n": forces,
                                "cube_position_world_m": cube_pos.cpu().tolist(),
                                "cube_velocity_world_m_s": cube.data.root_lin_vel_w.torch[0].cpu().tolist(),
                                "cube_quaternion_xyzw": cube.data.root_quat_w.torch[0].cpu().tolist(),
                                "tool_position_world_m": tool_pos.cpu().tolist(),
                                "cube_height_m": height,
                                "cube_tool_xy_separation_m": xy_separation,
                                "cube_tool_z_separation_m": z_separation,
                                "cube_speed_m_s": cube_speed,
                                "lift_hold_conditions": hold_conditions,
                                "lift_held_steps": held_steps,
                                "robot_base_quaternion_xyzw": robot.data.root_quat_w.torch[0].cpu().tolist(),
                                "trace_phase": "before_action_application",
                            }
                        )
                        state=getattr(base,'_four_stage',None)
                        panel=getattr(base,'_training_reward_panel',None)
                        sample=getattr(panel,'snapshot',None) or {}
                        if state is not None:
                            measurements=getattr(state,'measurements',{})
                            trace[-1].update(
                                task_stage=int(state.stage[0]),
                                graph_stage=sample.get('stage'),
                                episode_step=int(base.episode_length_buf[0]),
                                graph_episode_step=sample.get('episode_step'),
                                grasp_confirmed=sample.get('grasp_confirmed'),
                                stage_measurements={k:float(v[0]) for k,v in measurements.items()
                                    if k in ('pregrasp_distance','grasp_pose_distance','downward_cosine',
                                             'tool_height_above_cube','clearance','table_force','jaw',
                                             'reach_ready','cube_between_fingers','bilateral','enclosed',
                                             'fixed_contact_in_pad','moving_contact_in_pad',
                                             'fixed_contact_force_N','moving_contact_force_N','opposing_contact_forces')},
                                descent_unlocked=bool(state.descent_ready[0]),
                            )
                        if video is not None:
                            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.putText(
                                frame,
                                f"ep {len(results) + 1} step {steps} target {target:.2f} actual {actual:.2f}",
                                (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,
                                (255, 255, 255),
                                1,
                            )
                            if state is not None:
                                graph_index=sample.get('stage')
                                graph_label=FOUR_STAGE_NAMES[graph_index] if isinstance(graph_index,int) and 0<=graph_index<4 else 'waiting'
                                cv2.putText(frame,
                                    f'ENV 0 | task {FOUR_STAGE_NAMES[int(state.stage[0])]} | graph {graph_label} | grip {sample.get("grasp_confirmed")}',
                                    (8,42),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                            if overlay is not None:
                                trace[-1]["collision_pads"] = overlay.draw(frame, robot, base.scene["camera"], forces, base.scene)
                            video.write(frame)
                    obs, reward, done, extras = env.step(action)
                    panel=getattr(base,'_training_reward_panel',None)
                    displayed=getattr(panel,'snapshot',None) or {}
                    if trace and not bool(done[0]) and 'step_reward' in displayed:
                        trace[-1]['ppo_reward_after_action']=float(reward[0])
                        trace[-1]['graph_reward_after_action']=float(displayed['step_reward'])
                    log = extras.get("log", {}) if isinstance(extras, dict) else {}
                    current_stage = active_four_stage(log)
                    if current_stage is not None:
                        stage_step_counts[current_stage] += 1
                        if not observed_stages or observed_stages[-1] != current_stage:
                            observed_stages.append(current_stage)
                    terminal = bool(done[0].item())
                    # Four-stage measurements retain the terminal physical
                    # reading even after same-step autoreset moves home.
                    terminal_measurements=getattr(getattr(base,'_four_stage',None),'measurements',{}) if terminal else {}
                    if 'clearance' in terminal_measurements:
                        minimum_clearance=min(minimum_clearance,float(terminal_measurements['clearance'][0]))
                        peak_table_force=max(peak_table_force,float(terminal_measurements['table_force'][0]))
                        if trace:
                            trace[-1]['terminal_after_action']={
                                'hand_table_clearance_m':float(terminal_measurements['clearance'][0]),
                                'hand_table_force_n':float(terminal_measurements['table_force'][0]),
                            }
                    # Same-step autoreset: use terminal state, not the next episode's cube.
                    measured = extras["final_obs"] if terminal else obs
                    current = measured["critic"][0, task_start : task_start + 3]
                    tcp = measured["critic"][0, task_start + 6 : task_start + 9]
                    closest_distance = min(closest_distance, float(torch.norm(current - tcp)))
                    peak = max(peak, float(current[2]) - 0.425)
                    if float(previous[2]) < 0.435 and float(current[2]) < 0.435:
                        drag += float(torch.norm(current[:2] - previous[:2]))
                    reward_sum += float(reward[0])
                    steps += 1
                    if terminal:
                        termination_terms = (
                            "unsafe_table_contact", "physics_health", "time_out", "cube_dropped",
                            "robot_dropped", "gripper_out_of_view", "cube_out_of_view", success_term,
                        )
                        termination_reasons = [
                            name for name in termination_terms
                            if bool(base.termination_manager.get_term(name)[0])
                        ]
                        results.append(
                            {
                                "success": bool(
                                    cast("ManagerBasedRLEnv", env.unwrapped).termination_manager.get_term(success_term)[
                                        0
                                    ]
                                ),
                                "reward": reward_sum,
                                "steps": steps,
                                "peak_lift_cm": 100 * max(0.0, peak),
                                "table_slide_cm": 100 * drag,
                                "initial_tool_distance_cm": 100 * initial_distance,
                                "closest_tool_distance_cm": 100 * closest_distance,
                                "camera_visible_fraction": None,
                                "minimum_sampled_hand_table_clearance_mm": 1000 * minimum_clearance,
                                "peak_sampled_hand_table_force_n": peak_table_force,
                                "unsafe_table_termination": bool(base.termination_manager.get_term("unsafe_table_contact")[0]),
                                "termination_reasons": termination_reasons,
                                "observed_stages": list(observed_stages),
                                "stage_step_counts": dict(stage_step_counts),
                            }
                        )
                        if video is not None or args.trace_only:
                            episode_trace = trace[-steps:]
                            closed = [v for v in episode_trace if v["gripper_target_rad"] < 1.19]
                            supported = [
                                v for v in episode_trace
                                if len(v["cube_contact_normal_force_n"]) == 2
                                and min(v["cube_contact_normal_force_n"].values()) > 0.02
                            ]
                            results[-1].update({
                                "closure_command_steps": len(closed),
                                "peak_wrist_speed_rad_s": max(abs(v["wrist_roll_velocity_rad_s"]) for v in episode_trace),
                                "peak_supported_tool_angular_speed_rad_s": max(
                                    (math.sqrt(sum(x * x for x in v["tool_angular_velocity_rad_s"])) for v in supported),
                                    default=0.0,
                                ),
                                "peak_finger_normal_force_n": max(
                                    (f for v in episode_trace for f in v["cube_contact_normal_force_n"].values()), default=0.0
                                ),
                            })
                        print(f"Episode {len(results)}/{args.episodes}: {results[-1]}")
                        reward_sum, steps, peak, drag = 0.0, 0, 0.0, 0.0
                        stage_step_counts = {name: 0 for name in FOUR_STAGE_NAMES}
                        observed_stages = []
                        minimum_clearance, peak_table_force = float("inf"), 0.0
                        previous = obs["critic"][0, task_start : task_start + 3].clone()
                        initial_distance = float(
                            torch.norm(previous - obs["critic"][0, task_start + 6 : task_start + 9])
                        )
                        closest_distance, visible_steps = initial_distance, 0.0
                    else:
                        previous = current.clone()
            summary = {
                "robot_asset": cfg.scene.robot.spawn.usd_path,
                "robot_asset_sha256": hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest(),
                "task": args.task,
                "curriculum": os.environ["SO101_CURRICULUM"],
                "policy": args.policy,
                "smooth_control": False,
                "wrist_only_control": False,
                "cube_jitter_m": args.cube_jitter,
                "num_rerenders_on_reset": cfg.num_rerenders_on_reset,
                "checkpoint": str(checkpoint),
                "sample_actions": args.sample_actions,
                "seed": args.seed,
                "episodes": len(results),
                "viewer_binding":viewer_binding,
                "graph_task_stage_mismatches":sum(v.get('graph_stage') is not None and v.get('graph_stage')!=v.get('task_stage') for v in trace),
                "graph_ppo_reward_mismatches":sum(abs(v['ppo_reward_after_action']-v['graph_reward_after_action'])>1e-6
                    for v in trace if 'ppo_reward_after_action' in v),
                "success_rate": sum(r["success"] for r in results) / len(results),
                **{
                    f"mean_{key}": sum(r[key] for r in results) / len(results)
                    for key in ("reward", "steps", "peak_lift_cm", "table_slide_cm")
                },
                "episode_results": results,
                "success_definition": "Two-sided cube contact and geometric sustained lift for 0.5 seconds"
                if args.task == "lift"
                else "Prior sustained lift, release, withdrawal, and settled containment",
            }
            output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Success: {summary['success_rate']:.1%}; results saved to {output}")
        finally:
            if video is not None:
                video.release()
            if video is not None or args.trace_only:
                output.with_suffix(".trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            env.close()


if __name__ == "__main__":
    main()
