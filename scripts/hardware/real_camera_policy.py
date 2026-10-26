#!/usr/bin/env python3
"""Run the trained actor on REAL RGB and REAL encoders. No simulator is launched.

Default is read-only shadow inference with recorded RGB/action overlays.
--execute explicitly enables physical policy targets; it is not simulation mirroring.
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def close_policy_session(worker, bus, report):
    """Stop output without a recovery trajectory or torque release.

    A worker that cannot stop still owns the port; never race it with bus I/O.
    Preflight and shadow connections close without changing motor registers.
    """
    if worker is not None:
        try:
            worker.close()
        except Exception as exc:
            report["shutdown_error"] = str(exc)
            print(f"[WARN] Worker shutdown: {exc}", flush=True)
        report["motor_shutdown_state"] = worker.shutdown_state
        return
    if bus is not None and getattr(bus, "connected", False):
        try:
            bus.close(disable_torque=False)
            report["motor_shutdown_state"] = "serial_closed_torque_unchanged"
        except Exception as exc:
            report["motor_shutdown_state"] = "shutdown_unconfirmed"
            report["shutdown_error"] = str(exc)


def start_pose_check(measured_ticks, checkpoint_home_ticks, *, align_home, closed_gripper_tick):
    """Validate an execution start, accepting a verified closed photo-home jaw.

    With --align-home the worker will first open the jaw to the checkpoint
    reference.  This lets repeated attempts begin from the required closed
    photo home without weakening the arm-joint home gate.
    """
    if len(measured_ticks) != 6 or len(checkpoint_home_ticks) != 6:
        raise ValueError("Expected six measured and home ticks")
    limit = 50 if align_home else 25
    arm_error = max(abs(actual - target) for actual, target in zip(measured_ticks[:5], checkpoint_home_ticks[:5]))
    gripper_error = abs(measured_ticks[5] - checkpoint_home_ticks[5])
    gripper_closed_error = abs(measured_ticks[5] - closed_gripper_tick)
    closed_photo_home = align_home and gripper_closed_error <= limit
    allowed = arm_error <= limit and (gripper_error <= limit or closed_photo_home)
    return {
        "allowed": allowed,
        "limit_ticks": limit,
        "arm_error_ticks": arm_error,
        "gripper_error_ticks": gripper_error,
        "gripper_closed_error_ticks": gripper_closed_error,
        "starting_from_closed_photo_home": closed_photo_home,
    }


def safe_command_segment(clearance, start_q, target_q, *, margin_m=0.005):
    """Reject a proposed motor path that would put hand geometry near the table."""
    return float(clearance.check_segment(start_q, target_q, margin=margin_m, samples=25))


def safe_measured_clearance(clearance, measured_q, *, minimum_m=0.003):
    """Stop before commanding another target when encoder pose is too low."""
    clearance_m = float(clearance.minimum(measured_q))
    if clearance_m < minimum_m:
        raise RuntimeError(f"Measured hand/table clearance {clearance_m * 1000:.2f} mm is below {minimum_m * 1000:.2f} mm")
    return clearance_m


def main():
    import math
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    from hardware_bridge import FeetechServoBus
    from isaac_vision_rl.utils.calibration import CALIB_FILE_PATH, FollowerArmCalibration, JOINT_ORDER
    from isaac_vision_rl.utils.evaluation_mirror import MirrorWorker, checked_ticks
    from isaac_vision_rl.utils.home_pose import HOME_TICKS
    from isaac_vision_rl.utils.physical_workcell import photo_home_clearance
    from isaac_vision_rl.utils.real_policy import RealPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calib", default=CALIB_FILE_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=6, help="Duration of each episode in seconds")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to execute (default: 1)")
    parser.add_argument(
        "--reset-pause", type=float, default=2.0, help="Pause duration (s) at ready pose between episodes"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--align-home",
        action="store_true",
        help="Approach checkpoint home before policy; initial error must be <=50 ticks per joint",
    )
    parser.add_argument("--allow-holding", action="store_true")
    parser.add_argument(
        "--clearance-margin",
        type=float,
        default=0.002,
        help="Minimum clearance in meters between hand and tabletop (default: 0.002 = 2.0mm)",
    )
    parser.add_argument(
        "--park-on-finish",
        action="store_true",
        help="Smoothly park arm to safe table rest blocks and release torque when all episodes finish",
    )
    parser.add_argument(
        "--tracking-limit",
        type=int,
        default=500,
        help="Maximum allowed encoder tracking error in ticks before safety trip (default: 500)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    policy = RealPolicy(args.checkpoint, args.device)
    if args.execute and policy.motion_profile == "four_stage_smooth_v1":
        parser.error(
            "This experimental checkpoint is supported for read-only shadow runs only. "
            "Its simulator gripper visibility/alignment gate is not yet matched on hardware."
        )
    if not 0 < args.seconds <= policy.duration:
        parser.error(f"--seconds must be positive and <= trained episode length {policy.duration}s")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    folder = Path(args.output or f"outputs/real_policy/{time.time_ns()}")
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "execute": args.execute,
        "mode": "real_camera_policy" if args.execute else "read_only_shadow",
        "policy_camera": "real RealSense RGB",
        "joint_feedback": "real encoders",
        "velocity": "finite difference of encoder samples using host monotonic time",
        "nominal_policy_hz": policy.hz,
        "checkpoint_rgb_profile": policy.rgb_profile,
        "requested_camera_profile": {"width": policy.camera_width, "height": policy.camera_height, "fps": policy.hz},
        "steps": 0,
        "out_of_range_proposals": 0,
        "clearance_violations": 0,
        "clearance_margin_mm": args.clearance_margin * 1000,
        "minimum_commanded_hand_table_clearance_mm": None,
        "minimum_measured_hand_table_clearance_mm": None,
        "shadow_memory": "hypothetical action/filter history; no physical commands" if not args.execute else None,
    }
    pipe, bus, worker, started, writers = rs.pipeline(), None, None, False, []
    clearance = None
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, policy.camera_width, policy.camera_height, rs.format.rgb8, policy.hz)
        profile = pipe.start(cfg)
        started = True
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        report["camera_intrinsics"] = str(intrinsics)
        frame = pipe.wait_for_frames(3000).get_color_frame()
        if not frame:
            raise RuntimeError("No real RGB camera frame")
        calib = FollowerArmCalibration(args.calib)
        bus = FeetechServoBus(args.port, 1_000_000, calib)
        ticks = bus.read_current_positions()  # Reads only; no torque/target writes.
        if ticks is None:
            raise RuntimeError("Cannot read real motor positions")
        q = calib.ticks_to_rad(ticks)
        checked_ticks(calib, q)
        rgb = np.asanyarray(frame.get_data()).copy()
        policy.reset(q)
        policy.step(rgb, q, np.zeros(6))  # Warm up actor without motor output.
        policy.reset(q)
        if args.execute:
            expected = checked_ticks(calib, policy.home)
            offsets = {
                name: {"measured": int(actual), "home": int(home), "error_ticks": int(actual - home)}
                for name, actual, home in zip(JOINT_ORDER, ticks, expected)
            }
            report["starting_pose"] = offsets
            start_check = start_pose_check(
                ticks, expected, align_home=args.align_home, closed_gripper_tick=HOME_TICKS[5]
            )
            report["starting_pose_check"] = start_check
            if not start_check["allowed"]:
                details = "; ".join(
                    f"{name}: measured={v['measured']}, home={v['home']}, " f"error={v['error_ticks']} ticks"
                    for name, v in offsets.items()
                )
                raise RuntimeError(
                    f"Arm differs from checkpoint home by {start_check['arm_error_ticks']} arm ticks; "
                    f"allowed {start_check['limit_ticks']}; refusing policy motion. {details}"
                )
            report["initial_home_error_ticks"] = max(
                start_check["arm_error_ticks"],
                min(start_check["gripper_error_ticks"], start_check["gripper_closed_error_ticks"]),
            )
            initial_target = policy.home if args.align_home else q
            clearance = photo_home_clearance()
            alignment_clearance = safe_command_segment(clearance, q, initial_target, margin_m=args.clearance_margin)
            report["initial_alignment_hand_table_clearance_mm"] = alignment_clearance * 1000
        for name in ("real", "policy_reaction"):
            writer = cv2.VideoWriter(
                str(folder / f"{name}.mp4"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                30,
                (policy.camera_width, policy.camera_height),
            )
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open {name} video encoder")
        last_safe_target = np.copy(initial_target)
        if args.execute:
            worker = MirrorWorker(
                bus,
                calib,
                initial_target,
                rate_hz=policy.hz,
                speed=400,
                watchdog=1.5,
                tracking_limit=args.tracking_limit,
                gripper_speed=600,
                gripper_tracking_limit=args.tracking_limit,
                alignment_tolerance=12,
                allow_holding=args.allow_holding,
                release_on_exit=False,
                park_on_exit=False,
            )
            worker.start(cancelled=lambda: stop)
            with worker.lock:
                q = calib.ticks_to_rad(worker.feedback["measured_ticks"])
            initial_measured_clearance = safe_measured_clearance(clearance, q)
            report["minimum_measured_hand_table_clearance_mm"] = initial_measured_clearance * 1000
            policy.reset(q)
            worker.publish(initial_target)
        print("REAL-CAMERA POLICY" if args.execute else "SHADOW: motor registers will NOT be written", flush=True)
        print(f"Recording: {folder}", flush=True)
        start = time.monotonic()
        video_frames, previous_images = 0, None
        steps = 0
        with (folder / "actions.jsonl").open("w") as log:
            for ep in range(args.episodes):
                if stop:
                    break
                print(f"\n{'='*25} EPISODE {ep + 1}/{args.episodes} {'='*25}", flush=True)
                if ep > 0:
                    print(f"[RESET] Moving arm to ready pose for Episode {ep + 1}...", flush=True)
                    if worker is not None:
                        # 1. Retract up vertically to clear desk
                        with worker.lock:
                            last_ticks = list(worker.feedback.get("measured_ticks") or calib.rad_to_ticks(q))
                        last_q = calib.ticks_to_rad(last_ticks)
                        wp_lift_q = list(last_q)
                        wp_lift_q[1] = min(last_q[1], -1.15)  # Lift shoulder
                        wp_lift_q[2] = min(last_q[2], 1.05)  # Unfold elbow to raise hand
                        wp_lift_q[3] = 0.90  # Level wrist
                        wp_lift_q[5] = 1.20  # Open gripper

                        steps_lift = max(1, int(1.0 * policy.hz))
                        for s in range(1, steps_lift + 1):
                            if stop:
                                break
                            alpha = 0.5 * (1.0 - math.cos((s / steps_lift) * math.pi))
                            interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(last_q, wp_lift_q)]
                            worker.publish(interp_q)
                            time.sleep(1.0 / policy.hz)

                        # 2. Move smoothly to policy.home
                        target_home = policy.home
                        steps_home = max(1, int(1.2 * policy.hz))
                        for s in range(1, steps_home + 1):
                            if stop:
                                break
                            alpha = 0.5 * (1.0 - math.cos((s / steps_home) * math.pi))
                            interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(wp_lift_q, target_home)]
                            worker.publish(interp_q)
                            time.sleep(1.0 / policy.hz)

                        # 3. Settle / pause at reset pose
                        if args.reset_pause > 0:
                            print(
                                f"[RESET] Pausing {args.reset_pause:.1f}s for environment repositioning...", flush=True
                            )
                            settle_steps = max(1, int(args.reset_pause * policy.hz))
                            for _ in range(settle_steps):
                                if stop:
                                    break
                                worker.publish(target_home)
                                pipe.poll_for_frames()
                                time.sleep(1.0 / policy.hz)

                        with worker.lock:
                            q = calib.ticks_to_rad(worker.feedback["measured_ticks"])
                    else:
                        ticks = bus.read_current_positions()
                        if ticks is not None:
                            q = calib.ticks_to_rad(ticks)

                    policy.reset(q)
                    last_safe_target = np.copy(target_home)
                    if worker is not None:
                        worker.publish(target_home)

                ep_start = time.monotonic()
                last_sample = ep_start
                previous_q = q.copy()
                ep_steps = 0

                while not stop and time.monotonic() - ep_start < args.seconds:
                    frame = pipe.wait_for_frames(500).get_color_frame()
                    if not frame:
                        raise RuntimeError("Real RGB frame missing")
                    rgb = np.asanyarray(frame.get_data()).copy()
                    snapshot = {}
                    if worker is None:
                        ticks = bus.read_current_positions()
                        stamp = time.monotonic()
                    else:
                        worker.check()
                        with worker.lock:
                            snapshot = dict(worker.feedback)
                        ticks = snapshot.get("measured_ticks")
                        stamp = snapshot.get("measured_monotonic_s", 0)
                        if time.monotonic() - stamp > 0.1:
                            raise RuntimeError("Real encoder feedback became stale")
                    if ticks is None:
                        raise RuntimeError("Real encoder read failed")
                    q = calib.ticks_to_rad(ticks)
                    lower, upper = calib.get_joint_limits_rad()
                    q = np.clip(q, lower, upper)
                    measured_clearance = None
                    if clearance is not None:
                        try:
                            measured_clearance = safe_measured_clearance(
                                clearance, q, minimum_m=max(0.0005, args.clearance_margin * 0.5)
                            )
                        except RuntimeError:
                            report["measured_clearance_warnings"] = report.get("measured_clearance_warnings", 0) + 1
                            measured_clearance = float(clearance.minimum(q))
                        previous_min = report["minimum_measured_hand_table_clearance_mm"]
                        report["minimum_measured_hand_table_clearance_mm"] = 1000 * min(
                            measured_clearance,
                            float("inf") if previous_min is None else previous_min / 1000,
                        )
                    velocity = (q - previous_q) / (stamp - last_sample) if stamp > last_sample else np.zeros(6)
                    if stamp > last_sample:
                        previous_q, last_sample = q.copy(), stamp
                    raw, target, state = policy.step(rgb, q, velocity)
                    lower, upper = calib.get_joint_limits_rad()
                    target_clamped = np.clip(target, lower, upper)
                    target_valid = bool(np.allclose(target, target_clamped, atol=1e-3))
                    if not target_valid:
                        report["out_of_range_proposals"] += 1
                    target_ticks = calib.rad_to_ticks(target_clamped).tolist()
                    command_clearance = None
                    if worker is not None:
                        try:
                            command_clearance = safe_command_segment(
                                clearance, q, target_clamped, margin_m=args.clearance_margin
                            )
                            last_safe_target = np.copy(target_clamped)
                        except ValueError:
                            report["clearance_violations"] += 1
                            target_clamped[:5] = last_safe_target[:5]
                            command_clearance = args.clearance_margin
                        previous_min = report["minimum_commanded_hand_table_clearance_mm"]
                        report["minimum_commanded_hand_table_clearance_mm"] = 1000 * min(
                            command_clearance,
                            float("inf") if previous_min is None else previous_min / 1000,
                        )
                        worker.publish(target_clamped)
                        policy.set_executed_target(target_clamped)
                    elapsed = time.monotonic() - start
                    log.write(
                        json.dumps(
                            {
                                "episode": ep + 1,
                                "step": steps,
                                "ep_step": ep_steps,
                                "wall_s": elapsed,
                                "camera_timestamp_ms": frame.get_timestamp(),
                                "joint_ticks": list(ticks),
                                "joint_rad": q.tolist(),
                                "state_25": state.tolist(),
                                "raw_action": raw.tolist(),
                                "target_rad": target.tolist(),
                                "target_ticks": target_ticks,
                                "target_within_calibration": target_valid,
                                "predicted_hand_table_clearance_mm": None if command_clearance is None else 1000 * command_clearance,
                                "measured_hand_table_clearance_mm": None if measured_clearance is None else 1000 * measured_clearance,
                                "motor_write": args.execute,
                                "latest_sent_ticks": snapshot.get("command_ticks"),
                                "feedback_monotonic_s": snapshot.get("measured_monotonic_s"),
                            }
                        )
                        + "\n"
                    )
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    overlay = bgr.copy()
                    cv2.putText(
                        overlay,
                        f"LIVE EP {ep + 1}/{args.episodes}" if args.execute else f"SHADOW EP {ep + 1}/{args.episodes}",
                        (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )
                    if not target_valid:
                        cv2.putText(
                            overlay,
                            "PROPOSAL OUTSIDE HARDWARE LIMITS",
                            (4, 155),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1,
                        )
                    for i, name in enumerate(JOINT_ORDER):
                        cv2.putText(
                            overlay,
                            f"{name}: action {raw[i]:+.2f} target {target[i]:+.2f} rad",
                            (4, 35 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            (255, 255, 255),
                            1,
                        )
                    images = (bgr, overlay)
                    slot = int(elapsed * 30)
                    while video_frames <= slot:
                        chosen = previous_images if video_frames < slot and previous_images is not None else images
                        for writer, image in zip(writers, chosen):
                            writer.write(image)
                        video_frames += 1
                    previous_images = images
                    if not args.headless:
                        cv2.imshow("Real camera policy reaction", overlay)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop = True
                    steps += 1
                    ep_steps += 1
                    report["steps"] = steps
                    if args.execute and ep_steps > 10 and (time.monotonic() - ep_start) / ep_steps > 3 / policy.hz:
                        raise RuntimeError("Real policy loop is too slow for the checkpoint's control rate")
                print(f"[INFO] Episode {ep + 1}/{args.episodes} completed ({ep_steps} steps).", flush=True)

        if args.execute and worker is not None and not stop:
            print("[INFO] All episodes completed. Returning arm to ready stance...", flush=True)
            with worker.lock:
                last_ticks = list(worker.feedback.get("measured_ticks", []))
            if last_ticks:
                last_q = calib.ticks_to_rad(last_ticks)
                wp_lift_q = list(last_q)
                wp_lift_q[1] = min(last_q[1], -1.15)
                wp_lift_q[2] = min(last_q[2], 1.05)
                wp_lift_q[3] = 0.90
                wp_lift_q[5] = 1.20
                steps_lift = max(1, int(1.0 * policy.hz))
                for s in range(1, steps_lift + 1):
                    alpha = 0.5 * (1.0 - math.cos((s / steps_lift) * math.pi))
                    interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(last_q, wp_lift_q)]
                    worker.publish(interp_q)
                    time.sleep(1.0 / policy.hz)
                target_home = policy.home
                steps_home = max(1, int(1.2 * policy.hz))
                for s in range(1, steps_home + 1):
                    alpha = 0.5 * (1.0 - math.cos((s / steps_home) * math.pi))
                    interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(wp_lift_q, target_home)]
                    worker.publish(interp_q)
                    time.sleep(1.0 / policy.hz)
                print("[INFO] Arm safely at ready stance.", flush=True)

            if args.park_on_finish:
                print("[INFO] Parking arm onto table rest blocks...", flush=True)
                rest_q = calib.ticks_to_rad(list(HOME_TICKS))
                steps_rest = max(1, int(1.2 * policy.hz))
                for s in range(1, steps_rest + 1):
                    alpha = 0.5 * (1.0 - math.cos((s / steps_rest) * math.pi))
                    interp_q = [(1.0 - alpha) * a + alpha * b for a, b in zip(policy.home, rest_q)]
                    worker.publish(interp_q)
                    time.sleep(1.0 / policy.hz)
                print("[INFO] Arm safely resting on table blocks.", flush=True)
                worker.release_on_exit = True
        report.update(
            steps=steps,
            wall_seconds=time.monotonic() - start,
            video_frames=video_frames,
            episodes_requested=args.episodes,
            stopped_by_user=stop,
            observed_inference_hz=steps / (time.monotonic() - start),
        )
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            close_policy_session(worker, bus, report)
            if args.execute:
                print("[INFO] Policy session closed; torque remains at the last commanded targets.", flush=True)
        finally:
            for writer in writers:
                writer.release()
            if started:
                pipe.stop()
            if not args.headless:
                cv2.destroyAllWindows()
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Real-camera policy report: {folder / 'report.json'}", flush=True)
            if report["steps"] == 0:
                print("No policy frames recorded: the attempt did not start.", flush=True)


if __name__ == "__main__":
    main()
