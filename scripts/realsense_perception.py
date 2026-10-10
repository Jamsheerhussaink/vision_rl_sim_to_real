#!/usr/bin/env python3
"""Read-only RealSense cube localization. This script never commands robot motors."""

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", required=True, help="Measured color-optical-camera to robot-base transform JSON"
    )
    parser.add_argument("--serial")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", default="outputs/perception/realsense.json")
    args = parser.parse_args()
    import cv2
    import numpy as np

    from isaac_vision_rl.perception.cube import YellowCubeEstimator, load_calibration

    estimator = YellowCubeEstimator(load_calibration(args.calibration))
    try:
        # C++ SDK exports are dynamic; keep the typing boundary local.
        rs = cast(Any, import_module("pyrealsense2"))
    except ImportError:
        parser.error("Install the perception optional dependencies in env_isaaclab first")
    if args.frames < 1:
        parser.error("--frames must be positive")
    pipe = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cloud = rs.pointcloud()
    records = []
    started = False
    try:
        profile = pipe.start(cfg)
        started = True
        device = profile.get_device()
        print(
            "Camera:", device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number), flush=True
        )
        for _ in range(15):
            pipe.wait_for_frames()
        for _ in range(args.frames):
            frames = pipe.wait_for_frames()
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                continue
            if abs(depth.get_timestamp() - color.get_timestamp()) > 100:
                records.append({"valid": False, "reason": "RGB/depth timestamps differ by more than 100 ms"})
                continue
            rgb = np.asanyarray(color.get_data())
            # SDK deprojects native depth and maps it into distorted color pixels.
            # This avoids treating raw RGB pixels as an undistorted pinhole image.
            cloud.map_to(color)
            points = cloud.calculate(depth)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            uv = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            ext = depth.profile.get_extrinsics_to(color.profile)
            rotation = np.array(ext.rotation).reshape(3, 3, order="F")
            color_xyz = vertices @ rotation.T + np.array(ext.translation)
            pixels = uv * np.array([rgb.shape[1], rgb.shape[0]])
            estimate = estimator.estimate_cloud(rgb, color_xyz, pixels).to_dict()
            estimate["host_monotonic_s"] = time.monotonic()
            records.append(estimate)
            print(json.dumps(estimate), flush=True)
            if args.preview:
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    image,
                    str(estimate["center_base_m"] or estimate["reason"]),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow("RealSense cube estimate - read only", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if started:
            pipe.stop()
        if args.preview:
            cv2.destroyAllWindows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"robot_commands_sent": False, "calibration": args.calibration, "frames": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
