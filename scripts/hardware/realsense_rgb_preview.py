#!/usr/bin/env python3
"""RGB-only preview: no depth, policy inference or motor commands."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def find_realsense_rgb_device() -> str:
    from pathlib import Path
    for p in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            iface = (p / "device/bInterfaceNumber").read_text().strip()
            name = (p / "name").read_text().strip()
            if iface == "03" and "RealSense" in name:
                return f"/dev/{p.name}"
        except Exception:
            continue
    return "/dev/video4"


def run_v4l2_stream(device: str, frames: int, headless: bool, width: int = 1280, height: int = 720, fps: int = 30):
    import cv2
    import numpy as np
    import time
    import torch
    from isaac_vision_rl.utils.rgb_image import actor_rgb

    print(f"[V4L2] Connecting to RealSense RGB sensor on {device}...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {device}")
        return

    print(f"RGB profile: {width} {height} {fps} (V4L2)")
    # Discard warm-up frames
    for _ in range(5):
        cap.read()

    timestamps = []
    count = 0
    try:
        while True:
            if frames > 0 and count >= frames:
                break
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue
            count += 1
            timestamps.append(time.time() * 1000)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = actor_rgb(torch.from_numpy(rgb.copy()).unsqueeze(0))[0]
            preview = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if not headless:
                cv2.imshow("RGB capture", bgr)
                cv2.imshow("Actor image", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
            print("Observed timestamp FPS:", 1000 * (len(timestamps)-1) / (timestamps[-1]-timestamps[0]))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main():
    import cv2
    import numpy as np
    import torch
    from isaac_vision_rl.utils.rgb_image import actor_rgb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280, help="RGB image width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="RGB image height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="RGB frame rate (default: 30)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames (default: 300, set to 0 for unlimited)")
    parser.add_argument("--continuous", action="store_true", help="Run stream indefinitely until 'q' or Ctrl+C")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--v4l2", action="store_true", help="Force direct V4L2 device capture")
    args = parser.parse_args()
    if args.continuous:
        args.frames = 0
    if args.frames < 0:
        parser.error("--frames cannot be negative (use 0 or --continuous for unlimited)")

    if args.v4l2:
        run_v4l2_stream(find_realsense_rgb_device(), args.frames, args.headless, args.width, args.height, args.fps)
        return

    import pyrealsense2 as rs
    pipe, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    started, timestamps = False, []
    try:
        profile = pipe.start(config)
        started = True
        colour = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print("RGB profile:", colour.width(), colour.height(), colour.fps())
        print("Intrinsics:", colour.get_intrinsics())
        count = 0
        while True:
            if args.frames > 0 and count >= args.frames:
                break
            frame = pipe.wait_for_frames(3000).get_color_frame()
            if not frame:
                continue
            count += 1
            timestamps.append(frame.get_timestamp())
            rgb = np.asanyarray(frame.get_data())
            image = actor_rgb(torch.from_numpy(rgb.copy()).unsqueeze(0))[0]
            preview = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if not args.headless:
                cv2.imshow("RGB capture", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("Actor image", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
            print("Observed timestamp FPS:", 1000 * (len(timestamps)-1) / (timestamps[-1]-timestamps[0]))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[WARN] pyrealsense2 streaming interrupted ({e}). Falling back to native V4L2 capture...")
        if started:
            try:
                pipe.stop()
            except Exception:
                pass
            started = False
        if not args.headless:
            cv2.destroyAllWindows()
        run_v4l2_stream(find_realsense_rgb_device(), args.frames, args.headless, args.width, args.height, args.fps)
        return
    finally:
        if started:
            try:
                pipe.stop()
            except Exception:
                pass
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
