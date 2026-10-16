"""Show a simulation reference over live RGB for manual workcell alignment.

This is a two-dimensional visual guide, not calibrated robot localization.
No motor interfaces or policy inference are used.
"""
import argparse
from pathlib import Path


def placement_view(bgr, reference, ghost=True):
    import cv2
    import numpy as np

    # Preserve the reference aspect ratio instead of stretching its geometry.
    height, width = bgr.shape[:2]
    scale = min(width / reference.shape[1], height / reference.shape[0])
    rw, rh = round(reference.shape[1] * scale), round(reference.shape[0] * scale)
    x, y = (width - rw) // 2, (height - rh) // 2
    ref = cv2.resize(reference, (rw, rh))
    result = bgr.copy()
    region = result[y:y + rh, x:x + rw]
    if ghost:
        region[:] = cv2.addWeighted(region, 0.70, ref, 0.30, 0)
        # Cyan edges show the reference geometry without marking live pixels
        # as if they had been detected or calibrated.
        edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 65, 130)
        edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))
        region[edges > 0] = (255, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, width / 1800)
    thickness = max(1, round(width / 640))
    lines = (
        "CYAN = SIM TARGET | Align robot base/footprint first",
        "Visual alignment only; camera angle/height must also match",
        "G: toggle ghost   Q: close   No motor commands",
    )
    line_height = max(20, round(height * 0.04))
    cv2.rectangle(result, (0, 0), (width, line_height * 3 + 10), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(result, line, (10, (i + 1) * line_height), font,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def main():
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=root / "outputs/rgb_scene_reference.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 runs until Q or Ctrl+C")
    parser.add_argument("--output", type=Path, help="Save the first guide frame after warm-up")
    args = parser.parse_args()
    if args.frames < 0 or (args.headless and args.frames == 0):
        parser.error("Use a positive --frames count with --headless")
    reference = cv2.imread(str(args.reference))
    if reference is None:
        parser.error(f"Cannot read reference: {args.reference}")

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    started, ghost = False, True
    try:
        pipeline.start(config)
        started = True
        for _ in range(20):
            pipeline.wait_for_frames(5000)
        if not args.headless:
            cv2.namedWindow("Robot placement - cyan simulation guide", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Robot placement - cyan simulation guide", 1060, 600)
        count = 0
        while args.frames == 0 or count < args.frames:
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                continue
            bgr = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_RGB2BGR)
            guide = placement_view(bgr, reference, ghost)
            if count == 0 and args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.output), guide):
                    raise RuntimeError(f"Cannot save {args.output}")
                print(f"Guide saved: {args.output.resolve()}", flush=True)
            count += 1
            if not args.headless:
                cv2.imshow("Robot placement - cyan simulation guide", guide)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(
                    "Robot placement - cyan simulation guide", cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
                if key == ord("g"):
                    ghost = not ghost
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
