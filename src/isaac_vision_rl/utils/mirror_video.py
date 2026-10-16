"""Wall-clock video recording of simulation and a RealSense RGB reference camera."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np


def letterbox(frame, width, height):
    import cv2
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
    return canvas


class MirrorVideoRecorder:
    """Camera/encoding thread; never accesses the policy or motor bus.

    Movies share a host wall-clock timeline at 30 fps. Slow simulation frames are
    repeated, preserving real duration rather than accelerating slow inference.
    Frame timestamps describe host acquisition, not hardware camera synchronization.
    """

    def __init__(self, folder, initial_rgb):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.sim = np.asarray(initial_rgb, dtype=np.uint8).copy()
        self.sim_stamp = time.monotonic()
        self.phase = "initial_alignment"
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.error = None
        self.frames = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="mirror-video")

    def start(self):
        self.thread.start()
        if not self.ready.wait(10):
            self.close()
            raise TimeoutError("Real camera recording did not start")
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Video recording failed: {self.error}") from self.error

    def publish(self, rgb=None, phase="policy_evaluation"):
        self.check()
        with self.lock:
            if rgb is not None:
                self.sim = np.asarray(rgb, dtype=np.uint8).copy()
                self.sim_stamp = time.monotonic()
            self.phase = phase

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(5)
            if self.thread.is_alive():
                raise RuntimeError("Camera recording thread did not stop")
        self.check()

    def _run(self):
        pipe, started, writers = None, False, []
        metadata = {"fps": 30, "camera": "RealSense RGB, 640x480 requested",
                    "timing": "host monotonic acquisition; approximate sync, includes camera buffering",
                    "simulation_frames": "latest available frame repeated on wall-clock timeline",
                    "frames": 0}
        try:
            import cv2
            import pyrealsense2 as rs
            pipe, config = rs.pipeline(), rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipe.start(config)
            started = True
            colour = profile.get_stream(rs.stream.color).as_video_stream_profile()
            metadata["real_profile"] = {"width": colour.width(), "height": colour.height(), "fps": colour.fps()}
            for name, size in (("sim", (424, 240)), ("real", (640, 480)), ("comparison", (848, 268))):
                writer = cv2.VideoWriter(str(self.folder / f"{name}.mp4"),
                                         cv2.VideoWriter.fourcc(*"mp4v"), 30, size)
                writers.append(writer)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open {name}.mp4 encoder")
            first = pipe.wait_for_frames(3000).get_color_frame()
            if not first:
                raise RuntimeError("No RGB frame received")
            real = np.asanyarray(first.get_data()).copy()
            start = time.monotonic()
            metadata["start_unix_s"] = time.time()
            previous = None
            self.ready.set()
            with (self.folder / "frames.jsonl").open("w") as timing:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    with self.lock:
                        sim, sim_stamp, phase = self.sim.copy(), self.sim_stamp, self.phase
                    sim = letterbox(cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), 424, 240)
                    real_small = letterbox(real, 424, 240)
                    pair = np.zeros((268, 848, 3), dtype=np.uint8)
                    pair[28:, :424], pair[28:, 424:] = sim, real_small
                    cv2.putText(pair, f"SIM | {phase}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    cv2.putText(pair, "REAL RGB | reference camera", (432, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    current = (sim, real, pair, now, sim_stamp, phase)
                    # Fill missed output slots with the previous observation, not future images.
                    slot = int((now - start) * 30)
                    while self.frames <= slot:
                        chosen = previous if self.frames < slot and previous is not None else current
                        for writer, frame in zip(writers, chosen[:3]):
                            writer.write(frame)
                        timing.write(json.dumps({"frame": self.frames, "video_time_s": self.frames / 30,
                                                 "real_host_time_s": chosen[3] - start,
                                                 "sim_host_time_s": chosen[4] - start,
                                                 "phase": chosen[5]}) + "\n")
                        self.frames += 1
                    previous = current
                    frame = pipe.wait_for_frames(1000).get_color_frame()
                    if not frame:
                        raise RuntimeError("RGB stream stopped producing frames")
                    real = np.asanyarray(frame.get_data()).copy()
        except Exception as exc:
            self.error = exc
        finally:
            for writer in writers:
                writer.release()
            if started:
                try:
                    pipe.stop()
                except Exception as exc:
                    self.error = self.error or exc
            metadata["frames"] = self.frames
            metadata["duration_s"] = self.frames / 30
            metadata["error"] = str(self.error) if self.error else None
            (self.folder / "recording.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.ready.set()
