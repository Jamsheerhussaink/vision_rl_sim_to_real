"""Live real-vs-sim camera check: move the sim camera until it matches the D455.

    python scripts/live_compare.py                    # start from scene.xml pose
    python scripts/live_compare.py --z 0.52 --fovy 65 # override start height / fov

Window (all panels at the exact 424x240 camera resolution, no scaling):
    real        | sim
    50/50 blend | real with sim edges in red

Move the sim camera with the sliders in the window, or with keys (click the
window first so it has keyboard focus; 1 cm / 1 deg steps, SHIFT for 5x):
    w/s  forward/back     a/d  left/right     r/f  up/down
    j/l  yaw left/right   i/k  pitch up/down  [ ]  fovy -/+
    p    print the <camera> line for scene.xml
Stop with q/ESC, by closing the window, or Ctrl+C in the terminal.

Reads the colour stream straight from V4L2 (pyrealsense2 gets no frames over
the current USB 2.1 link). The sim robot sits in keyframe 0, so put the real
arm in the same pose before matching.
"""
import argparse
import math
import time

import cv2
import mujoco
import numpy as np

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME
from vision_rl.envs import SCENE_XML

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/video4", help="V4L2 colour node")
parser.add_argument("--camera", default=CAMERA_NAME)
parser.add_argument("--x", type=float)
parser.add_argument("--y", type=float)
parser.add_argument("--z", type=float)
parser.add_argument("--fovy", type=float)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
cam = model.camera(args.camera)

# Camera state: position, yaw (heading in the xy plane), pitch (down is positive).
pos = data.cam_xpos[cam.id].copy()
fwd = -data.cam_xmat[cam.id].reshape(3, 3)[:, 2]
yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
pitch = math.degrees(math.asin(-fwd[2]))
fovy = float(cam.fovy[0])
for i, v in enumerate((args.x, args.y, args.z)):
    if v is not None:
        pos[i] = v
if args.fovy is not None:
    fovy = args.fovy


def axes(yaw_deg, pitch_deg):
    """Camera right, up and forward vectors (MuJoCo cameras look along -z)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
    r = np.array([math.sin(y), -math.cos(y), 0.0])
    return r, np.cross(r, f), f


def apply_camera():
    r, u, f = axes(yaw, pitch)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([r, u, -f]).flatten())
    model.cam_pos[cam.id] = pos           # camera is a worldbody child: local == world
    model.cam_quat[cam.id] = quat
    model.cam_fovy[cam.id] = fovy


def xml_line():
    r, u, _ = axes(yaw, pitch)
    xy = " ".join(f"{v:.7f}".rstrip("0").rstrip(".") for v in (*r, *u))
    return (f'<camera name="{args.camera}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}"\n'
            f'        xyaxes="{xy}" fovy="{fovy:.1f}"/>')


cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise SystemExit(f"cannot open {args.device}")

STEP, ANG = 0.01, 1.0
moves = {  # key -> (dx_fwd, dx_right, dz, dyaw, dpitch, dfovy)
    "w": (STEP, 0, 0, 0, 0, 0), "s": (-STEP, 0, 0, 0, 0, 0),
    "d": (0, STEP, 0, 0, 0, 0), "a": (0, -STEP, 0, 0, 0, 0),
    "r": (0, 0, STEP, 0, 0, 0), "f": (0, 0, -STEP, 0, 0, 0),
    "j": (0, 0, 0, ANG, 0, 0), "l": (0, 0, 0, -ANG, 0, 0),
    "k": (0, 0, 0, 0, ANG, 0), "i": (0, 0, 0, 0, -ANG, 0),
    "]": (0, 0, 0, 0, 0, ANG), "[": (0, 0, 0, 0, 0, -ANG),
}
shifted = {"W": "w", "S": "s", "D": "d", "A": "a", "R": "r", "F": "f",
           "J": "j", "L": "l", "K": "k", "I": "i", "}": "]", "{": "["}

win = "real vs sim (424x240)"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

# Sliders mirror the camera state, so it can be moved with the mouse too.
SLIDERS = [  # name, min, max, units per metre/degree
    ("x mm", -1000, 1500, 1000), ("y mm", -1500, 1500, 1000), ("z mm", 400, 1200, 1000),
    ("yaw x0.1deg", -1800, 1800, 10), ("pitch x0.1deg", -900, 900, 10),
    ("fovy x0.1deg", 100, 1200, 10),
]


def state():
    return [pos[0], pos[1], pos[2], yaw, pitch, fovy]


def slider_values(values):
    return [int(round(v * s)) for v, (_, _, _, s) in zip(values, SLIDERS)]


for (name, lo, hi, _), v in zip(SLIDERS, slider_values(state())):
    cv2.createTrackbar(name, win, 0, hi - lo, lambda _: None)
    cv2.setTrackbarMin(name, win, lo)
    cv2.setTrackbarMax(name, win, hi)
    cv2.setTrackbarPos(name, win, v)
shown = slider_values(state())
last_key = "-"
frames, t_status = 0, time.time()
try:
    with mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH) as renderer:
        while True:
            ok, real = cap.read()
            if not ok:
                print("no frame from", args.device)
                continue
            if real.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
                raise SystemExit(f"camera gave {real.shape[1]}x{real.shape[0]}, "
                                 f"expected {CAM_WIDTH}x{CAM_HEIGHT}")

            now = [cv2.getTrackbarPos(n, win) for n, *_ in SLIDERS]
            if now != shown:   # a slider was dragged
                vals = [v / s for v, (_, _, _, s) in zip(now, SLIDERS)]
                pos[:] = vals[:3]
                yaw, pitch, fovy = vals[3:]
                shown = now

            apply_camera()
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=args.camera)
            sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            edges = real.copy()
            edges[cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 40, 120) > 0] = (0, 0, 255)
            grid = np.vstack([np.hstack([real, sim]), np.hstack([blend, edges])])
            hud = (f"pos {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}  yaw {yaw:.1f}  "
                   f"pitch {pitch:.1f}  fovy {fovy:.1f}  key {last_key}")
            cv2.putText(grid, hud, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow(win, grid)
            frames += 1
            if time.time() - t_status > 1.0:   # status line, once a second
                print(f"[{frames} frames] {hud}", flush=True)
                t_status = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:   # no key pressed
                continue
            ch = chr(key)
            last_key = ch
            if ch == "p":
                print(xml_line(), flush=True)
                continue
            scale = 5 if ch in shifted else 1
            ch = shifted.get(ch, ch)
            if ch in moves:
                dfwd, dright, dz, dyaw, dpitch, dfov = (scale * v for v in moves[ch])
                r, _, f = axes(yaw, 0.0)
                pos += dfwd * f + dright * r + np.array([0, 0, dz])
                yaw += dyaw
                pitch += dpitch
                fovy = min(max(fovy + dfov, 10.0), 120.0)
                shown = slider_values(state())
                for (name, *_), v in zip(SLIDERS, shown):
                    cv2.setTrackbarPos(name, win, v)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
print("final camera:\n" + xml_line())
