"""Set up the sim to match the real scene, watching the real D455 live.

    python scripts/setup.py

Left: real | sim on top, 50/50 blend | real with sim edges (red) below, each
at the exact 424x240 camera resolution. Right: sliders, in tabs, for the
camera (position, yaw/pitch/roll, fovy), robot base (x, y, yaw), the six
joints (degrees), the cube and the box (x, y, yaw).

The sim is posed only (no physics), so nothing falls or moves on its own.
"Save" writes everything back to scene.xml / robot.xml / cube.xml / box.xml
(a .bak copy of each file is written first) and updates the "home" keyframe.
Stop with the Quit button, by closing the window, or Ctrl+C in the terminal.
"""
import argparse
import math
import os
import re
import shutil

import cv2
import mujoco
import numpy as np
import PIL.Image
import PIL.ImageTk
import tkinter as tk
from tkinter import ttk

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME
from vision_rl.envs import ASSETS_DIR, SCENE_XML

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/video4", help="V4L2 colour node")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

cam_id = model.camera(CAMERA_NAME).id
base_id = model.body("base").id
box_id = model.body("target_box").id
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CUBE_QPOS = model.jnt_qposadr[model.joint("cube_joint").id]


# ---------- small rotation helpers ----------
def yaw_quat(deg):
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def quat_yaw(q):
    return math.degrees(2 * math.atan2(q[3], q[0]))


def cam_axes(yaw, pitch, roll):
    """Camera right, up, forward (world) from yaw / pitch (down +) / roll, degrees."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
    r0 = np.array([math.sin(y), -math.cos(y), 0.0])
    u0 = np.cross(r0, f)
    return math.cos(r) * r0 + math.sin(r) * u0, -math.sin(r) * r0 + math.cos(r) * u0, f


def cam_angles(mat):
    right, up = mat[:, 0], mat[:, 1]
    f = -mat[:, 2]
    yaw = math.degrees(math.atan2(f[1], f[0]))
    pitch = math.degrees(math.asin(np.clip(-f[2], -1, 1)))
    r0, u0, _ = cam_axes(yaw, pitch, 0.0)
    return yaw, pitch, math.degrees(math.atan2(np.dot(right, u0), np.dot(right, r0)))


# ---------- the editable values: (tab, name, min, max, step, initial) ----------
cyaw, cpitch, croll = cam_angles(data.cam_xmat[cam_id].reshape(3, 3))
cpos = data.cam_xpos[cam_id]
SPECS = [
    ("Camera", "cam x (m)", -1.0, 1.5, 0.001, cpos[0]),
    ("Camera", "cam y (m)", -1.5, 1.5, 0.001, cpos[1]),
    ("Camera", "cam z (m)", 0.40, 1.20, 0.001, cpos[2]),
    ("Camera", "cam yaw (deg)", -180, 180, 0.01, cyaw),
    ("Camera", "cam pitch (deg)", -90, 90, 0.01, cpitch),
    ("Camera", "cam roll (deg)", -45, 45, 0.01, croll),
    ("Camera", "cam fovy (deg)", 20, 100, 0.05, model.cam_fovy[cam_id]),
    ("Robot", "base x (m)", -0.5, 0.8, 0.001, model.body_pos[base_id][0]),
    ("Robot", "base y (m)", -0.6, 0.6, 0.001, model.body_pos[base_id][1]),
    ("Robot", "base yaw (deg)", -180, 180, 0.01, quat_yaw(model.body_quat[base_id])),
]
for j in JOINTS:
    lo, hi = np.degrees(model.joint(j).range)
    SPECS.append(("Robot", f"{j} (deg)", lo, hi, 0.01,
                  math.degrees(data.qpos[model.jnt_qposadr[model.joint(j).id]])))
SPECS += [
    ("Objects", "cube x (m)", -0.5, 0.9, 0.001, data.qpos[CUBE_QPOS]),
    ("Objects", "cube y (m)", -0.6, 0.6, 0.001, data.qpos[CUBE_QPOS + 1]),
    ("Objects", "cube yaw (deg)", -180, 180, 0.01, quat_yaw(data.qpos[CUBE_QPOS + 3:CUBE_QPOS + 7])),
    ("Objects", "box x (m)", -0.5, 0.9, 0.001, model.body_pos[box_id][0]),
    ("Objects", "box y (m)", -0.6, 0.6, 0.001, model.body_pos[box_id][1]),
    ("Objects", "box yaw (deg)", -180, 180, 0.01, quat_yaw(model.body_quat[box_id])),
    ("View", "blend real %", 0, 100, 1, 50),
]

root = tk.Tk()
root.title("setup env: real vs sim")
vars_ = {name: tk.DoubleVar(value=round(float(init), 4)) for _, name, *_, init in SPECS}
v = lambda name: vars_[name].get()   # noqa: E731


def apply_to_sim():
    """Push the slider values into the MuJoCo model / qpos and recompute poses."""
    right, up, f = cam_axes(v("cam yaw (deg)"), v("cam pitch (deg)"), v("cam roll (deg)"))
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([right, up, -f]).flatten())
    model.cam_pos[cam_id] = [v("cam x (m)"), v("cam y (m)"), v("cam z (m)")]
    model.cam_quat[cam_id] = q
    model.cam_fovy[cam_id] = v("cam fovy (deg)")

    model.body_pos[base_id][:2] = [v("base x (m)"), v("base y (m)")]
    model.body_quat[base_id] = yaw_quat(v("base yaw (deg)"))
    for j in JOINTS:
        data.qpos[model.jnt_qposadr[model.joint(j).id]] = math.radians(v(f"{j} (deg)"))

    data.qpos[CUBE_QPOS:CUBE_QPOS + 2] = [v("cube x (m)"), v("cube y (m)")]
    data.qpos[CUBE_QPOS + 3:CUBE_QPOS + 7] = yaw_quat(v("cube yaw (deg)"))
    model.body_pos[box_id][:2] = [v("box x (m)"), v("box y (m)")]
    model.body_quat[box_id] = yaw_quat(v("box yaw (deg)"))
    mujoco.mj_forward(model, data)


# ---------- saving back to the XML files ----------
def fmt(*xs):
    return " ".join(f"{x:.7f}".rstrip("0").rstrip(".") if abs(x) > 1e-9 else "0" for x in xs)


def set_attrs(text, tag_start, attrs):
    """Replace attribute values inside the first element starting with `tag_start`."""
    m = re.search(re.escape(tag_start) + r"[^>]*>", text, re.S)
    if m is None:
        raise RuntimeError(f"{tag_start} not found")
    elem = m.group(0)
    for key, val in attrs.items():
        elem, n = re.subn(rf'(\s{key}=")[^"]*(")', rf"\g<1>{val}\g<2>", elem)
        if n != 1:
            raise RuntimeError(f"{key} not found once in {tag_start}")
    return text[:m.start()] + elem + text[m.end():]


def save():
    apply_to_sim()
    right, up, _ = cam_axes(v("cam yaw (deg)"), v("cam pitch (deg)"), v("cam roll (deg)"))
    base_q = yaw_quat(v("base yaw (deg)"))
    box_q = yaw_quat(v("box yaw (deg)"))
    cube_q = yaw_quat(v("cube yaw (deg)"))
    cube_z = data.qpos[CUBE_QPOS + 2]
    joints = [data.qpos[model.jnt_qposadr[model.joint(j).id]] for j in JOINTS]
    cube_qpos = [v("cube x (m)"), v("cube y (m)"), cube_z, *cube_q]

    edits = {
        "scene.xml": [
            (f'<camera name="{CAMERA_NAME}"', {
                "pos": fmt(*model.cam_pos[cam_id]), "xyaxes": fmt(*right, *up),
                "fovy": f"{v('cam fovy (deg)'):.2f}"}),
            ('<key name="home"', {"qpos": f"{fmt(*joints)}  {fmt(*cube_qpos)}",
                                   "ctrl": fmt(*joints)}),
        ],
        "robot.xml": [('<body name="base"', {"pos": fmt(*model.body_pos[base_id]),
                                             "quat": fmt(*base_q)})],
        "cube.xml": [('<body name="cube"', {"pos": fmt(*cube_qpos[:3]), "quat": fmt(*cube_q)})],
        "box.xml": [('<body name="target_box"', {"pos": fmt(*model.body_pos[box_id]),
                                                  "quat": fmt(*box_q)})],
    }
    new_texts = {}
    for fname, items in edits.items():        # build every file first, then write
        path = os.path.join(ASSETS_DIR, fname)
        text = open(path).read()
        for tag, attrs in items:
            text = set_attrs(text, tag, attrs)
        new_texts[path] = text
    for path, text in new_texts.items():
        shutil.copy(path, path + ".bak")
        open(path, "w").write(text)
    mujoco.MjModel.from_xml_path(SCENE_XML)   # make sure the saved scene still loads
    status.set("saved (backups: *.xml.bak)")
    print("saved scene to", ASSETS_DIR, flush=True)


# ---------- layout ----------
left = ttk.Frame(root)
left.grid(row=0, column=0, sticky="n")
image_label = ttk.Label(left)
image_label.pack()
status = tk.StringVar(value="")
ttk.Label(left, textvariable=status).pack(anchor="w")

right = ttk.Frame(root)
right.grid(row=0, column=1, sticky="ns", padx=6)
tabs = ttk.Notebook(right)
tabs.pack(fill="both", expand=True)
frames = {}
for tab, name, lo, hi, step, _ in SPECS:
    if tab not in frames:
        frames[tab] = ttk.Frame(tabs)
        tabs.add(frames[tab], text=tab)
    tk.Scale(frames[tab], label=name, variable=vars_[name], from_=lo, to=hi,
             resolution=step, orient="horizontal", length=320).pack(fill="x")

initial = {name: vars_[name].get() for name in vars_}
buttons = ttk.Frame(right)
buttons.pack(fill="x", pady=4)
ttk.Button(buttons, text="Save to scene files", command=save).pack(side="left")
ttk.Button(buttons, text="Reset",
           command=lambda: [vars_[n].set(x) for n, x in initial.items()]).pack(side="left")
ttk.Button(buttons, text="Quit", command=root.destroy).pack(side="right")

# ---------- live loop ----------
cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
renderer = mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH)
blank = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)


def tick():
    ok, frame = cap.read()
    real = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else blank.copy()
    if not ok:
        cv2.putText(real, f"no frame from {args.device}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    try:
        apply_to_sim()
    except tk.TclError:          # a slider is mid-edit
        pass
    renderer.update_scene(data, camera=CAMERA_NAME)
    sim = renderer.render()

    a = v("blend real %") / 100
    blend = cv2.addWeighted(real, a, sim, 1 - a, 0)
    edges = real.copy()
    edges[cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 40, 120) > 0] = (255, 0, 0)
    grid = np.vstack([np.hstack([real, sim]), np.hstack([blend, edges])])
    photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(grid))
    image_label.configure(image=photo)
    image_label.image = photo
    root.after(5, tick)


root.protocol("WM_DELETE_WINDOW", root.destroy)
root.after(0, tick)
try:
    root.mainloop()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    renderer.close()
