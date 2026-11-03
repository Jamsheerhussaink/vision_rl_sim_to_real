"""Live sync: the sim camera follows the real D455 while you move it.

An ArUco marker lying flat on the table at a known spot gives the real
camera's full pose every frame; the sim camera is set to that pose.

    python scripts/live_sync.py --make-marker      # writes images/aruco_marker.png
    python scripts/live_sync.py --marker-size 0.15 --marker-dist 0.25

Placing the marker: flat on the table, centre `--marker-dist` metres straight
in front of the robot's base rotation axis (the direction the arm reaches), with
the word TOP pointing away from the robot. Measure the black square's edge
after printing (or on your phone screen) and pass it as --marker-size.

Window (each panel at the exact 424x240 camera resolution):
    real (marker axes drawn) | sim
    50/50 blend              | real with sim edges in red

Keys (click the window first): p prints the <camera> line for scene.xml,
q/ESC quits. Closing the window or Ctrl+C in the terminal also quits.
"""
import argparse

import cv2
import mujoco
import numpy as np
import PIL.Image

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME
from vision_rl.envs import SCENE_XML

# D455 factory colour intrinsics at 424x240 (read with pyrealsense2).
K = np.array([[213.276, 0.0, 214.191], [0.0, 212.753, 122.678], [0.0, 0.0, 1.0]])
DIST = np.array([-0.0563155, 0.0649003, 0.000712136, -0.000649843, -0.0208018])
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_ID = 0
TABLE_Z = 0.40

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/video4", help="V4L2 colour node")
parser.add_argument("--camera", default=CAMERA_NAME)
parser.add_argument("--marker-size", type=float, default=0.15,
                    help="black square edge in metres, as measured")
parser.add_argument("--marker-dist", type=float, default=0.25,
                    help="metres from the base rotation axis to the marker centre")
parser.add_argument("--smooth", type=float, default=0.3,
                    help="0..1, weight of the newest pose (1 = no smoothing)")
parser.add_argument("--make-marker", action="store_true",
                    help="write images/aruco_marker.png (print at 100%% scale) and exit")
args = parser.parse_args()

if args.make_marker:
    dpi, mm = 300, args.marker_size * 1000
    side = int(round(mm / 25.4 * dpi))
    marker = cv2.aruco.generateImageMarker(ARUCO, MARKER_ID, side)
    pad = side // 4
    page = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    page[pad:pad + side, pad:pad + side] = marker
    cv2.putText(page, "TOP (away from robot)", (pad, pad - pad // 4),
                cv2.FONT_HERSHEY_SIMPLEX, side / 900, 0, max(1, side // 300))
    PIL.Image.fromarray(page).save("images/aruco_marker.png", dpi=(dpi, dpi))
    print(f"saved images/aruco_marker.png: {mm:.0f} mm marker at {dpi} dpi; "
          "print at 100% / actual size, then measure it")
    raise SystemExit

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
cam = model.camera(args.camera)

# Marker frame in the world: centre on the table in front of the pan axis;
# marker x -> world -y, marker y (TOP) -> world +x, marker z -> world +z.
pan_xy = data.xpos[model.body("shoulder").id][:2]
MARKER_POS = np.array([pan_xy[0] + args.marker_dist, pan_xy[1], TABLE_Z])
R_WORLD_MARKER = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
h = args.marker_size / 2
CORNERS = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

# Current sim camera pose (world position, MuJoCo axes as matrix columns).
cam_pos = data.cam_xpos[cam.id].copy()
cam_mat = data.cam_xmat[cam.id].reshape(3, 3).copy()


def pose_from_marker(rvec, tvec):
    """World position and MuJoCo camera axes from the marker's pose in the camera."""
    r_cm, _ = cv2.Rodrigues(rvec)                      # marker -> OpenCV camera
    r_wc = R_WORLD_MARKER @ r_cm.T                     # OpenCV camera -> world
    pos = MARKER_POS + R_WORLD_MARKER @ (-r_cm.T @ tvec.ravel())
    return pos, r_wc @ np.diag([1.0, -1.0, -1.0])      # MuJoCo: y up, looks along -z


def orthonormal(m):
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def xml_line():
    xy = " ".join(f"{v:.7f}" for v in (*cam_mat[:, 0], *cam_mat[:, 1]))
    fovy = model.cam_fovy[cam.id]
    return (f'<camera name="{args.camera}" pos="{cam_pos[0]:.3f} {cam_pos[1]:.3f} '
            f'{cam_pos[2]:.3f}"\n        xyaxes="{xy}" fovy="{fovy:.2f}"/>')


cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise SystemExit(f"cannot open {args.device}")

detector = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
win = "live sync: real vs sim (424x240)"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
try:
    with mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH) as renderer:
        while True:
            ok, real = cap.read()
            if not ok:
                print("no frame from", args.device)
                continue

            corners, ids, _ = detector.detectMarkers(real)
            tracked = False
            if ids is not None and MARKER_ID in ids.ravel():
                c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2)
                ok, rvec, tvec = cv2.solvePnP(CORNERS, c, K, DIST,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    tracked = True
                    pos, mat = pose_from_marker(rvec, tvec)
                    a = args.smooth
                    cam_pos = (1 - a) * cam_pos + a * pos
                    cam_mat = orthonormal((1 - a) * cam_mat + a * mat)
                    cv2.drawFrameAxes(real, K, DIST, rvec, tvec, args.marker_size / 2)

            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, cam_mat.flatten())
            model.cam_pos[cam.id] = cam_pos          # worldbody camera: local == world
            model.cam_quat[cam.id] = quat
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=args.camera)
            sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            edges = real.copy()
            edges[cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 40, 120) > 0] = (0, 0, 255)
            grid = np.vstack([np.hstack([real, sim]), np.hstack([blend, edges])])
            above = (cam_pos[2] - TABLE_Z) * 100
            hud = (f"{'TRACKING' if tracked else 'marker not seen - holding'}  "
                   f"pos {cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f} "
                   f"({above:.1f} cm above table)")
            cv2.putText(grid, hud, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if tracked else (0, 0, 255), 1)
            cv2.imshow(win, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("p"):
                print(xml_line(), flush=True)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
print("final camera:\n" + xml_line())
