import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from PIL import Image
from vision.realsense_rgb import RealSenseD415RGB


def test_realsense_camera():
    model = mujoco.MjModel.from_xml_path("models/mujoco/scene.xml")
    model.vis.global_.offwidth = 1366
    model.vis.global_.offheight = 768
    data = mujoco.MjData(model)

    # Calibrated resting arm pose matching env final.png
    data.qpos[:6] = [-0.18, 1.12, 0.75, 0.55, 0.0, 0.0]
    data.ctrl[:6] = [-0.18, 1.12, 0.75, 0.55, 0.0, 0.0]

    # Position 5cm yellow cube
    j_cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    qadr = model.jnt_qposadr[j_cube]
    data.qpos[qadr:qadr+3] = [0.16, -0.01, 0.425]

    for _ in range(50):
        mujoco.mj_step(model, data)

    cam = RealSenseD415RGB(
        model,
        camera_name="realsense_d415_rgb",
        width=1366,
        height=768,
    )

    print("--- Capturing Intel RealSense D415 RGB Frame ---")
    rgb = cam.capture_rgb(data)
    print(f"RGB Frame Shape: {rgb.shape} (dtype: {rgb.dtype})")

    out_file = "tests/realsense_d415_snapshot.png"
    Image.fromarray(rgb).save(out_file)
    print(f"✅ Saved RealSense D415 RGB frame to: {out_file}")


if __name__ == "__main__":
    test_realsense_camera()
