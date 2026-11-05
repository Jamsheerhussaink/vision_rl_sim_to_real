"""Open the scene in the MuJoCo viewer, or save a camera picture with --save.

    python scripts/view.py
    python scripts/view.py --save images/current_env.png

--save writes the 424x240 camera frame and its 224x224 preview
(`<name>_224.png`). Training resolution is saved in each run's config.json.
"""
import argparse
import os

import mujoco
import mujoco.viewer
import PIL.Image

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAME, to_policy_image
from vision_rl.envs import SCENE_XML

parser = argparse.ArgumentParser()
parser.add_argument("--camera", default=CAMERA_NAME)
parser.add_argument("--save", help=f"save one {CAM_WIDTH}x{CAM_HEIGHT} camera "
                    "image here instead of opening the viewer")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

if args.save:
    with mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH) as renderer:
        renderer.update_scene(data, camera=args.camera)
        rgb = renderer.render()
    root, ext = os.path.splitext(args.save)
    PIL.Image.fromarray(rgb).save(args.save)
    PIL.Image.fromarray(to_policy_image(rgb)).save(f"{root}_224{ext}")
    print("saved", args.save, "and", f"{root}_224{ext}")
else:
    mujoco.viewer.launch(model, data)
