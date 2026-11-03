"""The scene loads in MuJoCo and MJX, and the cube rests on the table."""
import mujoco
import numpy as np
from mujoco import mjx

from vision_rl.envs import SCENE_XML


def load():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data


def test_names_exist():
    model, _ = load()
    for kind, name in [(mujoco.mjtObj.mjOBJ_BODY, "cube"), (mujoco.mjtObj.mjOBJ_BODY, "target_box"),
                       (mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
                       (mujoco.mjtObj.mjOBJ_CAMERA, "realsense_d455_rgb")]:
        assert mujoco.mj_name2id(model, kind, name) >= 0, name
    assert model.nu == 6


def test_cube_rests_on_table():
    model, data = load()
    for _ in range(400):  # 2 seconds
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert abs(data.body("cube").xpos[2] - 0.425) < 0.003


def test_mjx_steps():
    model, data = load()
    mx, dx = mjx.put_model(model), mjx.put_data(model, data)
    for _ in range(10):
        dx = mjx.step(mx, dx)
    assert np.isfinite(np.asarray(dx.qpos)).all()
