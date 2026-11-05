from pathlib import Path

import mujoco
import numpy as np

from vision_rl.envs import ASSETS_DIR, SCENE_XML
from vision_rl.envs.pick_place_env import placement_geometry, PlaceConfig
from vision_rl.train import TrainConfig


def geometry(cube=(.331, .008, .429), speed=(0, 0, 0), grasp=(.331, .008, .53), jaw=1.2):
    return placement_geometry(np.array(cube), np.eye(3), np.array([.331, .008, .4]),
                              np.eye(3), np.array(speed), np.array(grasp), jaw)


def test_full_cube_must_fit_and_rest_before_release_counts():
    inside, resting, withdrawn, _, _ = geometry()
    assert inside and resting and withdrawn
    assert not geometry(cube=(.410, .008, .429))[0]  # centre inside, cube intersects wall
    assert not geometry(cube=(.331, .008, .55))[1]   # held above box
    assert not geometry(speed=(.1, 0, 0))[1]        # moving / bouncing
    assert not geometry(jaw=-.17)[2]               # still closed
    assert not geometry(grasp=(.331, .008, .45))[2] # hand not withdrawn


def test_real_collision_box_catches_cube_and_original_scene_stays_without_box():
    m = mujoco.MjModel.from_xml_path(str(Path(ASSETS_DIR) / "place_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    b = m.body("target_box").id
    a = m.body_mocapid[b]
    d.qpos[6:9] = d.mocap_pos[a] + [0, 0, .14]
    d.qpos[9:13] = [1, 0, 0, 0]
    for _ in range(400):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[8] - (.4 + .004 + .025)) < .003
    cube = m.geom("cube_geom").id
    assert any(cube in (c.geom1, c.geom2) and m.geom("box_bottom").id in (c.geom1, c.geom2)
               for c in d.contact[:d.ncon])
    assert m.nmocap == 2
    original = mujoco.MjModel.from_xml_path(SCENE_XML)
    assert mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_GEOM, "box_bottom") == -1


def test_camera_is_identical_and_place_config_roundtrips():
    original = mujoco.MjModel.from_xml_path(SCENE_XML)
    placed = mujoco.MjModel.from_xml_path(str(Path(ASSETS_DIR) / "place_scene.xml"))
    np.testing.assert_array_equal(original.cam_pos, placed.cam_pos)
    np.testing.assert_array_equal(original.cam_quat, placed.cam_quat)
    cfg = TrainConfig(task="pick_place", place=PlaceConfig(box_range=.01, light_range=.08))
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
