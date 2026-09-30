import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from vision.realsense_rgb import RealSenseD415RGB


class SO101PickPlaceEnv(gym.Env):
    """
    Gymnasium Environment for SO-101 Pick-and-Place:
    - Task: Pick up 5cm yellow cube from dark green table and place into 15x20cm cardboard box.
    - Physics: Realistic STS3215 actuator modeling, contact solver, high-friction silicone pads.
    - Vision: Intel RealSense D415 RGB camera matching 'env final.png'.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = None,
        render_mode: str = None,
        max_steps: int = 250,
        observation_mode: str = "state",  # "state" or "rgb"
        camera_name: str = "realsense_d415_rgb",
        img_size: tuple = (128, 128),
    ):
        super().__init__()

        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "models", "mujoco", "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.img_size = img_size
        self.camera_name = camera_name
        self.step_count = 0

        # Object & body IDs
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.table_top_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Gripper collision geoms
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self.gripper_geoms = set()
        for bid in [self.gripper_body_id, self.jaw_body_id]:
            if bid >= 0:
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == bid:
                        self.gripper_geoms.add(gid)

        # RealSense D415 RGB Camera
        self.rgb_sensor = RealSenseD415RGB(
            self.model,
            camera_name=camera_name,
            width=img_size[1],
            height=img_size[0],
        )

        # Action space: 6 continuous normalized delta controls in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation space
        if self.observation_mode in ["rgb", "vision"]:
            self.observation_space = spaces.Dict(
                {
                    "rgb": spaces.Box(low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8),
                    "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                }
            )
        else:
            # 6 qpos + 6 qvel + 3 cube_pos + 3 box_pos + 3 gripper_pos + 3 rel_cube + 3 rel_box = 27
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        # Home configuration matching real-world resting pose in env final.png
        self.home_qpos = np.array([-0.18, 1.12, 0.75, 0.55, 0.0, 0.0], dtype=np.float32)
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.max_joint_delta = 0.06  # Smooth rad per step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset robot to home pose
        self.data.qpos[:6] = self.home_qpos
        self.data.ctrl[:6] = self.home_qpos

        # Domain randomization: Randomize yellow cube position slightly on the table
        cube_x = np.random.uniform(0.14, 0.18)
        cube_y = np.random.uniform(-0.03, 0.02)
        cube_z = 0.425  # 5cm cube resting on z=0.40m tabletop
        self.data.qpos[6:9] = [cube_x, cube_y, cube_z]
        self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]  # identity quat

        # Settle physics
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.initial_cube_z = float(self.data.xpos[self.cube_id][2])

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Apply continuous action as scaled delta position
        action = np.clip(action, -1.0, 1.0)
        target_ctrl = self.data.ctrl[:6] + action * self.max_joint_delta
        self.data.ctrl[:6] = np.clip(target_ctrl, self.ctrl_min, self.ctrl_max)

        # Sub-step MuJoCo physics (10 steps of 0.002s = 0.02s per env step, 50 Hz)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(action, info)

        # Termination conditions
        terminated = info["is_success"]
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos[:6].astype(np.float32)
        qvel = self.data.qvel[:6].astype(np.float32)
        cube_pos = self.data.xpos[self.cube_id].astype(np.float32)
        box_pos = self.data.xpos[self.box_id].astype(np.float32)
        gripper_pos = self.data.site_xpos[self.gripper_site_id].astype(np.float32)

        rel_cube = cube_pos - gripper_pos
        rel_box = box_pos - cube_pos

        if self.observation_mode in ["rgb", "vision"]:
            rgb = self.rgb_sensor.capture_rgb(self.data)
            proprio = np.concatenate([qpos, qvel])
            return {"rgb": rgb, "proprio": proprio}
        else:
            return np.concatenate([qpos, qvel, cube_pos, box_pos, gripper_pos, rel_cube, rel_box])

    def _get_info(self):
        cube_pos = self.data.xpos[self.cube_id]
        box_pos = self.data.xpos[self.box_id]
        gripper_pos = self.data.site_xpos[self.gripper_site_id]

        dist_ee_cube = float(np.linalg.norm(gripper_pos - cube_pos))
        dist_cube_box = float(np.linalg.norm(cube_pos[:2] - box_pos[:2]))
        cube_lift = float(cube_pos[2] - self.initial_cube_z)

        # Check contact between gripper and cube
        cube_grasped = False
        table_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.cube_gid in pair and any(g in pair for g in self.gripper_geoms):
                cube_grasped = True
            if self.table_top_gid in pair and any(g in pair for g in self.gripper_geoms):
                table_collision = True

        is_lifted = cube_lift > 0.04
        is_in_box = (dist_cube_box < 0.05) and (cube_pos[2] < 0.44) and (cube_pos[2] > 0.39)
        is_success = is_in_box and not cube_grasped  # Dropped cleanly into box

        return {
            "dist_ee_cube": dist_ee_cube,
            "dist_cube_box": dist_cube_box,
            "cube_lift": cube_lift,
            "is_grasped": cube_grasped,
            "is_lifted": is_lifted,
            "is_in_box": is_in_box,
            "is_success": is_success,
            "table_collision": table_collision,
        }

    def _compute_reward(self, action: np.ndarray, info: dict) -> float:
        # 1. Reaching stage
        r_reach = -info["dist_ee_cube"]

        # 2. Grasp & lift stage
        r_grasp = 1.0 if info["is_grasped"] else 0.0
        r_lift = np.clip(info["cube_lift"] * 15.0, 0.0, 3.0)

        # 3. Transport stage (only if lifted)
        r_transport = 0.0
        if info["is_lifted"]:
            r_transport = 2.0 - 2.0 * np.clip(info["dist_cube_box"] / 0.30, 0.0, 1.0)

        # 4. Placement success bonus
        r_success = 10.0 if info["is_success"] else 0.0

        # Action penalty & table collision penalty
        ctrl_penalty = 0.01 * np.sum(np.square(action))
        collision_penalty = 0.5 if info["table_collision"] else 0.0

        return float(r_reach + r_grasp + r_lift + r_transport + r_success - ctrl_penalty - collision_penalty)
