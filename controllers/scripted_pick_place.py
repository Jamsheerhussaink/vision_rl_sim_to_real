import numpy as np
import mujoco
from controllers.ik_solver import SO101IKSolver


class ScriptedPickAndPlace:
    """
    Finite State Machine (FSM) expert trajectory generator for Pick-and-Place:
    1. HOVER_CUBE: Move above the yellow cube (z + 0.08m) with gripper OPEN.
    2. DESCEND: Lower gripper directly over cube center.
    3. GRASP: Close gripper tightly on the cube.
    4. LIFT: Lift the cube straight up (z + 0.10m).
    5. HOVER_BOX: Transport cube horizontally above the cardboard box.
    6. RELEASE: Open gripper to drop cube cleanly into the box.
    """

    STATES = ["HOVER_CUBE", "DESCEND", "GRASP", "LIFT", "HOVER_BOX", "RELEASE", "DONE"]

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.ik = SO101IKSolver(model)
        self.cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self.reset()

    def reset(self):
        self.state_idx = 0
        self.timer = 0

    @property
    def current_state(self) -> str:
        return self.STATES[self.state_idx]

    def get_action(self, data: mujoco.MjData) -> np.ndarray:
        """
        Computes 6D actuator control [q0, q1, q2, q3, q4, gripper_ctrl].
        """
        cube_pos = data.xpos[self.cube_id].copy()
        box_pos = data.xpos[self.box_id].copy()

        gripper_open = 1.40
        gripper_closed = -0.15

        target_xyz = cube_pos.copy()
        gripper_target = gripper_open

        if self.current_state == "HOVER_CUBE":
            target_xyz[2] += 0.08
            gripper_target = gripper_open
            if self.timer > 50:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "DESCEND":
            target_xyz[2] += 0.015  # Align with cube center
            gripper_target = gripper_open
            if self.timer > 40:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "GRASP":
            target_xyz[2] += 0.015
            gripper_target = gripper_closed
            if self.timer > 50:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "LIFT":
            target_xyz[2] += 0.12
            gripper_target = gripper_closed
            if self.timer > 50:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "HOVER_BOX":
            target_xyz = box_pos.copy()
            target_xyz[2] += 0.12
            gripper_target = gripper_closed
            if self.timer > 60:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "RELEASE":
            target_xyz = box_pos.copy()
            target_xyz[2] += 0.08
            gripper_target = gripper_open
            if self.timer > 40:
                self.state_idx += 1
                self.timer = 0

        elif self.current_state == "DONE":
            target_xyz = box_pos.copy()
            target_xyz[2] += 0.15
            gripper_target = gripper_open

        self.timer += 1

        # Solve IK for arm
        arm_q = self.ik.solve(target_xyz, data)
        action = np.zeros(6, dtype=np.float32)
        action[:5] = arm_q
        action[5] = gripper_target

        return action
