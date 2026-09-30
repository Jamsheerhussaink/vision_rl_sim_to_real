import numpy as np
import mujoco


class SO101IKSolver:
    """
    Damped Least Squares (DLS) Inverse Kinematics solver for the SO-101 robot arm.
    Controls arm joints (indices 0 to 4) to position the gripperframe site.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "gripperframe",
        damping: float = 0.05,
        step_size: float = 0.5,
        max_iterations: int = 40,
        tolerance: float = 0.005,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Arm joints 0..4 (gripper joint 5 is independent)
        self.num_arm_joints = 5
        self.joint_limits_low = model.jnt_range[:self.num_arm_joints, 0]
        self.joint_limits_high = model.jnt_range[:self.num_arm_joints, 1]

    def solve(self, target_pos: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """
        Solves for target joint positions [q0, q1, q2, q3, q4] to reach target_pos [X, Y, Z].
        """
        q = data.qpos[:self.num_arm_joints].copy()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(self.max_iterations):
            # Evaluate forward kinematics at current q
            data.qpos[:self.num_arm_joints] = q
            mujoco.mj_forward(self.model, data)

            current_pos = data.site_xpos[self.site_id]
            error = target_pos - current_pos

            if np.linalg.norm(error) < self.tolerance:
                break

            # Compute Jacobian at site
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
            J = jacp[:, :self.num_arm_joints]  # 3 x 5

            # Damped Least Squares: dq = J.T @ (J @ J.T + lambda^2 * I)^-1 @ dx
            JJT = J @ J.T + (self.damping ** 2) * np.eye(3)
            inv_JJT = np.linalg.pinv(JJT)
            dq = J.T @ (inv_JJT @ error)

            q += self.step_size * dq
            q = np.clip(q, self.joint_limits_low, self.joint_limits_high)

        return q
