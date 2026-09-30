import numpy as np
import mujoco


class RealSenseD415RGB:
    """
    Intel RealSense D415 RGB Camera Streamer:
    Simulates the calibrated RealSense D415 RGB sensor matching 'env final.png'
    Resolution: 1366x768 (16:9 HD), FOV: 42.5 deg vertical.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "realsense_d415_rgb",
        width: int = 1366,
        height: int = 768,
    ):
        self.model = model
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """Captures RGB image matching the physical RealSense D415 camera."""
        self.renderer.update_scene(data, camera=self.camera_name)
        return self.renderer.render()
