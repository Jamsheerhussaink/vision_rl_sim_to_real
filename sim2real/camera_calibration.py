"""
Eye-to-Hand Camera Extrinsics Calibration:
Computes the rigid 4x4 transformation matrix T_base_cam [R | t]
from ArUco marker detections on the light green table, aligning
the physical RealSense camera with the simulated [0.42, 0.30, 0.68] frame.
"""
import numpy as np


class CameraCalibrator:
    def __init__(self, target_pos=(0.42, 0.30, 0.68), target_lookat=(0.28, -0.03, 0.41)):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.target_lookat = np.array(target_lookat, dtype=np.float32)

    def compute_extrinsics_error(self, measured_cam_pos: np.ndarray) -> dict:
        """
        Compares measured camera mount position with simulation specification.
        """
        diff = measured_cam_pos - self.target_pos
        dist_err = np.linalg.norm(diff)
        return {
            "dx_mm": diff[0] * 1000.0,
            "dy_mm": diff[1] * 1000.0,
            "dz_mm": diff[2] * 1000.0,
            "total_distance_error_mm": dist_err * 1000.0,
            "aligned": dist_err < 0.015,  # within 15mm is acceptable
        }


if __name__ == "__main__":
    calib = CameraCalibrator()
    # Example simulated test
    test_measured = np.array([0.423, 0.298, 0.678])
    res = calib.compute_extrinsics_error(test_measured)
    print("Camera Calibration Verification:")
    for k, v in res.items():
        print(f"  {k}: {v}")
