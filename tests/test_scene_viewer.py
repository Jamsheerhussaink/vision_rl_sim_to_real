"""
Interactive 3D MuJoCo Scene Viewer:
Visualizes the SO-101 pick-and-place scene:
- Light green tabletop with matte black legs
- 3.5cm bright yellow cube
- Small cardboard target box with 5 collision walls
- Room lighting (overhead key luminaire + angled fill light)
- Calibrated 3D workspace depth camera frustum
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer


def launch_viewer():
    xml_path = "models/mujoco/scene.xml"
    print(f"Loading scene from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set default home pose
    data.ctrl[:] = [0.0, 0.0, 1.20, -1.20, 0.0, 1.20]

    print("\n" + "=" * 60)
    print("Launching MuJoCo Interactive Viewer")
    print("Controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Translate / Pan")
    print("  - Scroll wheel: Zoom in / out")
    print("  - Press Space: Pause / Unpause physics")
    print("  - Press Tab: Cycle cameras (inspect 'workspace_depth_cam')")
    print("=" * 60)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                # 50 Hz refresh
                time_until_next_step = model.opt.timestep * 10 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"Interactive GUI viewer exited or not supported in headless mode: {e}")


if __name__ == "__main__":
    launch_viewer()
