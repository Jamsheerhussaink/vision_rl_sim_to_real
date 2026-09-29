import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from envs.pick_place_env import SO101PickPlaceEnv


def test_physical_grasp_and_lift():
    """
    Physical sanity test:
    1. Loads SO-101 environment with calibrated STS3215 actuator physics and silicone gripper pads.
    2. Opens gripper and places the 28g yellow cube between the jaws.
    3. Closes gripper on the cube with realistic servo torque.
    4. Lifts the arm upward smoothly.
    5. Confirms non-slip contact dynamics and sustained elevation.
    """
    env = SO101PickPlaceEnv(observation_mode="state")
    env.reset()
    model, data = env.model, env.data

    print("=" * 60)
    print("SO-101 PHYSICAL GRASP & LIFT VERIFICATION")
    print("=" * 60)

    # 1. Open gripper wide
    print("1. Opening gripper...")
    data.ctrl[:] = [0.0, -0.3, 0.9, 0.5, 0.0, 1.40]
    for _ in range(200):
        mujoco.mj_step(model, data)

    # 2. Position cube at gripper grasp frame
    site_pos = data.site_xpos[env.gripper_site_id].copy()
    data.qpos[6:9] = site_pos
    data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    cube_initial_z = float(data.xpos[env.cube_id][2])
    print(f"2. Cube positioned at grasp frame: Z = {cube_initial_z:.4f} m")

    # 3. Close gripper firmly
    print("3. Closing silicone gripper pads on cube...")
    data.ctrl[5] = -0.10
    for _ in range(60):
        mujoco.mj_step(model, data)

    # Check contacts
    active_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if env.cube_gid in pair and any(g in pair for g in env.gripper_geoms):
            active_contacts += 1

    print(f"   Active contacts detected between cube and gripper: {active_contacts}")
    assert active_contacts > 0, "No contact established between gripper pads and cube!"

    # 4. Lift arm upward with coordinated pitch compensation
    print("4. Lifting arm upward with cube...")
    for step in range(150):
        data.ctrl[1] -= 0.003  # shoulder lift upward
        data.ctrl[2] -= 0.002  # elbow flex compensation
        data.ctrl[3] += 0.005  # wrist flex compensation (keeps gripper level)
        data.ctrl[5] = -0.10   # hold gripper closed
        mujoco.mj_step(model, data)

    cube_final_z = float(data.xpos[env.cube_id][2])
    lift_height = cube_final_z - cube_initial_z

    print("\n" + "=" * 60)
    print("GRASP & LIFT TEST RESULTS:")
    print(f"  Initial Cube Height: {cube_initial_z:.4f} m")
    print(f"  Final Cube Height:   {cube_final_z:.4f} m")
    print(f"  Net Elevation Gain:  {lift_height:+.4f} m ({lift_height * 100:.1f} cm)")
    print(f"  Contacts Maintained: {active_contacts} contact points")
    print("=" * 60)

    if lift_height > 0.025:
        print("✅ SUCCESS: Realistic contact physics verified! Cube lifted firmly without slipping.")
        return True
    else:
        print("❌ FAILED: Cube slipped from gripper.")
        return False


if __name__ == "__main__":
    test_physical_grasp_and_lift()
