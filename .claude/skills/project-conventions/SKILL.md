---
name: project-conventions
description: SO-101 vision RL design decisions (observations, camera, frames, rewards). Consult before changing obs, camera, reward or env code.
user-invocable: false
---
Settled decisions (do not revisit without the user asking):
- Actor input: RGB camera + 6 joint angles ONLY. No qvel, gripper xyz, cube position, or depth.
- Stack 3 frames.
- Camera: RealSense D455, FIXED (never move it). Robot base is cut off at the left edge of the real view; ~12 cm above table; table 75 cm wide.
- 224x224 obs by RESIZE (squash) of the full 16:9 frame, not a crop. Sim and deploy must match.
- Visibility penalty applies to moving links only (base is out of frame). Spawn cube/box only in the visible area.
- Domain randomisation is wanted.
- Real D455 is on USB 2.1: use V4L2 /dev/video4 at 424x240, not pyrealsense2. Alignment tool: scripts/setup.py.
Stack: JAX/Flax/Optax PPO on MuJoCo-MJX. Keep env/reward code jittable (no Python control flow on traced values).
