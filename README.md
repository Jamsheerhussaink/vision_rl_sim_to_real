# SO-101 Vision Pick & Place (MuJoCo-JAX)

The SO-101 arm picks a 5 cm cube and drops it into a cardboard box on a green table.
It sees the scene through a simulated RealSense D455 camera (RGB only, 424x240 @ 30 fps, squashed to 224x224 for the policy). Fresh start: only the scene exists so far.

## Setup

```bash
conda activate vision_rl
pip install "jax[cuda12]"
pip install -e .            # makes `vision_rl` importable everywhere
```

## Try it

```bash
python scripts/view_scene.py                                   # open the scene
python scripts/view_scene.py --save images/camera_test.png     # save a 424x240 camera picture + its 224x224 version
pytest tests                                                   # scene loads in MuJoCo + MJX
python scripts/check_reward_hacking.py                         # audit the pick reward for hacks (ours vs Madhan's)
python scripts/check_task_feasibility.py                       # can the real robot/camera satisfy the reward? (~2 min)
```

## Folders

```
vision_rl/                 the Python package
  envs/                    environments (task logic goes here)
    assets/
      scene.xml            puts everything together: lights, cameras, start pose
      robot.xml            SO-101 robot (Madhan's menagerie version)
      table.xml            green table (ours)
      cube.xml             5 cm cube (ours)
      box.xml              cardboard box (ours)
      meshes/              robot 3D parts (.stl)
  rewards/                 reward functions, pure math (no physics)
    pick.py                staged top-down pick reward + hack fixes
    audit/                 toy world, honest + exploit strategies, checks
  models/                  neural networks: CNN + actor-critic   (to do)
  ppo/                     PPO + GAE                              (to do)
  rendering/               batch camera rendering on the GPU      (to do)
scripts/                   things you run: view, train, eval
tests/                     quick checks
images/                    real photos and sim screenshots
docs/                      notes
```

## Scene facts

- Table top at z = 0.40 m. Robot base at (0.045, 0, 0.40).
- Cube starts at (0.274, -0.128, 0.425). Box center at (0.326, 0.133, 0.40), turned -20°.
- Home pose: all arm joints 0 (arm points straight up), gripper closed.
- Physics: timestep 5 ms (Madhan's MJX settings, from robot.xml).
- Main camera: `realsense_d455_rgb`, 424x240 @ 30 fps, squashed to 224x224 (`vision_rl/camera.py`). Also `front_cam` and `top_cam`.
