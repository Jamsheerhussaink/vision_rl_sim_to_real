# Plan: lift 10–12 cm, hold 0.5 s, carry over the box wall

Status: **SUPERSEDED (2026-10-26).** The user chose no height stages: the model learns the final 10-12 cm / 0.5 s goal directly (see `docs/reward_only_roadmap.md`, Round 3). The height measurements below are still correct.

## Why

Goal: the robot lifts the cube **10–12 cm**, holds it still for **0.5 s**, then carries it to the box **high enough to clear the box wall**.

Today:
- "Success" = the cube about **3.5 cm** up (0.46 m) for **0.25 s**
- Lift points stop growing at about **5.5 cm** (0.48 m)
- Carry points (`place_cube`) start once the cube is only **1.5 cm** up (0.44 m)

## Key heights (measured from the project files)

| Thing | Height (world z) | Notes |
|---|---|---|
| Cube center resting on the table | 0.426 m | `cube.init_state.pos` |
| Cube half size | 0.025 m | 5 cm cube |
| Box wall top | ≈ 0.491 m | box at z 0.401 + wall top 0.090 (`usd/box.usd`) |
| Cube center needed to clear the wall (+1 cm margin) | ≈ 0.526 m | = **10 cm** above rest |
| Your goal: 10–12 cm | 0.526–0.546 m | |

**10 cm is exactly the height needed to carry the cube over the box wall.** Your goal and the box agree.

## The idea: one "lift height" setting, raised step by step

Right now the lift height is written in **4 different places**. Changing them one by one is error-prone. If they disagree, the robot gets mixed messages: points for 5.5 cm, but success needs 3.5 cm.

So first, make **one setting** that controls them all, the same way the project already uses `SO101_*` environment variables in `utils/pipeline.py`:

- `SO101_LIFT_HEIGHT_CM`: how high counts as "lifted" (default 3.4, the same as today)
- `SO101_LIFT_HOLD_S`: how long it must be held (default 0.25, the same as today)

With the defaults, nothing changes. Then you raise the numbers stage by stage, without editing code.

## Code changes (when you say go)

### 1. One setting drives all lift heights
File: `src/isaac_vision_rl/utils/pipeline.py` → `configure_curriculum`

Read the two variables, then set:
- `rewards.lift_completed.params["minimal_height"]` = 0.426 + height, and `["hold_time_s"]` = hold
- `terminations.lift_success.params` = the same two values. Success points and the "try ends on success" check must agree.
- `rewards.lift_cube.params["target_height"]` = 0.426 + height + 0.02. Lift points keep growing until just **above** the success height, so the robot has a reason to go all the way up.

### 2. The "move up" bonus follows the target
File: `rewards.py` → `lift_cube_upward`

The cutoff `z_effective < 0.465` is fixed. Change it to `z_effective < target_height - 0.015`, so the move-up bonus doesn't switch off at 4 cm when the goal is 11 cm.

### 3. Placing remembers the new lift rule
File: `rewards.py` → `PlaceSuccess`

It builds its own `LiftSuccess` with the **old** defaults (3.5 cm, 0.25 s). Pass it the same height and hold settings, so "was it lifted first?" uses the new rule.

### 4. Carry high enough to clear the wall
File: `rewards.py` → `object_goal_distance` (`place_cube`)

- Today: carry points switch fully on at 0.44 m (on/off).
- New: a **smooth ramp**. 0 points below 0.49 m (wall top), rising to full points at 0.526 m (10 cm). The robot "feels warmer" as it carries higher (lesson 4: smooth checks teach better than on/off ones).
- The settings go in `place_cube` params: `carry_low=0.49`, `carry_full=0.526`.

### 5. Tests (`tests/test_picklift_logic.py`)
- Default settings give exactly today's values (nothing changes by accident)
- `SO101_LIFT_HEIGHT_CM=11` sets all 4 places to 0.536 m (and the lift target to 0.556 m)
- `PlaceSuccess` doesn't count a 3.5 cm lift as "lifted" when the setting is 11 cm
- Carry points: 0 at 0.48 m, about half at 0.508 m, full at 0.53 m

## Checks before training

**A. Can the camera see the hand at 12 cm?** In manual inspection mode (`utils/manual_inspection.py`), move the hand to about 0.57 m with the cube held and check that `gripper_outside_camera_view` stays false. If the hand leaves the picture, the robot can never succeed at that height. Then we cap the goal lower or adjust the camera margin first.

**B. Is 6 seconds enough?** Lifting 11 cm slowly and holding 0.5 s takes longer. If tries often run out of time (`time_out` high) while lifting looks right, raise `episode_length_s` (lift stage) from 6 to 8 s.

## Training stages (step by step)

Start each stage from the **last good checkpoint** of the stage before. Move to the next stage when `lift_success` is **above 60%** for about 50 training steps (check with `tb_summary.py`).

| Stage | `SO101_LIFT_HEIGHT_CM` | `SO101_LIFT_HOLD_S` | Success height | Lift points grow up to |
|---|---|---|---|---|
| A | 5 | 0.3 | 0.476 m | 0.496 m |
| B | 8 | 0.4 | 0.506 m | 0.526 m |
| C (goal) | 10 | 0.5 | 0.526 m | 0.546 m (12 cm) |

Stage C: success counts from 10 cm, and the lift points aim at 12 cm. So the robot settles in your 10–12 cm range.

Example command (stage A):

```bash
SO101_LIFT_HEIGHT_CM=5 SO101_LIFT_HOLD_S=0.3 conda run -n env_isaaclab python scripts/train.py --checkpoint <last good model.pt> --envs 16 --iterations 2000 --task lift --curriculum fixed --policy rgb --wandb --wandb-project so101-pick-lift-v6 --viz newton
```

## What to watch at each stage

| Number | Good sign | If bad |
|---|---|---|
| `lift_success` | Climbs past 60% | Stays under 20% for 100+ steps: the stage is too big a jump. Use a half step (e.g. 6.5 cm). |
| `gripper_out_of_view` | Stays under 5% | Rises at higher stages: the camera limit (check A) |
| `bilateral_contact_fraction` | Over 30% | Grip too weak for a 0.5 s hold. Fix grab points first (case 3). |
| `time_out` | Goes down | Stays high: check B (try too short) |

## Not in this plan (separate next steps)

- Making grab points easier to earn (case 3 in the skill)
- Stopping the "push the cube into the box wall" habit (case 6)
- Going home: already done by the script `utils/ik_return_home.py`, not by training
