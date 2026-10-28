# Reward-only roadmap: full pick and place by reward tuning

Status: **Round 1 code is done** (2026-10-26): `grasp_cube` no longer multiplies the duplicate `downward` check, and its height-gate floor is 0.60 (`compute_grasp_height_gate(floor=...)`, default still 0.20 for `pinch_force_bonus`). Tests: 36/36 in `tests/test_picklift_logic.py`; full suite has the same 58 old failures. Rounds 2–4 are still plans. Chosen by the user: learn reward tuning properly; more projects are coming.

Note: Round 1 and the Round 3 final-height change are applied together in the first run (3 changes at once, because the user asked for no staging). That makes it harder to tell which change helped; compare `grasp_cube` and `bilateral_contact_fraction` for the grab fix, `lift_cube` and `lift_success` for the height change.

Method for every round (skill `reward-tuning-tutor`, lesson 9): change 1–2 things → restart from the newest good checkpoint → train about 50 steps → read the numbers with `tb_summary.py --compare` → keep or undo.

## What the code shows about the grab reward (Round 1 diagnosis)

**Finding 1: `grasp_cube` only pays for a NEW BEST score in a try** (`GraspCubeProgress`).
- Total paid per try ≈ weight (6) × the best score reached that try.
- Measured: 0.038 per second (the chart number is points per second; × 6 s = 0.23 points per try) → the best score reached is only about **3.8% of the maximum**.
- So the robot almost never reaches even a medium grab score, and gets no hint about grabbing.
- Good side: it can't be farmed by sitting still, because only improvements pay.

**Finding 2: the "hand points down" check is counted twice inside `grasp_cube`.**
- `downward` = `exp(-10 × (1 - cos))` and `wrist_gate` both measure the same thing: the hand angle.

| Hand tilt | `downward` | `wrist_gate` | Both multiplied |
|---|---|---|---|
| 10° | 0.86 | 1.00 | 0.86 |
| 20° | 0.55 | 0.85 | 0.47 |
| 30° | 0.26 | 0.33 | 0.09 |
| 45° | 0.05 | 0.00 | 0.00 |

- A 20° tilt, which is a decent grab, already loses more than half the points. 30° pays almost nothing.

**Finding 3: the grab happens on the table, but the height gate pays only 20% there** (`compute_grasp_height_gate` floor 0.20). The robot must grab on the table *before* it can lift, so the best moment to give hints is where the pay is smallest.

## Round 1 (2 changes): make the grab learnable

1. **Remove the duplicate hand-angle check** from `grasp_cube`: drop the `downward` factor, keep `wrist_gate` (it has a gentler ramp and is the one that blocks slanted approaches).
2. **Raise the height-gate floor for `grasp_cube` only: 0.20 → 0.60.** Safe because `grasp_cube` pays only for new bests, so sitting on the table can't farm it. Leave `pinch_force` and `contact_grade` alone: they pay every step and *could* be farmed.
   - Do this by giving `compute_grasp_height_gate` a `floor` parameter (default 0.20, so `pinch_force_bonus` stays exactly the same) and passing `floor=0.60` from `grasp_cube`.

Also in this restart (no code needed): your `out_of_view` weight of **-750** is in `env_cfg.py` now. A restart picks it up; the running training does not have it.

**Tests:** the existing height-gate test (`test_rebalanced_grasp_height_gate_and_upward_pull_shaping`) will need new expected ratios (table grab pays 0.6 instead of 0.2, so elevated/table ratio becomes 1/0.6). Add one test that `grasp_cube` pays the same at 10° and 30° tilt differences as `wrist_gate` alone says.

**Restart from:** `logs/rsl_rl/so101_picklift_rgb/2026-10-26_17-26-31/model_1050.pt` (newest, lifts steady at about 35%).

**What to expect after ~50 steps:**

| Number | Now | Hope |
|---|---|---|
| `Episode_Reward/grasp_cube` | 0.04 per try | 3–10× higher |
| `Grasp/bilateral_contact_fraction` | 21% | over 30% |
| `gripper_out_of_view` | 8% | under 5% (the -750 penalty is now live) |
| `lift_success` | 35% | not lower than 35% (if it drops, round 1 changed too much) |

## Round 2 (applied 2026-10-27): stop paying for a tipped cube

Seen in the 19-08 video: the robot tips the cube about 30-40 degrees and holds it low for many seconds. Two changes:
1. Reward terms (grab, squeeze, contact quality, close, lift, `stable_support`) now use a strict upright gate `compute_reward_upright`: 0 at 20 degrees of tilt, 1 at about 8 degrees (`REWARD_UPRIGHT_LOW/HIGH` = 0.94 / 0.99 in `rewards.py`). The success check keeps the loose ramp.
2. `cube_tilt` penalty weight -1 -> -4.
Tests: 40/40 in `tests/test_picklift_logic.py`; full suite has the same 58 old failures.

## Round 2b (still a plan): stop pushing the cube into the box wall (case 6)

- `stable_support` (weight 10) pays both fingers on a still, upright cube even on the table (25% floor). The wall holds the cube still.
- Candidate fix: pay `stable_support` only once the cube is off the table (floor 25% → 0), and/or a small penalty while the cube touches the box in the lift stage.
- Decide with a new video from the Round 1 run.

## Round 3 (changed by the user, 2026-10-26): NO height stages. The model learns the final height directly

Decision: the policy is trained on the final goal from the start: **cube 10 cm up (0.526 m), held 0.5 s**, with the dense lift reward growing smoothly up to **12 cm (0.546 m)** so the policy is pulled upward without hand-made steps.
- Settings live in one place in `env_cfg.py`: `LIFT_SUCCESS_HEIGHT_M`, `LIFT_TARGET_HEIGHT_M`, `LIFT_HOLD_TIME_S`. The success bonus (`lift_completed`) and the success check (`lift_success`) both read them (a test enforces this).
- `lift_cube_upward` / `LiftCubeProgress` take a `target_height` (default 0.48, so older code paths and tests are unchanged).
- Expected side effects (not failures): `lift_success` will fall at first (the old 3.5 cm lift no longer counts), and `Episode_Reward/lift_cube` will drop (the same 3.5 cm lift is now worth less). Watch `bilateral_contact_fraction`, `grasp_cube`, `gripper_out_of_view` and whether `lift_success` climbs again.
- Risks to watch: (1) the camera at 12 cm: the scripted teacher lifted 10-12 cm with camera safety on, so it is likely fine, and `gripper_out_of_view` will tell; (2) 6 s per try may be short: if `time_out` stays high while `lift_success` stays low, raise `episode_length_s` to 8.
- `PlaceSuccess` and the place-stage `lift_cube` still use the old 3.5 cm defaults. Fix them in Round 4.
- `docs/lift_height_plan.md` (the 3-stage plan) is superseded by this decision; its height measurements are still correct.

## Round 4: place stage

Train the place stage (`--task` for pick-and-place) from the best lift checkpoint. `place_cube` carry ramp (wall top 0.49 m → full at 0.526 m) from the height plan. Going home stays a fixed script (`utils/ik_return_home.py`), not a trained skill.

Applied 2026-10-28 (place stage): `place_cube` carry pay now ramps from 0 at the box wall top (`BOX_WALL_TOP_M` 0.49 m) to full at 10 cm lift (0.526 m), so pushing the cube along the table earns nothing; place-stage `lift_cube` target raised from 0.48 m to 0.546 m so the lift pull no longer stops below the wall. `cube_tilt` stays at -1 (a test asserts it). Tests: 45/45 in `tests/test_picklift_logic.py`; full suite 59 old failures.

## Timing reality check

Each round is about 1–2 hours (change, restart, 50+ steps, read numbers). Four rounds will not fit before a 1–2 day deadline with certainty. A safety net that costs no training: the scripted teacher already has recorded full pick-and-place videos. They can be shown as the working fixed-position demo while the learned policy is still improving.
Videos of full successful runs (end reason `placed`) that exist today: `outputs/four_stage/ik_home_validation01.mp4` (with `_return_home.mp4`), `teacher_demo_004.mp4` and `teacher_demo_006.mp4` (both with `_return_home.mp4`). Note: the Oct 23 run with exactly the 10–12 cm / 0.5 s settings has data but no video.
