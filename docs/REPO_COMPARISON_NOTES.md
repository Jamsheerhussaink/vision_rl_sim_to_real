# My repo vs Madhan's repo — study notes

> Where am I on the roadmap?
> MuJoCo → control → reward design → RL → **PPO → Vision RL → SO-101 pick & place** → sim-to-real → real robot.
> Both repos sit at the "Vision RL on SO-101" step.

---

## 1. The two projects in two lines each

**Mine (`jamshi/vision_rl`)**
SO-101 picks a 5 cm cube and drops it into a cardboard box. It uses normal MuJoCo on the CPU, a ready-made PPO from Stable-Baselines3 (SB3), and a camera calibrated to a real D415 photo.

**Madhan's (`madhan/vision_rl-main`)**
SO-101 picks a 3.5 cm cube and moves it onto a flat target pad (or just picks it, in the "pick only" task). It uses MuJoCo-MJX (JAX) so hundreds of robots run together on the GPU, with a hand-written PPO.

**Honest truth first:** neither repo has a policy that actually succeeds yet.
- Mine: `runs/ppo_lift` ran 1.28M steps → success = 0, lift = 0, and it grasped in only 5 of ~1300 episodes.
- Madhan's: the `checkpoints/` folder only has `it1` files (iteration 1). Their own notes show `succ=0.00`.

So "better" below means **better design**, not "already works".

---

## 2. Side-by-side table

| Topic | Mine | Madhan's | Who wins |
|---|---|---|---|
| Physics engine | MuJoCo (CPU), 1 robot per process | MJX (JAX), 256+ robots in one GPU batch | Madhan |
| Speed | **149 steps/sec** (from my log, 1 env) | ~2,100 steps/sec in their notes' example log (not measured on our machine); warp backend claims more | Madhan, by far |
| PPO code | SB3 library (tested, reliable, a "black box") | Hand-written in JAX (you can see every line) | Tie: SB3 is safer, theirs teaches more |
| Arm action | **Delta**: each step moves the target by at most 0.024 rad | **Absolute residual**: target = center + 0.9 × action, slew-limited to 0.08 rad/step | Madhan (easier to explore) |
| Gripper action | **Delta** too: starts closed, needs ~80 steps of "open" in a row to fully open | **Absolute**: action > 0 means open, < 0 means close, right away | Madhan (big one, see §3) |
| Cube start position | Fixed spot ±1 cm | Random inside an 11 × 22 cm area | Madhan |
| Target | Fixed box | Random pad position each episode | Madhan |
| What "grasp" means | **Real contact**: both gripper pads touch the cube | **Proxy**: gripper is near the cube AND the jaw is commanded closed | Mine |
| Success check | Cube lifted 10 cm and held 0.5 s / cube settled inside the box and not moving | Cube within 4 cm of the pad, low, and lifted earlier | Mine (stricter, more physical) |
| Reward style | Potential-based shaping + one-time bonuses | Many staged dense terms (reach, grasp, lift, place, progress) | Mine is cleaner in theory |
| Episode end | Ends on success or failure (cube falls) | Always runs 250 steps, no early end | Mine |
| Time limit handling | SB3 bootstraps the value at timeouts + clock in observation | GAE treats the timeout as "game over" and has no clock in observation | Mine (theirs is a real bug) |
| Domain randomization | Cube mass, friction, light, camera ±3 mm, pixel noise | None (only cube/target position + small joint noise) | Mine |
| Camera | Calibrated to a real D415 photo, 160×90 crop | `overhead_cam` 84×84, roughly matched to their real mount | Mine for realism |
| Network | SB3 NatureCNN + MLP [128,128], features shared | NatureCNN + LayerNorm + MLP [256,256], tanh-squashed mean, std clamped to [0.05, 1.35] | Madhan (std clamp prevents collapse) |
| Checkpoints/resume | Saves policy + VecNormalize; `--init-run` for warm start | One `.msgpack` with weights + optimizer + config; full resume | Madhan |
| Curriculum | lift → place stages, with a "mastery" auto-stop | Separate `so101_pick` and `so101_pick_place` tasks | Mine (place starts from lift) |
| Sim-to-real code | None (I deleted the old `sim2real/`) | `deploy/` bridge, real camera, LeRobot backend, calibration scripts | Madhan |
| Tests | Physics/collision checks, env unit tests | Smoke test + reward unit tests with hand-picked numbers | Madhan's reward tests are a nice idea |
| Expert / demos | Scripted IK pick & place (`controllers/`), **not used in training** | None | Mine has it but wastes it |

---

## 3. Why MY repo is not learning (ranked, with evidence)

1. **The gripper almost never opens.**
   The keyframe starts the gripper closed (−0.17 rad). Fully open is 1.75 rad. Each step can change it by only 0.024 rad.
   So the policy needs about 80 "open" actions in a row. Random exploration is like a drunk walk, so it almost never does that.
   **Evidence:** grasp happened in only 5 of ~1300 episodes, and lift happened 0 times.
   No grasp means no lift reward, so there is nothing to learn from.

2. **Way too slow / too few samples.**
   149 steps/sec with `num_envs = 1`. Vision RL usually needs 10M–50M steps.
   1.28M steps is maybe 3–5% of what's needed. At this speed, 20M steps ≈ 37 hours.

3. **Exploration collapsed.**
   The log shows `std = 0.225`. SB3 starts at std = 1.0, so the noise shrank a lot.
   The log also prints SB3's "Early stopping" message **1,213 times**. SB3 stops an update when KL > 1.5 × `target_kl` (0.045), so most updates were cut short.
   The policy became "confident" before it ever found the cube. Getting confident about nothing, lah.

4. **The cube is almost always in the same place (±1 cm).**
   The policy can memorise one arm motion and ignore the camera. That's fine for a first test, but it won't transfer to a real table where the cube moves.

5. **The scripted expert is unused.** I already have code that can do the full pick & place, but PPO starts from zero.

## 4. Where Madhan's repo is weak (ranked)

1. **Timeout treated as "game over" (value bias).**
   Every episode ends at step 250 by time limit. Their GAE uses `done` to mean "no future value", so the critic thinks step 250 is the end of the world.
   But the observation has **no clock**, so the critic can't know when that is. Value estimates get noisy and wrong near the end.
   Fix: store `truncated` separately and bootstrap with V(next_state), or add `step/250` to proprio.

2. **The grasp is a proxy in pick-place.**
   `held = near_cube AND jaw commanded closed`. The policy gets the grasp bonus without really holding anything.
   Their `so101_pick.py` found and patched these cheats (BUG C, D, E: open bonus, min_dz ratchet, descend cost), but **those fixes were NOT copied into `so101_pick_place.py`**.
   They used the proxy because the warp backend can't read `data.contact`.

3. **No domain randomization** of light, colour, friction, mass or camera. The CNN will overfit to perfect sim pixels, and sim-to-real will suffer.

4. **Success is paid every step, not once**, so `succ` in the logs means "fraction of steps", not "fraction of episodes". It's easy to misread.

5. **Many reward terms (12+ in pick).** Each new term was added to fix a cheat, and each one can create a new cheat. It's hard to reason about.

6. **The resume bug-guard is strict:** you can't extend a finished run, you must start a fresh one. That's fine, just know it.

---

## 5. Verdict

- **Madhan wins on training engineering:** speed (GPU batch), exploration-friendly actions, std clamp, LayerNorm, checkpoints, deploy code.
- **I win on realism and correctness:** real contact grasp, strict success, proper timeout handling, domain randomization, calibrated camera, curriculum, scripted expert.
- **Best plan:** keep my task definition and reward honesty, and borrow Madhan's action design and speed ideas.

---

## 6. Fixes for MY repo (do them in this order)

| # | Fix | Why | Where |
|---|---|---|---|
| 1 | **Absolute gripper**: `ctrl[5] = low + (a[5]+1)/2 × (high-low)` | One action opens or closes the jaw, so it explores grasping straight away | `envs/pick_place_env.py` `step()` |
| 2 | **Bigger arm step** (e.g. 0.05–0.08 rad/step) or Madhan-style absolute residual around a center pose | Reach the cube in fewer steps | `max_joint_delta` |
| 3 | **Stop the std collapse**: keep the default `log_std_init=0` (std = 1) or raise it, raise `ent_coef` to 0.02–0.05, and lower LR (1e-4) so KL stays under the cutoff. Watch `std`: if it drops below ~0.3 before grasps appear, exploration has collapsed. (Don't copy Madhan's −0.5: they have a std floor, SB3 doesn't.) | Keep exploring until grasps happen | `train_ppo.py` `policy_kwargs`, PPO args |
| 4 | **More envs**: `--num-envs 4–8` (RAM allowing; that lift run used only 1) and a smaller image that keeps the 16:9 shape, e.g. 72×128 before the crop | 4–8× more samples per hour | train command, `img_size` |
| 5 | **Use the scripted expert**: record ~200 demos, then do behaviour cloning (BC) as a warm start, or mix demos in | PPO starts already knowing roughly how to grasp | `controllers/scripted_pick_place.py` |
| 6 | **Easier first stage**: train in `state` mode first (no camera) to prove the reward works, then switch to RGB | If state mode fails, the problem is reward/actions, not vision | `--observation-mode state` |
| 7 | **Widen the cube spawn slowly**: ±1 cm → ±3 cm → ±6 cm once success > 80% | Forces the policy to actually use the camera | `reset()` |
| 8 | Later: look at MJX for speed (Madhan's code is the reference) | 10× speed, but a big rewrite | — |

⚠️ The scripted expert turns all 6 outputs (gripper too) into delta actions: `(target - ctrl) / max_joint_delta`. If you do fix 1, change its `action[5]` to a plain open (+1) / close (−1) command. If you do fix 2, its scaling changes too. Otherwise the demos for fix 5 will be wrong.

Quick check after fix 1: `ever_grasped` should rise well above 5/1300 within ~200k steps. If it doesn't, look at the reward next.

---

## 7. Mini glossary (simple words)

- **Delta action:** "move a little from where you are now." Smooth, but slow to explore.
- **Absolute / residual action:** "go to this position (around a center pose)." Faster to explore.
- **Slew limit:** a speed cap so the target can't jump too far in one step. Madhan uses it on absolute actions.
- **Potential-based shaping:** reward = γ·Φ(new) − Φ(old). It guides learning without changing what the best policy is. Hard to cheat.
- **Staged dense reward:** many small bonuses for each sub-step. Learns fast, but easy to cheat (reward hacking).
- **Termination vs truncation:** termination = the task really ended (success/fail), future value = 0. Truncation = time ran out, the future still exists, so bootstrap V(next).
- **Proxy grasp vs physical grasp:** a proxy *guesses* "is it holding?". A physical grasp checks real contacts.
- **Domain randomization:** randomly change light, mass, friction and camera in sim so the policy doesn't overfit and works on the real robot.
- **Entropy / std collapse:** the policy's noise shrinks too early, so it stops trying new things.
- **Behaviour cloning (BC):** copy an expert's actions with supervised learning before RL.

---

## 8. Remember this box

```
1. No grasp → no lift → no learning. Fix the GRIPPER first.
2. Vision RL needs 10M+ steps. 149 steps/sec is too slow.
3. Madhan = fast + easy actions.   Me = honest reward + realistic sim.
4. Timeout ≠ game over. (I do it right, Madhan doesn't.)
5. Proxy rewards get cheated. Real contacts don't.
6. Prove it in state mode first, then add the camera.
7. I already have an expert. Use it.
```
