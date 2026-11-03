"""Watch a trained pick policy: live 3D viewer + what the robot's camera sees.

    python scripts/watch_policy.py                          # runs/pick, latest checkpoint, live
    python scripts/watch_policy.py --world 3 --speed 0.5    # another world, half speed
    python scripts/watch_policy.py --ckpt ckpt_0016384000.msgpack
    python scripts/watch_policy.py --video eval.mp4         # record instead (no window needed)
    python scripts/watch_policy.py --run runs/pick_v2 --follow   # watch WHILE training:
                                         # loads each new checkpoint (~every 20 min), runs until closed

The policy acts on --worlds GPU worlds at once (mean action unless --stochastic); one of
them (--world) is shown. Each step prints one line; each episode ends with a summary of
all worlds. Close the viewer window or press Ctrl+C to stop.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.stdout.reconfigure(line_buffering=True)

import warp as wp  # noqa: E402

wp.config.quiet = True
wp.config.log_level = wp.LOG_WARNING

import cv2  # noqa: E402
import dataclasses  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from vision_rl.envs import SCENE_XML  # noqa: E402
from vision_rl.ppo import make_optimizer  # noqa: E402
from vision_rl.train import build, load_checkpoint, load_config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", default="runs/pick")
ap.add_argument("--ckpt", help="checkpoint file in --run (default: latest.txt)")
ap.add_argument("--worlds", type=int, default=1, help="worlds run in parallel (1 = only the one you see)")
ap.add_argument("--world", type=int, default=0, help="which world to show")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
ap.add_argument("--stochastic", action="store_true", help="sample actions (training noise) instead of the mean")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--video", help="write an mp4 here instead of opening windows")
ap.add_argument("--follow", action="store_true",
                help="keep running and load each new checkpoint of a run that is still training")
ap.add_argument("--no-robot-view", action="store_true", help="hide the camera-view window")
a = ap.parse_args()

# ---------------------------------------------------------------- policy + GPU env
if not os.path.exists(os.path.join(a.run, "latest.txt")):
    if not a.follow:
        sys.exit(f"{a.run} has no checkpoint yet (is training running? use --follow to wait)")
    print(f"waiting for the first checkpoint in {a.run} (saved ~20 min after training starts) ...")
    while not os.path.exists(os.path.join(a.run, "latest.txt")):
        time.sleep(10)
cfg = load_config(a.run)
cfg = dataclasses.replace(cfg, ppo=dataclasses.replace(cfg.ppo, num_envs=a.worlds))
venv, net = build(cfg)
env = venv.env
vs = venv.reset(jax.random.PRNGKey(a.seed))
params = net.init(jax.random.PRNGKey(0), jax.tree.map(lambda x: x[:1], vs.obs))
state = TrainState.create(apply_fn=net.apply, params=params, tx=make_optimizer(cfg.ppo))
ckpt = a.ckpt or open(os.path.join(a.run, "latest.txt")).read().strip()
state, _, steps = load_checkpoint(os.path.join(a.run, ckpt), state)
print(f"policy {a.run}/{ckpt} ({steps:,} training steps), {a.worlds} worlds, showing world {a.world}, "
      f"{'stochastic' if a.stochastic else 'mean'} actions")

cq = env.cube_q


@jax.jit
def policy_step(params, vs, key):
    pi, value = state.apply_fn(params, vs.obs)
    act = pi.sample(key) if a.stochastic else pi.mean
    nxt = venv.step(vs, act)
    q = nxt.env.data.qpos
    w, x, y, z = (q[:, cq + 3 + i] for i in range(4))
    r2 = jnp.abs(jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1))
    tilt = jnp.degrees(jnp.arccos(jnp.clip(r2.max(-1), 0.0, 1.0)))
    m = nxt.env.metrics
    info = dict(reward=nxt.env.reward, done=nxt.env.done, held=m["held"], aligned=m["aligned"],
                lift=m["cube_lift"], d_xy=m["d_xy"], success=m["success"], tilt=tilt,
                value=value.reshape(value.shape[0], -1)[:, 0], ep_return=nxt.last_return,
                ep_success=nxt.last_success, ep_held=nxt.last_held)
    return nxt, info, q, nxt.frames[..., -3:]


# ---------------------------------------------------------------- display (full-detail meshes)
vm = mujoco.MjModel.from_xml_path(SCENE_XML)
vd = mujoco.MjData(vm)
assert vm.nq == vs.env.data.qpos.shape[1], "viewer scene and training scene differ"

viewer = writer = renderer = None
if a.video:
    renderer = mujoco.Renderer(vm, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, -0.1, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 135, -30
    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"),
                             1.0 / cfg.env.ctrl_dt, (640 + 480, 480))
else:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(vm, vd, show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = [0.2, -0.1, 0.45]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.75, 135, -30
    if not a.no_robot_view:
        cv2.namedWindow("robot camera (policy input)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("robot camera (policy input)", 384, 384)

# ---------------------------------------------------------------- run
key = jax.random.PRNGKey(a.seed + 1)
dt = cfg.env.ctrl_dt / a.speed if a.speed > 0 else 0.0
w = a.world
ep, t, ep_ret = 1, 0, 0.0
results = []
print(f"\n=== episode {ep} ===")
try:
    while a.follow or ep <= a.episodes:
        t0 = time.time()
        key, k = jax.random.split(key)
        vs, info, qpos, img = policy_step(state.params, vs, k)
        info = jax.device_get(info)
        t += 1
        ep_ret += float(info["reward"][w])
        print(f"step {t:3d} | reward {info['reward'][w]:+.3f} | return {ep_ret:+7.2f} | "
              f"aligned {int(info['aligned'][w])} held {int(info['held'][w])} | "
              f"d_xy {info['d_xy'][w] * 100:4.1f}cm lift {info['lift'][w] * 100:+5.2f}cm "
              f"tilt {info['tilt'][w]:4.1f}deg | value {info['value'][w]:+6.2f}"
              + ("  SUCCESS" if info["success"][w] > 0 else ""))

        vd.qpos[:] = np.asarray(qpos[w])
        mujoco.mj_forward(vm, vd)
        robot_view = cv2.resize(np.asarray(img[w]), (480, 480), interpolation=cv2.INTER_NEAREST)
        if writer is not None:
            renderer.update_scene(vd, cam)
            frame = np.concatenate([renderer.render(), robot_view], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            if not viewer.is_running():
                break
            viewer.sync()
            if not a.no_robot_view:
                cv2.imshow("robot camera (policy input)", cv2.cvtColor(robot_view, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        if info["done"][w] > 0:   # all worlds share the same episode clock
            d = info["done"] > 0
            r = dict(ret=info["ep_return"][d].mean(), succ=info["ep_success"][d].mean(),
                     held=info["ep_held"][d].mean())
            results.append(r)
            print(f"--- episode {ep} done: shown world return {ep_ret:+.1f} | all {d.sum()} worlds: "
                  f"return {r['ret']:+.1f}, success {r['succ']:.2f}, held {r['held']:.2f}")
            ep, t, ep_ret = ep + 1, 0, 0.0
            if a.follow:
                newest = open(os.path.join(a.run, "latest.txt")).read().strip()
                if newest != ckpt:
                    ckpt = newest
                    state, _, steps = load_checkpoint(os.path.join(a.run, ckpt), state)
                    print(f"\n*** loaded new checkpoint {ckpt} ({steps:,} training steps) ***")
            if a.follow or ep <= a.episodes:
                print(f"\n=== episode {ep} ===")
        if dt:
            time.sleep(max(0.0, dt - (time.time() - t0)))
except KeyboardInterrupt:
    pass
finally:
    if writer is not None:
        writer.release()
        renderer.close()
        print(f"video saved: {a.video}")
    if viewer is not None:
        viewer.close()
    cv2.destroyAllWindows()

if results:
    print(f"\nover {len(results)} episode(s) x {a.worlds} worlds: "
          f"return {np.mean([r['ret'] for r in results]):+.1f}, "
          f"success {np.mean([r['succ'] for r in results]):.2f}, "
          f"held {np.mean([r['held'] for r in results]):.2f}")
