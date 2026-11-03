"""Vision PPO training, fully on the GPU.

One jitted `iteration` = collect `rollout_steps` x `num_envs` transitions
(Warp physics + Warp ray-traced camera + policy, all inside lax.scan), then
GAE and PPO updates. The host only prints, writes metrics.csv and saves
checkpoints.

Run with `python scripts/train.py --help`.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState

from vision_rl.envs.pick_env import PickEnv, PickEnvConfig
from vision_rl.envs.vec_env import VecEnv
from vision_rl.models import ActorCritic
from vision_rl.ppo import PPOConfig, gae, make_optimizer, ppo_update
from vision_rl.rendering import load_lowpoly_model
from vision_rl.rewards.pick import PickRewardConfig


@dataclass(frozen=True)
class TrainConfig:
    env: PickEnvConfig = field(default_factory=PickEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    image_size: int = 128
    frame_stack: int = 3
    shadows: bool = True
    exact_render: bool = False
    image_dr: bool = False
    contacts_per_world: int = 64
    njmax: int = 256
    log_every: int = 10                # iterations between progress lines
    ckpt_every: int = 200              # iterations
    every_step: bool = False           # also print one numbered line per robot step

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        env = dict(d["env"])
        env["reward"] = PickRewardConfig(**env["reward"])
        return cls(**{**d, "env": PickEnvConfig(**env), "ppo": PPOConfig(**d["ppo"])})


def build(cfg: TrainConfig):
    model = load_lowpoly_model()
    env = PickEnv(model, cfg.env, num_envs=cfg.ppo.num_envs,
                  contacts_per_world=cfg.contacts_per_world, njmax=cfg.njmax)
    venv = VecEnv(env, cfg.ppo.num_envs, frame_stack=cfg.frame_stack, image_size=cfg.image_size,
                  shadows=cfg.shadows, exact_render=cfg.exact_render, image_dr=cfg.image_dr)
    net = ActorCritic(action_dim=env.action_size, image_size=cfg.image_size)
    return venv, net


class ProgressUI:
    """Console UI, per log interval:

        [Step 81,920/49,995,776] ─── Processing: 0% |▒▒▒▒...| [00:01:02 < 08:30:11]
        Metrics ─── Time: 5.61s | Mean Reward/Step: -0.7520 | Mean Ep Len: 250.00 | Loss: 6.2466 | ...

    The Step line also stays on the bottom of the screen and is redrawn every
    iteration. When stdout is not a terminal (piped to a log file) only the
    per-interval lines are written.
    """

    WIDTH = 30

    def __init__(self, total_it: int, batch: int, enabled: bool = True):
        self.total_it, self.batch = total_it, batch
        self.live = enabled and sys.stdout.isatty()
        self.it, self.sps, self.t0, self.it0 = 0, 0.0, time.time(), None
        self.t_start = time.time()

    @staticmethod
    def _hms(s):
        s = int(s)
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def _bar(self):
        frac = self.it / max(self.total_it, 1)
        full = int(frac * self.WIDTH)
        left = self._hms((self.total_it - self.it) * self.batch / self.sps) if self.sps > 0 else "--:--:--"
        total = self.total_it * self.batch
        return (f"[Step {self.it * self.batch:,}/{total:,}] ─── Processing: {100 * frac:.0f}% "
                f"|{'█' * full}{'▒' * (self.WIDTH - full)}| "
                f"[{self._hms(time.time() - self.t_start)} < {left}]")

    def update(self, it: int):
        if self.it0 is None:
            self.it0, self.t0 = it, time.time()      # skip the compile time in sps / ETA
        elif it > self.it0:
            self.sps = (it - self.it0) * self.batch / (time.time() - self.t0)
        self.it = it
        if self.live:
            sys.stdout.write("\r\033[K" + self._bar())
            sys.stdout.flush()

    def print(self, text: str):
        """A permanent line above the bar."""
        sys.stdout.write(("\r\033[K" if self.live else "") + text + "\n")
        if self.live:
            sys.stdout.write(self._bar())
        sys.stdout.flush()

    def row(self, r, iter_time, ep_len, loss):
        """Per log interval: the Step line + a Metrics line (task numbers at the end)."""
        self.print(self._bar())
        self.print(f"Metrics ─── Time: {iter_time:.3f}s | Mean Reward/Step: {r['reward']:.4f} | "
                   f"Mean Ep Len: {ep_len:.2f} | Loss: {loss:.4f} | "
                   f"Success: {r['ep_success']:.2f} | Held: {r['ep_held']:.2f} | "
                   f"Lift: {r['cube_lift'] * 100:+.1f}cm")

    def close(self):
        if self.live:
            sys.stdout.write("\r\033[K" + self._bar() + "\n")
            sys.stdout.flush()


class StepPrinter:
    """One numbered line per robot step, printed from inside the jitted rollout.

    All worlds step at the same instant on the GPU; their results come back
    together and are printed as consecutive lines (world 0, 1, ..., N-1), so the
    step number goes up by exactly 1 per line.
    """

    def __init__(self, total: int):
        self.total, self.step = total, 0

    def __call__(self, reward, aligned, held, d_xy, lift):
        lines = []
        for w in range(len(reward)):
            self.step += 1
            lines.append(f"step {self.step:>11,} / {self.total:,} | world {w:>3} | reward {reward[w]:+.3f} | "
                         f"aligned {int(aligned[w])} | held {int(held[w])} | "
                         f"d_xy {d_xy[w] * 100:5.1f}cm | lift {lift[w] * 100:+.1f}cm")
        sys.stdout.write("\n".join(lines) + "\n")


def make_iteration(cfg: TrainConfig, venv: VecEnv, printer: StepPrinter | None = None):
    p = cfg.ppo

    def rollout(state: TrainState, vs, key):
        def step(carry, k):
            vs = carry
            k_act, _ = jax.random.split(k)
            pi, value = state.apply_fn(state.params, vs.obs)
            action = pi.sample(k_act)
            nxt = venv.step(vs, action)
            e = nxt.env
            done = e.done
            tr = {"obs": vs.obs, "actions": action, "log_probs": pi.log_prob(action), "values": value,
                  "rewards": e.reward, "dones": done,
                  # per-step task stats
                  **{f"m/{k}": v for k, v in e.metrics.items()},
                  # finished-episode stats (summed; divided by the episode count later)
                  "ep/n": done, "ep/return": done * nxt.last_return,
                  "ep/success": done * nxt.last_success, "ep/held": done * nxt.last_held,
                  "ep/len": done * nxt.last_len}
            if printer is not None:
                mt = e.metrics
                jax.debug.callback(printer, e.reward, mt["aligned"], mt["held"], mt["d_xy"],
                                   mt["cube_lift"], ordered=True)
            return nxt, tr

        vs, traj = jax.lax.scan(step, vs, jax.random.split(key, p.rollout_steps))
        _, last_value = state.apply_fn(state.params, vs.obs)
        return vs, traj, last_value

    @jax.jit
    def iteration(state: TrainState, vs, key):
        k_roll, k_upd = jax.random.split(key)
        vs, traj, last_value = rollout(state, vs, k_roll)
        adv, ret = gae(traj["rewards"], traj["values"], traj["dones"], last_value, p.gamma, p.gae_lambda)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        batch = {"obs": jax.tree.map(flat, traj["obs"]), "actions": flat(traj["actions"]),
                 "log_probs": flat(traj["log_probs"]), "values": flat(traj["values"]),
                 "advantages": flat(adv), "returns": flat(ret)}
        state, m = ppo_update(state, batch, k_upd, p)
        m["reward"] = traj["rewards"].mean()
        for k, v in traj.items():
            if k.startswith("m/"):
                m[k[2:]] = v.mean()
        for k in ("return", "success", "held", "len"):
            m[f"ep_{k}_sum"] = traj[f"ep/{k}"].sum()
        m["ep_count"] = traj["ep/n"].sum()
        return state, vs, m

    return iteration


# ---------------------------------------------------------------------- checkpoints
def save_checkpoint(run_dir, state: TrainState, it: int, steps: int, cfg: TrainConfig):
    path = os.path.join(run_dir, f"ckpt_{steps:010d}.msgpack")
    blob = serialization.to_bytes({"params": state.params, "opt_state": state.opt_state,
                                   "iteration": it, "steps": steps})
    with open(path + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(path + ".tmp", path)
    with open(os.path.join(run_dir, "latest.txt"), "w") as f:
        f.write(os.path.basename(path))
    return path


def load_checkpoint(path, state: TrainState):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        d = serialization.from_bytes({"params": state.params, "opt_state": state.opt_state,
                                      "iteration": 0, "steps": 0}, raw)
        return state.replace(params=d["params"], opt_state=d["opt_state"]), int(d["iteration"]), int(d["steps"])
    except Exception:
        d = serialization.from_bytes({"params": state.params, "iteration": 0, "steps": 0}, raw)
        return state.replace(params=d["params"]), int(d.get("iteration", 0)), int(d.get("steps", 0))


def load_config(run_dir) -> TrainConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        return TrainConfig.from_dict(json.load(f))


# ---------------------------------------------------------------------- logging
def _tensorboard(run_dir):
    """TensorBoard event files in run_dir/tb (`tensorboard --logdir runs`). None if not installed."""
    try:
        from tensorboard.summary import Writer
    except ImportError:
        return None
    return Writer(os.path.join(run_dir, "tb"))


def _tb_tag(k):
    if k.startswith("ep_") or k == "episodes":
        return f"1_episode/{k}"
    if k in ("aligned", "held", "lifted", "success", "d_xy", "cube_lift", "fell", "unstable", "reward"):
        return f"2_task/{k}"
    if k.endswith(("_r", "_pr", "_pen", "_cost")):
        return f"3_reward_terms/{k}"
    return f"4_ppo/{k}"


def _wandb(run_dir, cfg: TrainConfig, project: str, name: str | None):
    """Start (or, on --resume, continue) a W&B run. Its id is kept in run_dir/wandb_id.txt."""
    import wandb

    id_file = os.path.join(run_dir, "wandb_id.txt")
    run_id = open(id_file).read().strip() if os.path.exists(id_file) else uuid.uuid4().hex[:8]
    with open(id_file, "w") as f:
        f.write(run_id)
    run = wandb.init(project=project, name=name or os.path.basename(os.path.abspath(run_dir)),
                     id=run_id, resume="allow", config=cfg.to_dict(), dir=run_dir)
    print(f"[wandb] {run.url}")
    return run


# ---------------------------------------------------------------------- main loop
def train(cfg: TrainConfig, run_dir: str, resume: bool = False,
          wandb_project: str | None = None, wandb_name: str | None = None):
    """wandb_project=None: no W&B. TensorBoard + metrics.csv are always written."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    p = cfg.ppo
    print(f"[init] devices: {jax.devices()}")
    t0 = time.time()
    venv, net = build(cfg)
    key = jax.random.PRNGKey(p.seed)
    key, k_init, k_reset = jax.random.split(key, 3)
    state = TrainState.create(apply_fn=net.apply, params=net.init(k_init, venv.sample_obs()),
                              tx=make_optimizer(p))
    start_it, steps = 0, 0
    if resume:
        with open(os.path.join(run_dir, "latest.txt")) as f:
            ck = os.path.join(run_dir, f.read().strip())
        state, start_it, steps = load_checkpoint(ck, state)
        print(f"[resume] {ck} (iteration {start_it}, {steps:,} steps)")
    n_params = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"[init] {p.num_envs} worlds, image {cfg.image_size}x{cfg.image_size}x{3 * cfg.frame_stack}, "
          f"{n_params:,} params, {p.num_iterations} iterations of {p.batch_size:,} steps")
    print(f"[init] warp buffers: {venv.env.naconmax:,} contacts total, {venv.env.njmax} rows/world")
    printer = None
    if cfg.every_step:
        printer = StepPrinter(p.num_iterations * p.batch_size)
        printer.step = steps                               # continue the count on --resume
    iteration = make_iteration(cfg, venv, printer)
    vs = jax.jit(venv.reset)(k_reset)
    print(f"[init] built in {time.time() - t0:.0f}s; compiling the training step (1-3 min)...")

    csv_path = os.path.join(run_dir, "metrics.csv")
    new_csv = not os.path.exists(csv_path) or not resume
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    writer = None
    tb = _tensorboard(run_dir)
    wb = _wandb(run_dir, cfg, wandb_project, wandb_name) if wandb_project else None
    acc, t_log, steps_log = [], time.time(), steps
    ui = ProgressUI(p.num_iterations, p.batch_size, enabled=not cfg.every_step)
    for it in range(start_it + 1, p.num_iterations + 1):
        key, k = jax.random.split(key)
        state, vs, m = iteration(state, vs, k)
        steps += p.batch_size
        acc.append(m)
        jax.block_until_ready(m)                       # step count = finished steps
        ui.update(it)
        if it == start_it + 1:                         # this call also compiled the step
            jax.block_until_ready(m)
            ui.print(f"[init] compiled in {time.time() - t_log:.0f}s")
            acc, t_log, steps_log = [], time.time(), steps
            continue
        if it % cfg.log_every == 0 or it == p.num_iterations:
            ms = jax.device_get(acc)
            row = {k: float(np.mean([x[k] for x in ms])) for k in ms[0] if not k.startswith("ep_")}
            n_ep = sum(float(x["ep_count"]) for x in ms)
            for k in ("return", "success", "held", "len"):
                row[f"ep_{k}"] = sum(float(x[f"ep_{k}_sum"]) for x in ms) / max(n_ep, 1.0)
            row["episodes"] = n_ep
            duration = time.time() - t_log
            sps = (steps - steps_log) / duration
            row.update(iteration=it, steps=steps, sps=sps)
            n_it = len(ms)
            loss = row["pg_loss"] + p.value_coef * row["vf_loss"] - p.entropy_coef * row["entropy"]
            ui.row(row, duration / n_it, row["ep_len"], loss)
            if writer is None:
                cols = ["iteration", "steps", "sps", "episodes", "ep_return", "ep_success", "ep_held"] + \
                       sorted(k for k in row if k not in ("iteration", "steps", "sps", "episodes",
                                                          "ep_return", "ep_success", "ep_held"))
                writer = csv.DictWriter(csv_f, fieldnames=cols, extrasaction="ignore")
                if new_csv:
                    writer.writeheader()
            writer.writerow(row)
            csv_f.flush()
            if tb is not None:
                for k, v in row.items():
                    if k not in ("iteration", "steps"):
                        tb.add_scalar(_tb_tag(k), v, step=steps)
                tb.flush()
            if wb is not None:
                wb.log({_tb_tag(k): v for k, v in row.items() if k not in ("iteration", "steps")}, step=steps)
            acc, t_log, steps_log = [], time.time(), steps
        if it % cfg.ckpt_every == 0 or it == p.num_iterations:
            path = save_checkpoint(run_dir, state, it, steps, cfg)
            ui.print(f"[ckpt] {path}")
    ui.close()
    csv_f.close()
    if tb is not None:
        tb.close()
    if wb is not None:
        wb.finish()
    return state
