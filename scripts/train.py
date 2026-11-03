"""Train the vision PPO pick policy on the GPU.

    python scripts/train.py                              # defaults: 256 worlds, 128 px, 50M steps
    python scripts/train.py --image-dr --run runs/pick_dr
    python scripts/train.py --resume runs/pick           # continue from the latest checkpoint
    python scripts/train.py --smoke                      # 2-minute pipeline check
    python scripts/train.py --wandb                      # also log to Weights & Biases

Outputs in --run: config.json, metrics.csv, tb/ (TensorBoard), ckpt_*.msgpack, latest.txt.
Watch live:  tensorboard --logdir runs    (then open http://localhost:6006)
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# JAX grabs 75% of the GPU up front by default; Warp (physics + renderer) has its own
# allocator and then runs out of memory building its CUDA graphs. Allocate on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.stdout.reconfigure(line_buffering=True)   # progress lines show up even when piped to a log

import warp as wp  # noqa: E402

wp.config.quiet = True                        # hide Warp's start-up banner
wp.config.log_level = wp.LOG_WARNING          # hide "Module ... load on device" lines (warnings still show)

from vision_rl.envs.pick_env import PickEnvConfig  # noqa: E402
from vision_rl.ppo import PPOConfig  # noqa: E402
from vision_rl.rewards.pick import PickRewardConfig  # noqa: E402
from vision_rl.train import TrainConfig, load_config, train  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", default="runs/pick", help="output folder")
ap.add_argument("--resume", metavar="RUN", help="continue RUN from its latest checkpoint (its own config)")
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--steps", type=float, default=50e6, help="total env steps")
ap.add_argument("--rollout", type=int, default=32, help="steps per world per iteration")
ap.add_argument("--minibatches", type=int, default=8)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
ap.add_argument("--entropy", type=float, default=0.005)
ap.add_argument("--max-joint-delta", type=float, default=None,
                help="max joint delta per step (rad; default 0.04, lower = slower)")
ap.add_argument("--rate-scale", type=float, default=None,
                help="penalty scale on action difference between consecutive steps (default 0.02)")
ap.add_argument("--success-bonus", type=float, default=None, help="bonus on successful lift (default 5.0)")
ap.add_argument("--grasp-bonus", type=float, default=None, help="bonus per step for holding (default 0.1)")
ap.add_argument("--image-size", type=int, default=128, help="policy image is SIZE x SIZE (224 = camera.py default)")
ap.add_argument("--frame-stack", type=int, default=3)
ap.add_argument("--no-shadows", action="store_true", help="~1.6x faster rendering")
ap.add_argument("--exact-render", action="store_true", help="render 424x240 and squash (slower, pixel-exact)")
ap.add_argument("--image-dr", action="store_true", help="brightness/colour/noise randomisation per episode")
ap.add_argument("--cube-range", type=float, default=0.03, help="cube spawn: home spot +/- this (m)")
ap.add_argument("--episode-steps", type=int, default=250)
ap.add_argument("--pick-height", type=float, default=None, help="success lift (m); reward default 0.085 (clears the 7.5 cm box walls)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wandb", nargs="?", const="vision_rl", metavar="PROJECT",
                help="also log to Weights & Biases (project default: vision_rl); run `wandb login` once first")
ap.add_argument("--wandb-name", help="W&B run name (default: the --run folder name)")
ap.add_argument("--log-every", type=int, default=None,
                help="print a progress line every N iterations (default 10, ~1 min); works with --resume")
ap.add_argument("--every-step", action="store_true",
                help="print one numbered line per robot step (~1,450 lines/s at 256 worlds); works with --resume")
ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
a = ap.parse_args()

if a.resume:
    cfg = load_config(a.resume)
    run = a.resume
    resume = True
else:
    reward = PickRewardConfig()
    rew_updates = {}
    if a.pick_height is not None:
        rew_updates["pick_height"] = a.pick_height
    if a.rate_scale is not None:
        rew_updates["rate_scale"] = a.rate_scale
    if a.success_bonus is not None:
        rew_updates["success_bonus"] = a.success_bonus
    if a.grasp_bonus is not None:
        rew_updates["grasp_bonus"] = a.grasp_bonus
    if rew_updates:
        reward = dataclasses.replace(reward, **rew_updates)
    ppo = PPOConfig(num_envs=a.num_envs, rollout_steps=a.rollout, num_minibatches=a.minibatches,
                    update_epochs=a.epochs, learning_rate=a.lr, weight_decay=a.weight_decay,
                    entropy_coef=a.entropy, total_steps=int(a.steps), seed=a.seed)
    env_kwargs = dict(cube_range=a.cube_range, episode_steps=a.episode_steps, reward=reward)
    if a.max_joint_delta is not None:
        env_kwargs["max_joint_delta"] = a.max_joint_delta
    env = PickEnvConfig(**env_kwargs)
    cfg = TrainConfig(env=env, ppo=ppo, image_size=a.image_size, frame_stack=a.frame_stack,
                      shadows=not a.no_shadows, exact_render=a.exact_render, image_dr=a.image_dr)
    if a.smoke:
        cfg = dataclasses.replace(cfg, log_every=2, ckpt_every=10,
                                  ppo=dataclasses.replace(ppo, num_envs=32, rollout_steps=16,
                                                          num_minibatches=4, total_steps=32 * 16 * 10))
    run = a.run
    resume = False
    if os.path.exists(os.path.join(run, "latest.txt")):
        sys.exit(f"{run} already has checkpoints: use --resume {run} or pick another --run")

if a.log_every is not None:
    cfg = dataclasses.replace(cfg, log_every=max(1, a.log_every))
if a.every_step:
    cfg = dataclasses.replace(cfg, every_step=True)

train(cfg, run, resume=resume, wandb_project=a.wandb, wandb_name=a.wandb_name)
