"""Train box placement from an existing vision pick checkpoint.

python scripts/pick_place.py --check-only
python scripts/pick_place.py --run runs/pick_place_v1

Image DR, moving scene light, cube/box position DR default ON; camera is fixed.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import json
import warp as wp
wp.config.quiet = True
wp.config.log_level = wp.LOG_WARNING
from vision_rl.train import load_config, train
from vision_rl.envs.pick_place_env import PlaceConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="runs/first_lift40_live")
    p.add_argument("--ckpt")
    p.add_argument("--run", default="runs/pick_place_v1")
    p.add_argument("--steps", type=float, default=50e6)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--box-range", type=float, default=.015)
    p.add_argument("--light-range", type=float, default=.12)
    p.add_argument("--check-only", action="store_true", help="print configuration; do not build or train")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    out = Path(a.run)
    if a.resume:
        cfg = load_config(a.run)
        if cfg.task != "pick_place":
            p.error("Resume requires a pick_place run; source pick run is never modified")
    else:
        source = Path(a.source)
        checkpoint = source / (a.ckpt or (source / "latest.txt").read_text().strip())
        if not checkpoint.is_file():
            p.error(f"Checkpoint not found: {checkpoint}")
        original = load_config(str(source))
        cfg = replace(original, task="pick_place", initial_checkpoint=str(checkpoint.resolve()),
                      env=replace(original.env, episode_steps=600,
                                  solver_iterations=max(original.env.solver_iterations, 80),
                                  ls_iterations=max(original.env.ls_iterations, 60)),
                      ppo=replace(original.ppo, num_envs=a.num_envs, total_steps=int(a.steps),
                                  learning_rate=1e-4),
                      place=PlaceConfig(box_range=a.box_range, light_range=a.light_range),
                      image_dr=True, every_step=False, log_every=10, ckpt_every=200,
                      contacts_per_world=max(original.contacts_per_world, 128),
                      njmax=max(original.njmax, 512))
        if out.exists():
            p.error(f"Output already exists: {out}; use a fresh folder or --resume")
    if a.num_envs < 1 or a.box_range < 0 or a.light_range < 0 or a.steps <= 0:
        p.error("Invalid environment count, steps, or randomisation range")
    if a.check_only:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    train(cfg, a.run, resume=a.resume)


if __name__ == "__main__":
    main()
