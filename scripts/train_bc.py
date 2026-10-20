#!/usr/bin/env python3
"""Offline Behavior Cloning (BC) Trainer for SO-101 RGB Pick-and-Place Policy.

Learns directly from eligible recorded IK-teacher demonstrations.
Trains InitialPoseCNN actor with Smooth L1 loss and updates empirical observation
normalizer statistics on demonstration states.
Outputs an RSL-RL compatible warm-start checkpoint for PPO fine-tuning.
"""

import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Auto-detect and switch to env_isaaclab Python if executed in another conda environment
try:
    import torch
    from tensordict import TensorDict
except ImportError:
    isaac_py = "/home/ai/miniconda3/envs/env_isaaclab/bin/python"
    if os.path.exists(isaac_py) and sys.executable != isaac_py:
        os.execv(isaac_py, [isaac_py] + sys.argv)

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from isaac_vision_rl.utils.demonstrations import DemonstrationDataset, find_eligible_demonstrations
from isaac_vision_rl.utils.rl_agent import ExplorationPPO


def train_bc(
    demo_dirs,
    output_path,
    epochs=50,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    val_split=0.1,
    device="cuda:0",
    seed=42,
    quiet=False,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.manual_seed_all(seed)
    else:
        device = "cpu"

    os.environ.setdefault("SO101_POLICY", "rgb")
    os.environ.setdefault("SO101_REWARD_PROFILE", "four-stage")
    os.environ.setdefault("SO101_REWARD_STYLE", "continuous")
    os.environ.setdefault("SO101_GRASP_REWARD_MODE", "bounded")
    os.environ.setdefault("SO101_TRAINING_GOAL", "place")
    os.environ.setdefault("SO101_RGB_FPS", "30")

    if not quiet:
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Loading demonstrations from {len(demo_dirs)} directories...")

    dataset = DemonstrationDataset(demo_dirs)
    n_total = len(dataset)
    if not quiet:
        print(f"[INFO] Total demonstration transitions loaded: {n_total}")

    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # Instantiate InitialPoseCNN actor and MLPModel critic using standard PPORunnerCfg
    cfg = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
    config = cfg.to_dict()
    config["multi_gpu"] = None

    mock_obs = TensorDict(
        {"policy": torch.zeros(4, 29), "critic": torch.zeros(4, 51), "image": torch.zeros(4, 3, 128, 128)},
        batch_size=[4],
    )
    mock_env = SimpleNamespace(num_envs=4, num_actions=6)
    alg = ExplorationPPO.construct_algorithm(mock_obs, mock_env, copy.deepcopy(config), device)
    actor = alg.actor
    critic = alg.critic

    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        actor.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            policy = batch["policy"].to(device)
            image = batch["image"].to(device)
            target_action = batch["action"].to(device)
            bs = policy.shape[0]

            obs = TensorDict({"policy": policy, "image": image}, batch_size=[bs])

            # Update empirical normalizer with demonstration policy features
            actor.update_normalization(obs)

            pred_action = actor(obs, stochastic_output=False)
            loss = loss_fn(pred_action, target_action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * bs
            train_samples += bs

        train_loss = running_train_loss / max(1, train_samples)

        actor.eval()
        running_val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                policy = batch["policy"].to(device)
                image = batch["image"].to(device)
                target_action = batch["action"].to(device)
                bs = policy.shape[0]

                obs = TensorDict({"policy": policy, "image": image}, batch_size=[bs])
                pred_action = actor(obs, stochastic_output=False)
                loss = loss_fn(pred_action, target_action)

                running_val_loss += loss.item() * bs
                val_samples += bs

        val_loss = running_val_loss / max(1, val_samples)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if not quiet and (epoch % max(1, epochs // 10) == 0 or epoch == epochs or epoch == 1):
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f}"
            )

    # Package checkpoint for PPO warm-start
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cam_w = os.environ.get("SO101_CAMERA_WIDTH", "1280")
    cam_h = os.environ.get("SO101_CAMERA_HEIGHT", "720")

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),  # Fresh initial critic
        "optimizer_state_dict": {},  # Fresh optimizer for PPO
        "iter": 0,
        "infos": {
            "source": "behavior_cloning",
            "epochs": epochs,
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "total_samples": n_total,
            "train_samples": n_train,
            "val_samples": n_val,
            "demos": [str(d) for d in demo_dirs],
        },
        "is_bc_checkpoint": True,
        "so101_agent_schema": "four_stage_v2_bounded",
        "so101_policy_variant": "rgb",
        "so101_reward_profile": "four-stage",
        "so101_reward_style": "continuous",
        "so101_control_hz": int(os.environ.get("SO101_RGB_FPS", "30")),
        "so101_rgb_profile": f"{cam_w}x{cam_h}_letterbox128_v1",
        "exploration_entropy_coef": 0.01,
    }

    torch.save(checkpoint, out)
    if not quiet:
        print(f"\n[INFO] Behavior Cloning checkpoint successfully saved to: {out}")
        print(f"[INFO] Final Train Loss: {history['train_loss'][-1]:.5f} | Val Loss: {history['val_loss'][-1]:.5f}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--demos",
        nargs="+",
        default=None,
        help="Paths to eligible demonstration folders (default: auto-discover all eligible in outputs/four_stage)",
    )
    parser.add_argument(
        "--output",
        default="outputs/four_stage/bc_policy.pt",
        help="Target checkpoint file (.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation data split fraction")
    parser.add_argument("--device", default="cuda:0", help="Computation device (cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (suppress periodic progress prints)")
    args = parser.parse_args()

    demo_dirs = args.demos
    if demo_dirs is None:
        eligible = find_eligible_demonstrations("outputs/four_stage")
        if not eligible:
            parser.error("No eligible demonstrations found in outputs/four_stage. Provide paths with --demos.")
        demo_dirs = [str(d) for d in eligible]
        print(f"[INFO] Auto-discovered {len(demo_dirs)} eligible demonstration directories:")
        for d in demo_dirs:
            print(f"  - {d}")

    train_bc(
        demo_dirs=demo_dirs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

