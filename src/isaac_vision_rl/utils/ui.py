# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ultra-clean, single-line dynamic progress bar for headless training."""

from __future__ import annotations

import atexit
import datetime
import os
import pathlib
import shutil
import statistics
import sys

import torch


class SingleLineTrainingBar:
    """Dynamic single-line progress bar that updates in place without scrolling."""

    def __init__(self, task_name: str = "IsaacVisionRl-Pickplace-SO101"):
        self.task_name = task_name
        self.is_tty = sys.stdout.isatty()
        self.start_datetime = datetime.datetime.now()
        atexit.register(self.close)

    def update(
        self,
        it: int,
        total_it: int,
        sps: float,
        elapsed_s: str,
        eta_s: str,
        rewards: dict[str, float],
        losses: dict[str, float],
        terms: dict[str, float],
        device: str = "cuda:0",
        num_envs: int = 16,
        log_dir: str = "",
    ) -> None:
        """Render single-line progress update."""
        term_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        it = it + 1  # RSL-RL reports zero-based iteration indices.
        pct = (it / total_it) * 100.0 if total_it > 0 else 0.0

        mean_rew = rewards.get("Mean Total", 0.0)
        reach = rewards.get("Reach Cube", 0.0)
        lift = rewards.get("Lift Cube", 0.0)
        place = rewards.get("Place Cube", 0.0)
        v_loss = losses.get("Value Loss", 0.0)
        exploration = f" | Std:{losses.get('Action Std', 0.0):.2f} | S:{terms.get('Success', 0.0):.2f}"

        # Core statistics prefix & suffix
        prefix = f"[{it:4d}/{total_it}] "
        suffix = (
            f" {pct:5.1f}% | {int(sps):,} sps | Rew:{mean_rew:+.2f} "
            f"(R:{reach:+.2f} L:{lift:+.2f} P:{place:+.2f}) | Loss:{v_loss:.2e}{exploration} | ETA:{eta_s}"
        )

        # Calculate progress bar width based on terminal size
        used_len = len(prefix) + len(suffix) + 2
        bar_width = max(term_width - used_len - 2, 8)
        filled = int(bar_width * it / total_it) if total_it > 0 else 0
        bar_str = "━" * filled + ("╸" if filled < bar_width else "")
        bar_str = bar_str.ljust(bar_width, "─")

        if self.is_tty:
            # Color-coded single line with \r and clear-to-end-of-line \033[K
            rew_color = 32 if mean_rew >= 0 else 31
            line = (
                f"\r\033[1;32m{prefix}\033[0m"
                f"\033[36m[{bar_str}]\033[0m"
                f"\033[1;37m {pct:5.1f}%\033[0m | "
                f"\033[1;33m{int(sps):,} sps\033[0m | "
                f"\033[1mRew:\033[0m \033[1;{rew_color}m{mean_rew:+.2f}\033[0m "
                f"\033[90m(R:{reach:+.2f} L:{lift:+.2f} P:{place:+.2f})\033[0m | "
                f"\033[33mLoss:{v_loss:.2e}{exploration}\033[0m | "
                f"\033[35mETA:{eta_s}\033[0m\033[K"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            # When writing to a log file or redirected pipe
            line = (
                f"[{it:4d}/{total_it}] [{bar_str}] {pct:5.1f}% | "
                f"{int(sps):,} sps | "
                f"Rew: {mean_rew:+.2f} (R:{reach:+.2f} L:{lift:+.2f} P:{place:+.2f}) | "
                f"Loss: {v_loss:.2e}{exploration} | "
                f"ETA: {eta_s}\n"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

    def close(self) -> None:
        """Safely print final newline."""
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass


# Global single-line bar instance
_GLOBAL_BAR: SingleLineTrainingBar | None = None


def get_training_bar() -> SingleLineTrainingBar:
    global _GLOBAL_BAR
    if _GLOBAL_BAR is None:
        _GLOBAL_BAR = SingleLineTrainingBar()
    return _GLOBAL_BAR


def setup_clean_terminal_ui() -> None:
    """Setup ultra-clean, single-line terminal progress bar by hooking RSL-RL logger."""
    os.environ["CARB_APP_LOG_LEVEL"] = "warn"
    os.environ["OMNI_LOG_LEVEL"] = "error"

    try:
        import rsl_rl.utils.logger as rsl_logger
        from rsl_rl.utils.logger import LogWriter

        def clean_single_line_log(
            self,
            it: int,
            start_it: int,
            total_it: int,
            collect_time: float,
            learn_time: float,
            loss_dict: dict,
            learning_rate: float,
            action_std: torch.Tensor,
            rnd_weight: float | None,
            print_minimal: bool = False,
            width: int = 80,
            pad: int = 40,
        ) -> None:
            # 1. Record metrics in TensorBoard / Logger writer
            if self.writer is not None:
                collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
                iteration_time = collect_time + learn_time
                self.tot_timesteps += collection_size
                self.tot_time += iteration_time

                # Extract extras
                parsed_extras = {}
                if self.ep_extras:
                    for key in {k for ep_info in self.ep_extras for k in ep_info}:
                        infotensor = torch.tensor([], device=self.device)
                        for ep_info in self.ep_extras:
                            if key not in ep_info:
                                continue
                            val = ep_info[key]
                            if not isinstance(val, torch.Tensor):
                                val = torch.Tensor([val])
                            if len(val.shape) == 0:
                                val = val.unsqueeze(0)
                            infotensor = torch.cat((infotensor, val.to(self.device)))
                        val_scalar = torch.mean(infotensor).item()
                        if "/" in key:
                            self.writer.add_scalar(key, val_scalar, it)
                        else:
                            self.writer.add_scalar("Episode/" + key, val_scalar, it)
                        parsed_extras[key] = val_scalar

                # Losses
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"Loss/{key}", value, it)
                self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
                self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

                # Performance
                fps = int(collection_size / max(collect_time + learn_time, 1e-6))
                self.writer.add_scalar("Perf/total_fps", fps, it)
                self.writer.add_scalar("Perf/collection_time", collect_time, it)
                self.writer.add_scalar("Perf/learning_time", learn_time, it)

                # Rewards
                mean_ep_rew = 0.0
                mean_ep_len = 0.0
                if len(self.rewbuffer) > 0:
                    mean_ep_rew = statistics.mean(self.rewbuffer)
                    mean_ep_len = statistics.mean(self.lenbuffer)
                    self.writer.add_scalar("Train/mean_reward", mean_ep_rew, it)
                    self.writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)

                # Timing
                elapsed_str = str(datetime.timedelta(seconds=int(self.tot_time)))
                done_it = it + 1 - start_it
                remaining_it = max(total_it - start_it - done_it, 0)
                eta_sec = int(self.tot_time / max(done_it, 1) * remaining_it)
                eta_str = str(datetime.timedelta(seconds=eta_sec))

                rewards_display = {
                    "Reach Cube": (
                        parsed_extras.get("Episode_Reward/reach_cube", 0.0)
                        + parsed_extras.get("Episode_Reward/reach_progress", 0.0)
                    ),
                    "Lift Cube": parsed_extras.get("Episode_Reward/lift_cube", 0.0),
                    "Place Cube": parsed_extras.get("Episode_Reward/place_cube", 0.0),
                    "Action Rate": parsed_extras.get("Episode_Reward/action_rate", 0.0),
                    "Mean Total": mean_ep_rew,
                }

                losses_display = {
                    "Value Loss": loss_dict.get("value", 0.0),
                    "Surrogate": loss_dict.get("surrogate", 0.0),
                    "Entropy": loss_dict.get("entropy", 0.0),
                    "Action Std": action_std.mean().item(),
                    "LR": learning_rate,
                }

                terms_display = {
                    "Mean Ep Len": f"{mean_ep_len:.1f} steps",
                    "Success": parsed_extras.get(
                        "Episode_Termination/lift_success", parsed_extras.get("Episode_Termination/placed", 0.0)
                    ),
                }

                # 2. Update Single-Line Progress Bar!
                bar = get_training_bar()
                bar.update(
                    it=it,
                    total_it=total_it,
                    sps=fps,
                    elapsed_s=elapsed_str,
                    eta_s=eta_str,
                    rewards=rewards_display,
                    losses=losses_display,
                    terms=terms_display,
                    device=str(self.device),
                    num_envs=self.num_envs,
                    log_dir=str(self.log_dir),
                )

                # 3. Save videos if any
                if isinstance(self.writer, LogWriter):
                    for video in pathlib.Path(self.log_dir).rglob("*.mp4"):
                        self.writer.save_video(video, it)

        # Hook RSL-RL
        rsl_logger.Logger.log = clean_single_line_log

    except Exception as e:
        print(f"[Warning] Could not initialize single-line terminal UI hook: {e}")
