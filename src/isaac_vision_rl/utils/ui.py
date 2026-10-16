# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live terminal training dashboard, with plain progress lines for log files."""

from __future__ import annotations

import atexit
import datetime
import os
import pathlib
import shutil
import statistics
import sys

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


class SingleLineTrainingBar:
    """Streaming progress bar logger that continuously outputs clean progress lines with newlines."""

    def __init__(self, task_name: str = "IsaacVisionRl-PickLift-SO101", print_new_line: bool = True):
        self.task_name = task_name
        self.print_new_line = print_new_line
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
        """Render streaming progress update line with newline."""
        term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        it = it + 1  # 1-indexed for display
        pct = (it / total_it) * 100.0 if total_it > 0 else 0.0

        mean_rew = rewards.get("Mean Total", 0.0)
        reach = rewards.get("Reach Cube", 0.0)
        lift = rewards.get("Lift Cube", 0.0)
        grasp = rewards.get("Grasp Cube", 0.0)
        v_loss = losses.get("Value Loss", 0.0)
        act_std = losses.get("Action Std", 0.0)
        succ = terms.get("Success", 0.0) * 100.0

        # Basic monochrome progress bar
        bar_width = max(min(term_width - 120, 30), 12)
        filled = int(bar_width * it / total_it) if total_it > 0 else 0
        bar_str = "=" * filled + (">" if filled < bar_width else "")
        bar_str = bar_str.ljust(bar_width, "-")

        prefix = f"[{it:4d}/{total_it}]"

        # Pure monochrome line with standard formatting
        line = (
            f"{prefix} [{bar_str}] {pct:5.1f}% | "
            f"{int(sps):4d} sps | "
            f"Rew: {mean_rew:+.2f} (R:{reach:+.2f} L:{lift:+.2f} G:{grasp:+.2f}) | "
            f"Loss: {v_loss:.2e} | "
            f"Std: {act_std:.2f} Succ: {succ:4.1f}% | "
            f"ETA: {eta_s}\n"
        )

        sys.stdout.write(line)
        sys.stdout.flush()

    def close(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass


class TrainingDashboard(SingleLineTrainingBar):
    """Render separate progress, reward, learning and timing rows in a terminal."""

    def __init__(self):
        self._live = None
        super().__init__()

    def update(
        self, it, total_it, sps, elapsed_s, eta_s, rewards, losses, terms,
        device="cuda:0", num_envs=16, log_dir="",
    ):
        if not self.is_tty:
            return super().update(
                it, total_it, sps, elapsed_s, eta_s, rewards, losses, terms,
                device, num_envs, log_dir,
            )
        completed = it + 1
        pct = 100 * completed / max(total_it, 1)
        progress = Text(
            f"Update {completed:,} / {total_it:,}  ({pct:.1f}%)   "
            f"{num_envs} environments   {device}", style="bold cyan",
        )
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="bold", ratio=1)
        table.add_column(ratio=4)
        table.add_row(
            "Rewards",
            f"Episode {rewards.get('Mean Total', 0):+.3f}   "
            f"Reach {rewards.get('Reach Cube', 0):+.4f}   "
            f"Grasp {rewards.get('Grasp Cube', 0):+.4f}   "
            f"Lift {rewards.get('Lift Cube', 0):+.4f}",
        )
        table.add_row(
            "Penalties",
            f"Drag {rewards.get('Drag', 0):+.4f}   "
            f"Displacement {rewards.get('Displacement', 0):+.4f}   "
            f"Action change {rewards.get('Action Rate', 0):+.4f}",
        )
        table.add_row(
            "Learning",
            f"Value loss {losses.get('Value Loss', 0):.3e}   "
            f"Policy loss {losses.get('Surrogate', 0):.3e}   "
            f"Learning rate {losses.get('LR', 0):.2e}",
        )
        table.add_row(
            "Exploration",
            f"Action std {losses.get('Action Std', 0):.3f}   "
            f"Entropy {losses.get('Entropy', 0):.3f}   "
            f"Entropy coefficient {losses.get('Entropy Coefficient', 0):.4f}",
        )
        table.add_row(
            "Training signal",
            f"Lift-success termination metric {terms.get('Success', 0):.4f}   "
            f"Mean episode length {terms.get('Mean Ep Len', 'n/a')}",
        )
        table.add_row(
            "Timing",
            f"{sps:,.0f} samples/s   Elapsed {elapsed_s}   Remaining {eta_s}",
        )
        progress_row = Table.grid(expand=True)
        progress_row.add_row(ProgressBar(total=max(total_it, 1), completed=completed))
        content = Group(
            progress, progress_row,
            table, Text(f"Checkpoints: {log_dir}", style="dim"),
            Text("Training metrics are not an independent success evaluation.", style="dim"),
        )
        panel = Panel(content, title="SO-101 PPO Training", border_style="cyan")
        if self._live is None:
            self._live = Live(panel, console=Console(), auto_refresh=False)
            self._live.start(refresh=True)
        else:
            self._live.update(panel, refresh=True)

    def close(self):
        if self._live is not None:
            self._live.stop()
            self._live = None
        else:
            super().close()


# Global dashboard instance
_GLOBAL_BAR: SingleLineTrainingBar | None = None


def get_training_bar() -> SingleLineTrainingBar:
    global _GLOBAL_BAR
    if _GLOBAL_BAR is None:
        _GLOBAL_BAR = SingleLineTrainingBar()
    return _GLOBAL_BAR


def setup_clean_terminal_ui() -> None:
    """Install the training dashboard while preserving TensorBoard metrics."""
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
                    "Grasp Cube": (
                        parsed_extras.get("Episode_Reward/grasp_cube", 0.0)
                        + parsed_extras.get("Episode_Reward/pinch_force", 0.0)
                    ),
                    "Drag": parsed_extras.get("Episode_Reward/drag_penalty", 0.0),
                    "Displacement": parsed_extras.get("Episode_Reward/displacement_penalty", 0.0),
                    "Place Cube": parsed_extras.get("Episode_Reward/place_cube", 0.0),
                    "Action Rate": parsed_extras.get("Episode_Reward/action_rate", 0.0),
                    "Camera Boundary": parsed_extras.get("Episode_Reward/camera_boundary", 0.0),
                    "Premature Close": parsed_extras.get("Episode_Reward/premature_close", 0.0),
                    "Mean Total": mean_ep_rew,
                }

                losses_display = {
                    "Value Loss": loss_dict.get("value", 0.0),
                    "Surrogate": loss_dict.get("surrogate", 0.0),
                    "Entropy": loss_dict.get("entropy", 0.0),
                    "Entropy Coefficient": loss_dict.get("entropy_coefficient", 0.0),
                    "Action Std": action_std.mean().item(),
                    "LR": learning_rate,
                }

                terms_display = {
                    "Mean Ep Len": f"{mean_ep_len:.1f} steps",
                    "Success": (
                        parsed_extras.get("Episode_Termination/lift_success", 0.0)
                        + parsed_extras.get("Episode_Termination/placed", 0.0)
                    ),
                    "Out of View": (
                        parsed_extras.get("Episode_Termination/gripper_out_of_view", 0.0)
                        + parsed_extras.get("Episode_Termination/cube_out_of_view", 0.0)
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
