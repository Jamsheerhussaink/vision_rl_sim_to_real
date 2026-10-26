"""W&B logger that records metrics/config but keeps artifacts local."""

import os
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.utils.wandb_log_writer import WandbLogWriter


class MetricsOnlyWandbLogWriter(WandbLogWriter):
    """Keep TensorBoard-compatible metrics in W&B without uploading artifacts."""

    def __init__(self, log_dir: str, project_name: str) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs=10)
        run_name = os.path.split(log_dir)[-1]
        entity = os.environ.get("WANDB_USERNAME", None)
        try:
            import wandb
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                config={"log_dir": log_dir},
            )
        except Exception as e:
            print(f"[Warning] Could not initialize W&B run: {e}")
            self.run = None
        self.logged_videos = set()

    def save_model(self, model_path: str, it: int) -> None:
        # Checkpoints remain in logs/rsl_rl for local evaluation and recovery.
        return None

    def save_file(self, path: str) -> None:
        # In particular, do not upload RSL-RL's generated git diff.
        return None

    def save_video(self, video, it: int) -> None:
        # Evaluation/training videos remain local unless separately shared.
        return None
