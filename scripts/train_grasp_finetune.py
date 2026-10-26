#!/usr/bin/env python3
"""Fine-tune the old RGB actor with a gated, signed jaw-closing reward."""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Import the selected reward mode before train.py initializes its environment.
os.environ.setdefault("SO101_GRASP_REWARD_MODE", "overlap")
os.environ.setdefault("SO101_REWARD_STYLE", "continuous")
from isaac_vision_rl.experiments.grasp_finetune import install

install()
runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")
