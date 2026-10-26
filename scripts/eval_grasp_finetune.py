#!/usr/bin/env python3
"""Evaluate a checkpoint with the isolated grasp-finetune reward/stage logic."""

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--checkpoint", required=True)
args, _ = parser.parse_known_args()

import torch

metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
schema = metadata.get("so101_agent_schema")
if schema not in ("four_stage_v1", "four_stage_v2_bounded"):
    raise ValueError(f"Unsupported checkpoint schema for this evaluator: {schema!r}")

os.environ["SO101_REWARD_PROFILE"] = "four-stage"
os.environ["SO101_REWARD_STYLE"] = "continuous"
os.environ["SO101_GRASP_REWARD_MODE"] = "overlap" if schema == "four_stage_v1" else "bounded"
os.environ["SO101_TRAINING_GOAL"] = metadata.get("so101_training_goal", "place")
os.environ["SO101_RGB_FPS"] = str(metadata.get("so101_control_hz", 30))

from isaac_vision_rl.experiments.grasp_finetune import install

install()
runpy.run_path(str(ROOT / "scripts" / "eval_pipeline.py"), run_name="__main__")
