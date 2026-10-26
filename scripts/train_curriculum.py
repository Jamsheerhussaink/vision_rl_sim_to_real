#!/usr/bin/env python3
"""Train one cumulative curriculum lesson using the RGB simple-reward actor."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lesson', choices=('approach', 'grasp', 'lift', 'place'), required=True)
    args, remaining = parser.parse_known_args()
    from isaac_vision_rl.experiments import simple_rewards
    from isaac_vision_rl.experiments.simple_rewards.lesson_curriculum import activate
    simple_rewards.activate = lambda: activate(args.lesson)
    sys.argv = [str(Path(__file__).with_name('train_simple_rewards.py')), *remaining]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
