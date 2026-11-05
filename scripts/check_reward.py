"""Check the pick reward for known reward hacks.

    python scripts/check_reward.py                 # ours vs Madhan's logic
    python scripts/check_reward.py --only ours     # just ours (exit 1 on any failure)
    python scripts/check_reward.py --terms ledge_park   # per-term breakdown

Plays scripted honest and exploit strategies in a toy world (see
vision_rl/rewards/audit) and checks: honest stages climb, every exploit loses
to the honest step it replaces, and closed trips earn nothing.
"""
import argparse
import sys

from vision_rl.rewards.audit.checks import check_all, run_episode
from vision_rl.rewards.audit.strategies import build_strategies
from vision_rl.rewards.audit.world import Camera
from vision_rl.rewards.pick import PickRewardConfig

CONFIGS = {"madhan": PickRewardConfig.madhan(), "ours": PickRewardConfig()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(CONFIGS))
    ap.add_argument("--terms", metavar="STRATEGY", help="print per-term totals for one strategy")
    ap.add_argument("--loops", type=int, default=20, help="random closed-trip trials")
    args = ap.parse_args()

    camera = Camera()
    names = [args.only] if args.only else list(CONFIGS)

    if args.terms:
        strat = {s.name: s for s in build_strategies()}[args.terms]
        for n in names:
            r = run_episode(CONFIGS[n], strat, camera)
            print(f"\n[{n}] {r.name}: total {r.total:.2f}, held {r.held} steps, success {r.successes} steps")
            for k, v in r.terms.items():
                if abs(v) > 1e-6:
                    print(f"   {k:11s} {v:+9.2f}")
        return

    results = {n: check_all(CONFIGS[n], camera, args.loops) for n in names}

    print("\nEpisode returns (250 steps)")
    print(f"{'strategy':22s}" + "".join(f"{n:>12s}" for n in names) + "   held/success steps")
    for s in build_strategies():
        row = "".join(f"{results[n][0][s.name].total:12.2f}" for n in names)
        r = results[names[-1]][0][s.name]
        print(f"{s.name:22s}{row}   {r.held:3d}/{r.successes:3d}")

    failed = False
    for n in names:
        verdicts = results[n][1]
        bad = [v for v in verdicts if not v.ok]
        print(f"\n=== {n}: {len(verdicts) - len(bad)}/{len(verdicts)} checks pass ===")
        for v in verdicts:
            print(f"  {'PASS' if v.ok else 'FAIL'}  {v.kind:7s} {v.name:30s} {v.detail}")
        failed |= n == "ours" and bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
