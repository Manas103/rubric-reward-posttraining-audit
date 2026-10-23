"""Picks the rubric criterion to down-weight from a measured baseline eval,
rather than assuming one in advance: whichever criterion the untrained base
model already satisfies most often is the one a rubric-only reward signal
has the least room to teach, so down-weighting it should free up reward
mass for the criteria (and indirectly, correctness) still worth learning.

Usage:
  python pick_downweight.py --baseline-eval docs/eval_base_heldout.json --out configs/downweighted.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rubric import CRITERIA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--downweighted-share", type=float, default=0.05)
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline_eval).read_text())
    rates = baseline["criterion_satisfaction_pct"]
    most_satisfied = max(CRITERIA, key=lambda c: rates[c])

    remaining_share = 1.0 - args.downweighted_share
    others = [c for c in CRITERIA if c != most_satisfied]
    weights = {c: remaining_share / len(others) for c in others}
    weights[most_satisfied] = args.downweighted_share

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(weights, indent=2))
    print(f"most-satisfied baseline criterion: {most_satisfied} ({rates[most_satisfied]:.1f}%)")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
