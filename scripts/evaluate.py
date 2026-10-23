"""Evaluates a model (base or LoRA-adapted) on one of the fixed problem splits.

Usage:
  python evaluate.py --split heldout_verifier --adapter none --out docs/eval_base_heldout.json
  python evaluate.py --split heldout_verifier --adapter runs/grpo_uniform/adapter --out docs/eval_uniform_heldout.json
  python evaluate.py --split no_regression_holdout --adapter none --out docs/eval_base_noregression.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_split import heldout_verifier_split, no_regression_holdout_split, train_split
from model_io import generate_batch, load_base_model, load_lora_model, load_tokenizer, render_prompt
from problems import make_prompt
from rubric import CRITERIA, DEFAULT_WEIGHTS, score_rubric
from verifier import is_correct

SPLITS = {
    "train": train_split,
    "heldout_verifier": heldout_verifier_split,
    "no_regression_holdout": no_regression_holdout_split,
}

MAX_WORDS = 60
MAX_NEW_TOKENS = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="default", help="'default' or path to json weights file")
    ap.add_argument("--limit", type=int, default=None, help="debug: evaluate only the first N items")
    ap.add_argument("--chunk-size", type=int, default=24,
                     help="max items generated per process invocation before exiting cleanly, "
                          "so a crash during a long run only loses one chunk, not the whole split")
    args = ap.parse_args()

    problems = SPLITS[args.split]()
    if args.limit:
        problems = problems[: args.limit]

    weights = DEFAULT_WEIGHTS
    if args.weights != "default":
        weights = json.loads(Path(args.weights).read_text())

    jsonl_path = Path(args.out).with_suffix(".items.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass  # a crash mid-write can truncate the last line; drop it and regenerate that item

    remaining = [p for p in problems if p.id not in done_ids]
    to_run = remaining[: args.chunk_size]

    if to_run:
        tok = load_tokenizer()
        model = load_base_model() if args.adapter == "none" else load_lora_model(args.adapter)
        rendered = [render_prompt(tok, make_prompt(p, MAX_WORDS)) for p in to_run]

        t0 = time.time()
        completions = generate_batch(model, tok, rendered, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = time.time() - t0

        with jsonl_path.open("a") as f:
            for problem, completion in zip(to_run, completions):
                r = score_rubric(completion, problem.noun, MAX_WORDS, weights=weights)
                correct = is_correct(completion, problem.answer)
                record = {
                    "id": problem.id,
                    "answer": problem.answer,
                    "completion": completion,
                    "rubric_score": r.score,
                    "satisfied": r.satisfied,
                    "verifier_correct": correct,
                }
                f.write(json.dumps(record) + "\n")
        print(f"chunk done: {len(to_run)} items in {elapsed:.1f}s "
              f"({len(done_ids) + len(to_run)}/{len(problems)} total)")

    still_remaining = len(problems) - (len(done_ids) + len(to_run))
    if still_remaining > 0:
        print(f"REMAINING {still_remaining} items; rerun this command to continue.")
        return

    all_records = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_records[rec["id"]] = rec
    per_item = [all_records[p.id] for p in problems]

    n = len(problems)
    criterion_hits = {c: sum(1 for it in per_item if it["satisfied"][c]) for c in CRITERIA}
    summary = {
        "split": args.split,
        "adapter": args.adapter,
        "weights_file": args.weights,
        "n": n,
        "mean_rubric_score": sum(it["rubric_score"] for it in per_item) / n,
        "verifier_accuracy_pct": 100.0 * sum(1 for it in per_item if it["verifier_correct"]) / n,
        "criterion_satisfaction_pct": {c: 100.0 * criterion_hits[c] / n for c in CRITERIA},
        "items": per_item,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"split={args.split} adapter={args.adapter} n={n} mean_rubric_score={summary['mean_rubric_score']:.2f} "
          f"verifier_accuracy_pct={summary['verifier_accuracy_pct']:.2f}")
    print("criterion_satisfaction_pct:", summary["criterion_satisfaction_pct"])


if __name__ == "__main__":
    main()
