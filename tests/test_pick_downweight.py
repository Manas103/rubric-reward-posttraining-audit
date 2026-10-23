import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_picks_the_most_satisfied_criterion(tmp_path):
    baseline = {
        "criterion_satisfaction_pct": {
            "has_answer_marker": 99.0,
            "sentence_limit": 80.0,
            "mentions_noun": 100.0,
            "word_limit": 95.0,
        }
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    out_path = tmp_path / "downweighted.json"

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pick_downweight.py"),
         "--baseline-eval", str(baseline_path), "--out", str(out_path)],
        check=True, capture_output=True, text=True,
    )

    weights = json.loads(out_path.read_text())
    assert weights["mentions_noun"] == 0.05
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    other_weights = [v for k, v in weights.items() if k != "mentions_noun"]
    assert len(set(other_weights)) == 1  # the three remaining criteria split evenly
