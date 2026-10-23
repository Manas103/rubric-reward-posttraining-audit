import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_split import heldout_verifier_split, no_regression_holdout_split, train_split


def test_splits_are_disjoint():
    train_ids = {p.id for p in train_split()}
    verify_ids = {p.id for p in heldout_verifier_split()}
    holdout_ids = {p.id for p in no_regression_holdout_split()}
    assert not (train_ids & verify_ids)
    assert not (train_ids & holdout_ids)
    assert not (verify_ids & holdout_ids)


def test_split_sizes():
    assert len(train_split()) == 240
    assert len(heldout_verifier_split()) == 120
    assert len(no_regression_holdout_split()) == 150
