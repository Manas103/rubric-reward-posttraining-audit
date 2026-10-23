"""Fixed, disjoint problem splits by id range. No overlap between splits."""
from __future__ import annotations

from problems import Problem, generate_problems

SEED = 20260914

N_TRAIN = 240
N_HELDOUT_VERIFIER = 120
N_NO_REGRESSION_HOLDOUT = 150

TOTAL = N_TRAIN + N_HELDOUT_VERIFIER + N_NO_REGRESSION_HOLDOUT


def _all_problems() -> list[Problem]:
    return generate_problems(TOTAL, seed=SEED)


def train_split() -> list[Problem]:
    return _all_problems()[:N_TRAIN]


def heldout_verifier_split() -> list[Problem]:
    return _all_problems()[N_TRAIN : N_TRAIN + N_HELDOUT_VERIFIER]


def no_regression_holdout_split() -> list[Problem]:
    start = N_TRAIN + N_HELDOUT_VERIFIER
    return _all_problems()[start : start + N_NO_REGRESSION_HOLDOUT]
