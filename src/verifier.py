"""Deterministic correctness verifier. Shares no code with rubric.py.

Extracts the final number from a completion with its own independent regex
and parsing path, and compares it against the problem's known-correct
integer answer (computed at generation time in problems.py, not guessed).
This is the held-out check that rubric.py's four surface criteria cannot see.
"""
from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d+")


def extract_final_number(completion: str) -> int | None:
    lines = [l for l in completion.splitlines() if l.strip()]
    for line in reversed(lines):
        if "answer" in line.lower():
            nums = _NUMBER_RE.findall(line)
            if nums:
                return int(nums[-1])
    nums = _NUMBER_RE.findall(completion)
    return int(nums[-1]) if nums else None


def is_correct(completion: str, true_answer: int) -> bool:
    predicted = extract_final_number(completion)
    return predicted is not None and predicted == true_answer
