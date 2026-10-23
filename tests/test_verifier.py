import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifier import extract_final_number, is_correct


def test_extracts_number_after_answer_label():
    assert extract_final_number("Some reasoning.\nAnswer: 42") == 42


def test_extracts_negative_number():
    assert extract_final_number("Answer: -7") == -7


def test_falls_back_to_last_number_if_no_answer_label():
    assert extract_final_number("I counted 3 then 5 then 9 total.") == 9


def test_no_number_returns_none():
    assert extract_final_number("I am not sure.") is None


def test_is_correct_true_and_false():
    assert is_correct("Answer: 12", 12) is True
    assert is_correct("Answer: 13", 12) is False


def test_fluent_but_wrong_is_not_correct():
    """Guards the exact reward-hacking case rubric.py cannot see."""
    completion = "This is about apples. It is short. It is on-topic.\nAnswer: 999999"
    assert is_correct(completion, 12) is False
