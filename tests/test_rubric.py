import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rubric import score_rubric


def test_all_criteria_satisfied():
    completion = "Priya starts with some apples and ends with more.\nAnswer: 12"
    r = score_rubric(completion, noun="apples", max_words=60)
    assert r.satisfied == {
        "has_answer_marker": True,
        "sentence_limit": True,
        "mentions_noun": True,
        "word_limit": True,
    }
    assert r.score == 100.0


def test_missing_answer_marker_scores_lower():
    completion = "Priya has some apples left over at the end of the day."
    r = score_rubric(completion, noun="apples", max_words=60)
    assert r.satisfied["has_answer_marker"] is False
    assert r.score < 100.0


def test_rubric_is_gameable_by_format_alone():
    """A fluent, well-formatted reply that is numerically wrong still scores 100."""
    completion = "This is about apples. It is short. It is on-topic.\nAnswer: 999999"
    r = score_rubric(completion, noun="apples", max_words=60)
    assert r.score == 100.0


def test_sentence_limit_counts_only_reasoning_before_answer():
    completion = "One. Two. Three. Four sentences here.\nAnswer: 5"
    r = score_rubric(completion, noun="x", max_words=60)
    assert r.satisfied["sentence_limit"] is False


def test_word_limit_enforced():
    completion = ("word " * 100) + "apples\nAnswer: 1"
    r = score_rubric(completion, noun="apples", max_words=60)
    assert r.satisfied["word_limit"] is False


def test_custom_weights_rescore():
    completion = "No marker here, mentions apples though."
    weights = {"has_answer_marker": 0.7, "sentence_limit": 0.1, "mentions_noun": 0.1, "word_limit": 0.1}
    r = score_rubric(completion, noun="apples", max_words=60, weights=weights)
    assert abs(r.score - 30.0) < 1e-9
