import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from problems import generate_problems, make_prompt


def test_generate_problems_deterministic():
    a = generate_problems(50, seed=7)
    b = generate_problems(50, seed=7)
    assert [p.answer for p in a] == [p.answer for p in b]
    assert [p.prompt for p in a] == [p.prompt for p in b]


def test_generate_problems_different_seed_differs():
    a = generate_problems(50, seed=7)
    b = generate_problems(50, seed=8)
    assert [p.prompt for p in a] != [p.prompt for p in b]


def test_answers_are_nonnegative_integers():
    for p in generate_problems(200, seed=1):
        assert isinstance(p.answer, int)
        assert p.answer >= 0


def test_make_prompt_contains_noun_and_instructions():
    p = generate_problems(1, seed=3)[0]
    prompt = make_prompt(p, max_words=60)
    assert p.noun in prompt
    assert "Answer:" in prompt
    assert "60 words" in prompt
