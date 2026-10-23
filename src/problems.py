"""Synthetic arithmetic word-problem generator with a known-correct integer answer.

Every problem is generated from a fixed seed, so the ground-truth answer is
computed directly from the template parameters, not inferred. This is the thing
project 3 and this project both lean on: a ground truth that comes from how the
item was built, not from a second model's guess.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

NOUNS = [
    "apples", "marbles", "stickers", "pencils", "coins", "books",
    "stamps", "cards", "cookies", "balloons", "tickets", "beads",
]

TEMPLATES = [
    "add_give",   # start + gain - loss
    "multiply",   # price * quantity (+ flat add)
    "share",      # total split evenly, then some given away
]


@dataclass
class Problem:
    id: int
    prompt: str
    noun: str
    answer: int
    template: str


def _add_give(rng: random.Random, noun: str) -> tuple[str, int]:
    start = rng.randint(40, 220)
    gain = rng.randint(15, 90)
    loss = rng.randint(10, min(70, start + gain - 1))
    triple = rng.randint(2, 6)
    text = (
        f"Priya has {start} {noun}. A friend gives her {gain} more {noun}. "
        f"Then Priya gives away {loss} {noun}. Finally she splits what remains evenly "
        f"among herself and {triple - 1} friends. How many {noun} does Priya end up with?"
    )
    remaining = start + gain - loss
    return text, remaining // triple


def _multiply(rng: random.Random, noun: str) -> tuple[str, int]:
    price = rng.randint(12, 45)
    qty = rng.randint(8, 30)
    flat = rng.randint(5, 40)
    discard = rng.randint(3, 25)
    text = (
        f"Each bag holds {price} {noun}. Dev buys {qty} bags and also finds "
        f"{flat} loose {noun}, but {discard} of the {noun} turn out to be damaged and are "
        f"thrown out. How many {noun} does Dev have in total?"
    )
    return text, price * qty + flat - discard


def _share(rng: random.Random, noun: str) -> tuple[str, int]:
    per_person = rng.randint(20, 70)
    people = rng.randint(3, 8)
    given_away = rng.randint(5, per_person - 1)
    bonus = rng.randint(5, 30)
    total = per_person * people
    text = (
        f"A box of {total} {noun} is split evenly among {people} people. "
        f"One of them then gives away {given_away} {noun} and later receives a bonus of "
        f"{bonus} more {noun}. How many {noun} does that person end up with?"
    )
    return text, per_person - given_away + bonus


_BUILDERS = {"add_give": _add_give, "multiply": _multiply, "share": _share}


def generate_problems(n: int, seed: int) -> list[Problem]:
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        noun = rng.choice(NOUNS)
        text, answer = _BUILDERS[template](rng, noun)
        problems.append(Problem(id=i, prompt=text, noun=noun, answer=answer, template=template))
    return problems


def make_prompt(problem: Problem, max_words: int) -> str:
    return (
        f"{problem.prompt}\n\n"
        f"Reply in at most 3 sentences of reasoning, mention \"{problem.noun}\" at least "
        f"once in your reasoning, keep the whole reply under {max_words} words, and end "
        f"with a line of the exact form \"Answer: <number>\"."
    )
