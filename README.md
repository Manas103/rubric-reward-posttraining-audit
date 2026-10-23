# Rubric-Reward Post-Training with a Reward-Hacking Audit

GRPO fine-tunes Qwen2.5-1.5B-Instruct against a four-criterion checklist
rubric reward on a two-step arithmetic task, then audits the result with a
deterministic verifier that shares no code with the rubric and checks the
one thing the rubric cannot see: whether the final number is actually
correct. Every number below was measured on this machine, not targeted: the
rubric went up, and a held-out verifier shows task correctness went down,
not just lagged behind.

## Why this exists

A checklist rubric reward is gameable by construction: a model can learn to
satisfy format, length, and keyword criteria without the underlying answer
being right. This project builds the smallest honest version of that failure
mode and then tries one fix (down-weighting a criterion the policy already
satisfies at baseline) to see whether it actually helps.

## Honest framing

- **What this is not.** Not a general-purpose reasoning benchmark; the task
  is a procedurally generated two- and three-step arithmetic word problem
  with a known-correct integer answer, chosen specifically so a verifier can
  check correctness without any model-based judging.
- **Scale was cut from the resume's 1,200/400/500 prompts, and this is a
  compute and time budget decision for a single consumer GPU in one build
  session, not a measurement one.** 240 training prompts, 120 held-out
  verifier prompts, and a 150-prompt no-regression holdout were used, all
  disjoint by id. The mechanism measured (rubric score and verifier
  correctness diverging under pure rubric-only RL) does not depend on using
  more prompts; it would only tighten the confidence interval.
- **vLLM was not used.** The claimed stack names vLLM for fast generation;
  on this machine's single 12 GB RTX 4070 SUPER, running a vLLM generation
  server alongside LoRA training in the same process was not attempted after
  confirming TRL's default (Hugging Face `generate`) GRPO loop fit
  comfortably in the available VRAM. Training used the default generation
  backend; this is a disclosed deviation, not a hidden one.
- **The rubric's four criteria are surface-level and rule-based by design**
  (`src/rubric.py`): does the reply contain an `Answer: <number>` line, is
  the reasoning three sentences or fewer, does it mention the problem's
  named object, is the whole reply under the word budget. None of the four
  checks the arithmetic.
- **The verifier is a separate file with its own regex and parsing path**
  (`src/verifier.py`), checked by `tests/test_verifier.py::test_fluent_but_wrong_is_not_correct`
  to confirm it is not fooled by exactly the kind of fluent-but-wrong
  completion the rubric would score 100 on.
- **The down-weighted criterion was picked from a measurement, not assumed.**
  `scripts/pick_downweight.py` reads the baseline model's own per-criterion
  satisfaction rates and down-weights whichever criterion is already most
  satisfied (measured: `mentions_noun` at 100%), rather than the builder
  guessing which one that would be in advance.
- **Machine and toolchain, exactly as measured.** 8 physical / 16 logical
  cores (AMD Ryzen 7 7800X3D), Windows 11 Home (build 10.0.26200), WSL2
  Ubuntu 22.04 (capped at 12 cores / 12 GB RAM), Python 3.10, PyTorch
  2.6.0+cu124, Transformers 5.20.2, TRL 1.14.2 (`GRPOTrainer`), PEFT 0.13.2
  (LoRA rank 16), RTX 4070 SUPER (sm_89), bf16, model
  Qwen/Qwen2.5-1.5B-Instruct loaded directly to GPU (`device_map`).

## Architecture

```
src/
  problems.py        generates two- and three-step synthetic arithmetic word
                     problems with a known-correct integer answer
  rubric.py           the four-criterion checklist reward function, with
                      configurable per-criterion weights
  verifier.py          independent correctness check: extracts the final
                       number and compares it to the true answer, sharing no
                       code or regex with rubric.py
  dataset_split.py      fixed, disjoint id-range splits: 240 train / 120
                        held-out verifier / 150 no-regression holdout
  model_io.py           model loading (direct-to-GPU device_map, found by
                        trial to avoid a WSL2/CUDA instability triggered by
                        the default .to(device) path, see Findings) and
                        batched generation
scripts/
  evaluate.py            chunked, resumable evaluation (crash loses at most
                        one chunk of 20 items); writes per-item jsonl plus a
                        summary json
  train_grpo.py           GRPO training (TRL GRPOTrainer, LoRA), checkpointed
                         every 12 steps and resumable from the last checkpoint
  pick_downweight.py       picks the down-weight target from a measured
                         baseline eval instead of assuming one
tests/
  test_problems.py, test_rubric.py, test_verifier.py, test_dataset_split.py,
  test_pick_downweight.py
docs/
  eval_base_heldout.json, eval_base_noregression.json,
  eval_uniform_heldout.json, eval_uniform_noregression.json,
  eval_downweighted_heldout.json, eval_downweighted_noregression.json,
  train_log_uniform.json, train_log_downweighted.json,
  benchmark_output.txt, test_output.txt
configs/
  downweighted.json      the measured down-weight config used for run B
```

**Why rubric.py and verifier.py are separate files with no shared code.**
The entire audit depends on the rubric being unable to see what the verifier
checks; if they shared a parsing path, a bug in one would silently become a
bug in the other and the "reward hacking" finding would be an artifact of
the harness, not of the training run.

## Validation

**1. Test suite** (`docs/test_output.txt`):

```
...................
19 passed in 0.16s
```

**2. Benchmark runs** (`docs/benchmark_output.txt`, full per-item data in the
six `docs/eval_*.json` files):

```
base_heldout: n=120 mean_rubric_score=78.75 verifier_accuracy_pct=78.33
base_noreg: n=150 mean_rubric_score=79.83 verifier_accuracy_pct=74.00
uniform_heldout: n=120 mean_rubric_score=95.00 verifier_accuracy_pct=66.67
uniform_noreg: n=150 mean_rubric_score=93.33 verifier_accuracy_pct=67.33
downweighted_heldout: n=120 mean_rubric_score=93.93 verifier_accuracy_pct=69.17
downweighted_noreg: n=150 mean_rubric_score=94.51 verifier_accuracy_pct=62.00
```

## Findings

**GRPO against the rubric alone did not just fail to improve correctness, it
actively regressed it.** On the held-out verifier set, rubric score rose
16.25 points (78.75 to 95.00) while verifier-confirmed correctness *fell*
11.67 points (78.33% to 66.67%); on the separate no-regression holdout,
rubric rose 13.5 points while correctness fell 6.67 points. The resume's
framing assumed correctness would rise more slowly than the rubric; the
honest measured result is that optimizing the rubric alone made the model
worse at the actual task on both held-out sets. Looking at which criteria
moved explains why: `sentence_limit` jumped from 46.7% to 82.5% and
`word_limit` from 74.2% to 97.5%, meaning the policy learned to produce
much shorter, more clipped reasoning to satisfy the length and sentence-count
checks, and that shorter reasoning was measurably worse at getting the
arithmetic right.

**Down-weighting the already-satisfied criterion (`mentions_noun`, 100%
baseline) gave an inconsistent result across the two held-out sets, and
reporting both is the honest thing to do rather than quoting the one that
looks better.** On the held-out verifier set, the down-weighted run's
verifier accuracy was 69.17% against the uniform run's 66.67%, a recovery of
2.5 points of the gap, in the claimed direction. On the no-regression
holdout, the down-weighted run's verifier accuracy was 62.00% against the
uniform run's 67.33%, a further 5.33-point loss, in the opposite direction.
At 120 and 150 held-out items and a single training run per condition, this
is within the range of run-to-run noise rather than a reliably replicated
effect; a real claim of recovery would need multiple seeds per condition,
which this build's time budget did not include.

**A genuine WSL2/CUDA instability was found and routed around, not
suppressed.** The first few evaluation and training attempts intermittently
crashed the entire WSL2 VM (not just the Python process) with low-level I/O
errors, traced to two causes: loading the model with the default
`AutoModelForCausalLM.from_pretrained(...).to(device)` pattern (fixed by
loading directly to GPU with `device_map={"": device}` instead, see
`model_io.py`), and, separately, the host machine's C: drive filling to 0
bytes free from WSL2's dynamically-sized virtual disk, which corrupted the
running WSL session until the host drive had space freed and WSL was
restarted. Every GPU-facing script here is chunked and checkpointed
specifically because of this instability: `evaluate.py` resumes from its
last completed 20-item chunk, and `train_grpo.py` resumes from its last
checkpoint every 12 steps, so a crash during a 120-step run or a 120-item
eval loses minutes, not the whole run. One resume cycle also hit a
corrupted trailing line in a resume jsonl (a write truncated mid-crash);
both `evaluate.py` and the sibling project's equivalent scripts now skip an
unparseable line instead of crashing on it.

## Measured results

Machine: 8 physical / 16 logical cores (AMD Ryzen 7 7800X3D), Windows 11
Home (build 10.0.26200), WSL2 Ubuntu 22.04 (12 cores / 12 GB RAM), RTX 4070
SUPER, CUDA 12.4, PyTorch 2.6.0+cu124, TRL 1.14.2, Qwen2.5-1.5B-Instruct,
bf16, LoRA rank 16, 120 GRPO steps per run (2 prompts x 4 generations per
step, 1 epoch over 240 train prompts).

**Headline: GRPO against the rubric alone raised rubric score 16.25 points
but lowered verifier-confirmed correctness 11.67 points on held-out
problems (and 6.67 points on a separate no-regression holdout); down-weighting
the one criterion the model already satisfied at baseline recovered 2.5
points of that loss on the held-out verifier set but lost a further 5.33
points on the no-regression holdout, an inconsistent result reported as
measured.**

| Claim | Measured | Meets claim |
|---|---|---|
| 1.5B open-weight model | Qwen/Qwen2.5-1.5B-Instruct | yes |
| GRPO against checklist rubric rewards | TRL `GRPOTrainer`, 4-criterion rubric reward | yes |
| 1,200 open-ended prompts | 240 training prompts (budget-reduced; see Honest framing) | no, by design |
| held-out deterministic verifier sharing no code with the reward model | `verifier.py` independent of `rubric.py`, confirmed by test | yes |
| rubric score rose 21.4 points | +16.25 (held-out verifier set) | no, same direction, lower magnitude |
| verifier-confirmed correctness rose 7.1 | -11.67 (fell, not rose) | no, opposite direction |
| 400 held-out prompts | 120 (budget-reduced; see Honest framing) | no, by design |
| down-weighting criteria the policy already satisfied recovered 9.8 of that gap | +2.5 points recovered on held-out verifier set; -5.33 points (further loss) on no-regression holdout | no, inconsistent across the two held-out sets |
| no regression beyond 0.6 points on a 500-prompt holdout | -6.67 points (real regression, on a 150-prompt holdout) | no |

## Building and running

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

GRPO training and evaluation require the GPU-side venv (WSL2 Ubuntu 22.04,
packages in `requirements.txt` against CUDA 12.4 torch):

```
python scripts/evaluate.py --split heldout_verifier --adapter none --out docs/eval_base_heldout.json
python scripts/evaluate.py --split no_regression_holdout --adapter none --out docs/eval_base_noregression.json

python scripts/train_grpo.py --out-dir runs/grpo_uniform --weights default --max-steps 120
python scripts/evaluate.py --split heldout_verifier --adapter runs/grpo_uniform/adapter --out docs/eval_uniform_heldout.json
python scripts/evaluate.py --split no_regression_holdout --adapter runs/grpo_uniform/adapter --out docs/eval_uniform_noregression.json

python scripts/pick_downweight.py --baseline-eval docs/eval_base_heldout.json --out configs/downweighted.json
python scripts/train_grpo.py --out-dir runs/grpo_downweighted --weights configs/downweighted.json --max-steps 120
python scripts/evaluate.py --split heldout_verifier --adapter runs/grpo_downweighted/adapter --weights configs/downweighted.json --out docs/eval_downweighted_heldout.json
python scripts/evaluate.py --split no_regression_holdout --adapter runs/grpo_downweighted/adapter --weights configs/downweighted.json --out docs/eval_downweighted_noregression.json
```

`evaluate.py` and `train_grpo.py` are chunked/checkpointed and resumable:
rerun the same command if it stops partway (see Findings for why this
matters on this machine).

## Sibling comparison

This portfolio's `hybrid-human-loop-reasoning-data`
(https://github.com/Manas103/hybrid-human-loop-reasoning-data) was built in
the same session and shares this project's core instinct that a cheap
automated signal (there, generation confidence; here, a rubric score) can
diverge from ground truth, and that the gap is worth measuring directly
rather than assuming. That project measures a routing/review budget question
on top of the gap; this one measures what happens when an RL policy is
allowed to optimize the cheap signal directly.

## Limitations

- Scale is reduced from the resume's 1,200/400/500 prompts; see Honest
  framing.
- Single run per condition (uniform, down-weighted), no multiple seeds; the
  inconsistent down-weighting result across the two held-out sets may be
  noise rather than a real effect (see Findings).
- vLLM was not used for generation; training used TRL's default
  Hugging Face `generate` backend (see Honest framing).
- The task (two- and three-step synthetic arithmetic) is deliberately simple
  so a verifier can check it without a model judge; the reward-hacking
  effect measured here is specific to a rubric that rewards format and
  length over a task where format and correctness can be decoupled, and may
  not generalize to every rubric-reward setup.
- The down-weight target is picked automatically from one baseline eval's
  criterion satisfaction rates; a different baseline sample could pick a
  different criterion.
