"""GRPO training against the checklist rubric reward (TRL GRPOTrainer, LoRA).

Usage:
  python train_grpo.py --out-dir runs/grpo_uniform --weights default
  python train_grpo.py --out-dir runs/grpo_downweighted --weights ../configs/downweighted.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from dataset_split import train_split
from model_io import BASE_MODEL, render_prompt
from problems import make_prompt
from rubric import DEFAULT_WEIGHTS, score_rubric

MAX_WORDS = 60


def build_dataset(tokenizer) -> Dataset:
    problems = train_split()
    rows = {
        "prompt": [render_prompt(tokenizer, make_prompt(p, MAX_WORDS)) for p in problems],
        "noun": [p.noun for p in problems],
        "answer": [p.answer for p in problems],
    }
    return Dataset.from_dict(rows)


def make_reward_func(weights: dict):
    def rubric_reward(prompts, completions, noun, answer, **kwargs):
        rewards = []
        for completion, n in zip(completions, noun):
            r = score_rubric(completion, n, MAX_WORDS, weights=weights)
            rewards.append(r.score / 100.0)
        return rewards

    rubric_reward.__name__ = "rubric_reward"
    return rubric_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default="default")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    args = ap.parse_args()

    weights = DEFAULT_WEIGHTS
    if args.weights != "default":
        weights = json.loads(Path(args.weights).read_text())

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    dataset = build_dataset(tokenizer)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    config = GRPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.num_generations * args.prompts_per_step,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=120,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="steps",
        save_steps=12,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        temperature=0.7,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func(weights)],
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    existing_checkpoints = sorted(Path(args.out_dir).glob("checkpoint-*")) if Path(args.out_dir).exists() else []
    resume = str(existing_checkpoints[-1]) if existing_checkpoints else None
    if resume:
        print(f"resuming from {resume}")
    train_result = trainer.train(resume_from_checkpoint=resume)

    adapter_dir = str(Path(args.out_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    log_path = Path(args.out_dir) / "train_log.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"adapter saved to {adapter_dir}")
    print(json.dumps(train_result.metrics, indent=2))


if __name__ == "__main__":
    main()
