"""Shared model loading and generation helpers for the GRPO run and its evals."""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    return model


def load_lora_model(adapter_path: str, device: str = "cuda"):
    base = load_base_model(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def render_prompt(tokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 120, batch_size: int = 6) -> list[str]:
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(len(batch)):
            new_tokens = gen[i][enc["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        del enc, gen
        torch.cuda.empty_cache()
    return outputs
