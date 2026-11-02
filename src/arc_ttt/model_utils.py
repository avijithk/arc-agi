"""Shared model helpers: loading, tokenizing ARC prompts, greedy grid prediction.

Used by both the baseline and the test-time-training code.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .augment import Augmentation
from .data import Grid, format_prompt, grid_to_text, text_to_grid

MAX_OUTPUT_TOKENS = 30 * 31 + 8  # 30x30 grid + newlines + slack


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_dtype(training: bool = False) -> torch.dtype:
    """Weight dtype.

    A10G/L4/A100: bf16. T4/P100 (no bf16): fp32 weights for training (with fp16
    autocast, see amp_dtype), fp16 for inference. CPU: fp32.
    """
    if bf16_ok():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float32 if training else torch.float16
    return torch.float32


def amp_dtype() -> torch.dtype | None:
    """Autocast dtype for training: fp16 on GPUs without bf16, else none."""
    if torch.cuda.is_available() and not bf16_ok():
        return torch.float16
    return None


def causal_lm_loss(model, input_ids: torch.Tensor, n_target: int) -> torch.Tensor:
    """Cross-entropy on the last n_target tokens only.

    Only the logits that predict the answer are computed (logits_to_keep), so an
    8k-token prompt doesn't materialize an 8k x 152k-vocab logit matrix.
    """
    logits = model(input_ids=input_ids, logits_to_keep=n_target + 1).logits[:, :-1].float()
    targets = input_ids[:, -n_target:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def load_model(name_or_path: str, dtype: torch.dtype | None = None, gradient_checkpointing: bool = False,
               training: bool = False):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype or pick_dtype(training),
        attn_implementation="sdpa",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if torch.cuda.is_available():
        model.to("cuda")
    return tok, model


def build_example(tok, demos: list[dict], query_in: Grid, query_out: Grid | None, max_len: int):
    """Tokenize one ARC example. Loss is only on the output grid (+ EOS).

    If the sequence is too long, demos are dropped from the front until it fits
    (keeping at least one). Returns None if it still doesn't fit.
    """
    demos = list(demos)
    while True:
        prompt_ids = tok(format_prompt(demos, query_in), add_special_tokens=False)["input_ids"]
        target_ids = []
        if query_out is not None:
            target_ids = tok(grid_to_text(query_out), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        budget = max_len - (len(target_ids) if query_out is not None else MAX_OUTPUT_TOKENS)
        if len(prompt_ids) <= budget:
            break
        if len(demos) <= 1:
            return None
        demos = demos[1:]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "n_demos": len(demos),
        "n_target": len(target_ids),
    }


@torch.no_grad()
def predict_grid(tok, model, demos: list[dict], test_input: Grid, aug: Augmentation, max_len: int) -> dict:
    """Greedy-decode an output grid under one augmentation, mapped back to the original frame."""
    a_demos = [{"input": aug.apply(d["input"]), "output": aug.apply(d["output"])} for d in demos]
    ex = build_example(tok, a_demos, aug.apply(test_input), None, max_len)
    if ex is None:
        return {"grid": None, "text": "", "seconds": 0.0, "skipped": "too_long"}
    ids = torch.tensor([ex["input_ids"]], device=model.device)
    t0 = time.time()
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=MAX_OUTPUT_TOKENS,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    grid = text_to_grid(text)
    return {
        "grid": aug.invert(grid) if grid is not None else None,
        "text": text,
        "seconds": time.time() - t0,
        "n_demos": ex["n_demos"],
    }
