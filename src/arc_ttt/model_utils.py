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
def predict_grid(tok, model, demos: list[dict], test_input: Grid, aug: Augmentation, max_len: int,
                 constrained: bool = False) -> dict:
    """Greedy-decode an output grid under one augmentation, mapped back to the original frame."""
    a_demos = [{"input": aug.apply(d["input"]), "output": aug.apply(d["output"])} for d in demos]
    ex = build_example(tok, a_demos, aug.apply(test_input), None, max_len)
    if ex is None:
        return {"grid": None, "text": "", "seconds": 0.0, "skipped": "too_long"}
    t0 = time.time()
    text = generate_batch(tok, model, [ex["input_ids"]], constrained=constrained)[0]
    grid = text_to_grid(text)
    return {
        "grid": aug.invert(grid) if grid is not None else None,
        "text": text,
        "seconds": time.time() - t0,
        "n_demos": ex["n_demos"],
    }


class GridConstraint:
    """Logits processor that only allows well-formed grids.

    Allowed tokens: the 10 digits, newline, and end-of-sequence. The first row may be
    1-30 cells long; every later row must match it exactly; at most 30 rows. A row can
    only end (newline or EOS) when it is full, so every output parses as a grid.
    Requires a tokenizer where each digit and the newline are single tokens.
    """

    def __init__(self, tok, prompt_len: int, max_side: int = 30):
        ids = [tok(str(d), add_special_tokens=False)["input_ids"] for d in range(10)]
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        if any(len(i) != 1 for i in ids) or len(nl) != 1:
            raise ValueError("tokenizer does not encode digits/newline as single tokens")
        self.digits = [i[0] for i in ids]
        self.nl, self.eos = nl[0], tok.eos_token_id
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.prompt_len, self.max_side = prompt_len, max_side

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float("-inf"))
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:].tolist()
            if gen and gen[-1] in (self.eos, self.pad):   # finished: keep emitting EOS (padding)
                mask[b, self.eos] = 0.0
                continue
            width, rows, cur = None, 0, 0
            for t in gen:
                if t == self.nl:
                    width = cur if width is None else width
                    rows += 1
                    cur = 0
                else:
                    cur += 1
            limit = width if width is not None else self.max_side
            if cur < limit:
                mask[b, self.digits] = 0.0
            if cur > 0 and (width is None or cur == width):
                mask[b, self.eos] = 0.0
                if rows + 1 < self.max_side:
                    mask[b, self.nl] = 0.0
        return scores + mask


@torch.no_grad()
def generate_batch(tok, model, prompts: list[list[int]], max_new_tokens: int = MAX_OUTPUT_TOKENS,
                   constrained: bool = False) -> list[str]:
    """Greedy-decode several prompts at once (left-padded). Returns the generated text for each.

    constrained=True restricts decoding to well-formed grids (see GridConstraint).
    """
    if not prompts:
        return []
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        mask[i, L - len(p):] = 1
    extra = {}
    if constrained:
        from transformers import LogitsProcessorList
        extra["logits_processor"] = LogitsProcessorList([GridConstraint(tok, L)])
    out = model.generate(
        ids.to(model.device), attention_mask=mask.to(model.device), max_new_tokens=max_new_tokens,
        do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, **extra,
    )
    return [tok.decode(o[L:], skip_special_tokens=True) for o in out]


@torch.no_grad()
def target_logprob(model, prompt_ids: list[int], target_ids: list[int]) -> float:
    """Sum of log p(target tokens | prompt) under the model."""
    ids = torch.tensor([prompt_ids + target_ids], device=model.device)
    n = len(target_ids)
    logits = model(input_ids=ids, logits_to_keep=n + 1).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(target_ids, device=model.device)
    return lp.gather(1, tgt[:, None]).sum().item()
