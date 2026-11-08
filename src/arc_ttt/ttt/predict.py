"""Turn a (possibly test-time-trained) model into 2 guesses for a test input.

1. Generate one candidate per augmentation (batched greedy decoding), map each back.
2. Voting: rank candidates by how many augmentations produced them.
3. Product of experts (Franzen et al., 2025): rank candidates by the sum over
   augmentations of log p(candidate | augmented prompt), i.e. the product of
   per-augmentation probabilities.
"""
from __future__ import annotations

import json
import time
from collections import Counter

from ..augment import Augmentation
from ..data import Grid, grid_to_text, text_to_grid
from ..model_utils import build_example, generate_batch, target_logprob


def _key(g: Grid) -> str:
    return json.dumps(g)


def augmented_prompt(tok, demos: list[dict], test_input: Grid, aug: Augmentation, max_len: int):
    a_demos = [{"input": aug.apply(d["input"]), "output": aug.apply(d["output"])} for d in demos]
    ex = build_example(tok, a_demos, aug.apply(test_input), None, max_len)
    return None if ex is None else ex["input_ids"]


def generate_candidates(tok, model, demos, test_input, augs: list[Augmentation], max_len: int) -> list[Grid | None]:
    prompts, which = [], []
    for i, aug in enumerate(augs):
        p = augmented_prompt(tok, demos, test_input, aug, max_len)
        if p is not None:
            prompts.append(p); which.append(i)
    texts = generate_batch(tok, model, prompts)
    out: list[Grid | None] = [None] * len(augs)
    for i, text in zip(which, texts):
        g = text_to_grid(text)
        out[i] = augs[i].invert(g) if g is not None else None
    return out


def vote_top2(cands: list[Grid | None]) -> list[Grid]:
    """Most frequent valid candidates; ties keep first-seen order (identity augmentation first)."""
    counts = Counter(_key(c) for c in cands if c is not None)
    order = {}
    for c in cands:
        if c is not None:
            order.setdefault(_key(c), len(order))
    ranked = sorted(counts, key=lambda k: (-counts[k], order[k]))
    return [json.loads(k) for k in ranked[:2]]


def poe_scores(tok, model, demos, test_input, cands: list[Grid | None], augs: list[Augmentation],
               max_len: int, max_candidates: int = 8) -> dict[str, float]:
    """Sum over augmentations of log p(candidate | prompt) for the top-voted unique candidates."""
    counts = Counter(_key(c) for c in cands if c is not None)
    keys = [k for k, _ in counts.most_common(max_candidates)]
    prompts = [augmented_prompt(tok, demos, test_input, a, max_len) for a in augs]
    scores = {}
    for k in keys:
        g = json.loads(k)
        total = 0.0
        for a, p in zip(augs, prompts):
            if p is None:
                continue
            tgt = tok(grid_to_text(a.apply(g)), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            total += target_logprob(model, p, tgt)
        scores[k] = total
    return scores


def poe_top2(scores: dict[str, float]) -> list[Grid]:
    return [json.loads(k) for k in sorted(scores, key=lambda k: -scores[k])[:2]]


def predict(tok, model, demos, test_input, augs, max_len, poe=True, max_candidates=8) -> dict:
    t0 = time.time()
    cands = generate_candidates(tok, model, demos, test_input, augs, max_len)
    t_gen = time.time() - t0
    rec = {"vote": vote_top2(cands), "n_unique": len({_key(c) for c in cands if c is not None}),
           "n_valid": sum(c is not None for c in cands), "gen_seconds": round(t_gen, 2)}
    if poe:
        t1 = time.time()
        scores = poe_scores(tok, model, demos, test_input, cands, augs, max_len, max_candidates)
        rec["poe"] = poe_top2(scores)
        rec["poe_seconds"] = round(time.time() - t1, 2)
    return rec
