"""Turn a (possibly test-time-trained) model into 2 guesses for a test input.

1. Generate one candidate per augmentation (batched greedy decoding, optionally
   constrained to well-formed grids), map each back to the original frame.
2. Optional size filter: drop candidates whose shape breaks the size pattern of the
   demonstrations (kept only if at least one candidate passes).
3. Voting: rank candidates by how many augmentations produced them.
4. Product of experts (Franzen et al., 2025): rank candidates by the sum over
   augmentations of log p(candidate | augmented prompt).

Adaptive inference: generate candidates in batches of `batch` augmentations and stop
once the top candidate has at least `min_votes` votes (or `max_augs` is reached).

Everything needed to replay other settings offline is logged: every candidate in
generation order, generation time per batch, the size-filter verdict per candidate,
and the per-augmentation PoE log-probabilities.
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


def _shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


# ---------------------------------------------------------------- size rule

def size_rule(demos: list[dict]):
    """Infer how the output size depends on the input size from the demos.

    Returns a function test_input -> (h, w), or None when no simple rule fits.
    Rules tried: same size as input; one fixed size; integer scale up; integer scale down.
    """
    pairs = [(_shape(d["input"]), _shape(d["output"])) for d in demos]
    if all(i == o for i, o in pairs):
        return lambda g: _shape(g)
    outs = {o for _, o in pairs}
    if len(outs) == 1:
        const = next(iter(outs))
        return lambda g: const
    if all(i[0] and i[1] and o[0] % i[0] == 0 and o[1] % i[1] == 0 for i, o in pairs):
        ks = {(o[0] // i[0], o[1] // i[1]) for i, o in pairs}
        if len(ks) == 1:
            a, b = next(iter(ks))
            return lambda g: (_shape(g)[0] * a, _shape(g)[1] * b)
    if all(o[0] and o[1] and i[0] % o[0] == 0 and i[1] % o[1] == 0 for i, o in pairs):
        ks = {(i[0] // o[0], i[1] // o[1]) for i, o in pairs}
        if len(ks) == 1:
            a, b = next(iter(ks))
            return lambda g: (_shape(g)[0] // a, _shape(g)[1] // b)
    return None


# ---------------------------------------------------------------- generation

def augmented_prompt(tok, demos: list[dict], test_input: Grid, aug: Augmentation, max_len: int):
    a_demos = [{"input": aug.apply(d["input"]), "output": aug.apply(d["output"])} for d in demos]
    ex = build_example(tok, a_demos, aug.apply(test_input), None, max_len)
    return None if ex is None else ex["input_ids"]


def generate_candidates(tok, model, demos, test_input, augs: list[Augmentation], max_len: int,
                        constrained: bool = False, temperature: float = 0.0) -> list[Grid | None]:
    prompts, which = [], []
    for i, aug in enumerate(augs):
        p = augmented_prompt(tok, demos, test_input, aug, max_len)
        if p is not None:
            prompts.append(p); which.append(i)
    texts = generate_batch(tok, model, prompts, constrained=constrained, temperature=temperature)
    out: list[Grid | None] = [None] * len(augs)
    for i, text in zip(which, texts):
        g = text_to_grid(text)
        out[i] = augs[i].invert(g) if g is not None else None
    return out


# ---------------------------------------------------------------- ranking

def vote_top2(cands: list[Grid | None]) -> list[Grid]:
    """Most frequent valid candidates; ties keep first-seen order (identity augmentation first)."""
    counts = Counter(_key(c) for c in cands if c is not None)
    order = {}
    for c in cands:
        if c is not None:
            order.setdefault(_key(c), len(order))
    ranked = sorted(counts, key=lambda k: (-counts[k], order[k]))
    return [json.loads(k) for k in ranked[:2]]


def poe_top2(scores: dict[str, float]) -> list[Grid]:
    return [json.loads(k) for k in sorted(scores, key=lambda k: -scores[k])[:2]]


def poe_matrix(tok, model, demos, test_input, keys: list[str], augs: list[Augmentation], max_len: int):
    """log p(candidate | prompt under aug) for every candidate key x augmentation (None if prompt too long)."""
    prompts = [augmented_prompt(tok, demos, test_input, a, max_len) for a in augs]
    mat = {}
    for k in keys:
        g = json.loads(k)
        row = []
        for a, p in zip(augs, prompts):
            if p is None:
                row.append(None)
                continue
            tgt = tok(grid_to_text(a.apply(g)), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            row.append(target_logprob(model, p, tgt))
        mat[k] = row
    return mat


def poe_scores(tok, model, demos, test_input, cands: list[Grid | None], augs: list[Augmentation],
               max_len: int, max_candidates: int = 8) -> dict[str, float]:
    """Sum over augmentations of log p(candidate | prompt) for the top-voted unique candidates."""
    counts = Counter(_key(c) for c in cands if c is not None)
    keys = [k for k, _ in counts.most_common(max_candidates)]
    mat = poe_matrix(tok, model, demos, test_input, keys, augs, max_len)
    return {k: sum(v for v in row if v is not None) for k, row in mat.items()}


def rank_from_log(rec: dict, n_augs: int | None = None, size_filter: bool = False):
    """Recompute vote and PoE top-2 from a logged prediction, using only the first n_augs
    augmentations and optionally the size filter. Used for offline replay."""
    uniq = rec["unique"]
    order = rec["aug_cand"][: n_augs if n_augs is not None else len(rec["aug_cand"])]
    ok = rec.get("size_ok") or [True] * len(uniq)
    keep = set(range(len(uniq)))
    if size_filter and any(ok[i] for i in set(order) if i >= 0):
        keep = {i for i in keep if ok[i]}
    used = [i for i in order if i >= 0 and i in keep]
    counts = Counter(used)
    first = {}
    for i in used:
        first.setdefault(i, len(first))
    vote = [uniq[i] for i in sorted(counts, key=lambda i: (-counts[i], first[i]))[:2]]
    poe = []
    if rec.get("poe_lp"):
        m = len(order)
        sc = {int(i): sum(v for v in row[:m] if v is not None)
              for i, row in rec["poe_lp"].items() if int(i) in counts}
        poe = [uniq[i] for i in sorted(sc, key=lambda i: -sc[i])[:2]]
    return vote, poe


# ---------------------------------------------------------------- main entry

def predict(tok, model, demos, test_input, augs, max_len, poe=True, max_candidates=8,
            constrained=False, size_filter=False, adaptive=None, n_samples=0, temperature=0.7) -> dict:
    """adaptive = {"batch": 2, "min_votes": 3} enables early stopping on agreement;
    None generates all augmentations in one batch."""
    rule = size_rule(demos)   # verdicts are always logged; size_filter decides whether they are applied
    expected = rule(test_input) if rule else None

    t0 = time.time()
    cands: list[Grid | None] = []
    batch_seconds, n_used = [], 0
    step = adaptive["batch"] if adaptive else len(augs)
    while n_used < len(augs):
        b0 = time.time()
        chunk = augs[n_used:n_used + step]
        cands += generate_candidates(tok, model, demos, test_input, chunk, max_len, constrained)
        n_used += len(chunk)
        batch_seconds.append(round(time.time() - b0, 3))
        if adaptive:
            pool = [c for c in cands if c is not None and (not size_filter or expected is None or _shape(c) == expected)]
            if not pool:
                pool = [c for c in cands if c is not None]
            if pool and Counter(_key(c) for c in pool).most_common(1)[0][1] >= adaptive["min_votes"]:
                break
    used_augs = augs[:n_used]
    if n_samples:   # optional: extra sampled candidates (one per augmentation, cycling) for more diversity
        extra = [used_augs[i % len(used_augs)] for i in range(n_samples)]
        cands += generate_candidates(tok, model, demos, test_input, extra, max_len, constrained, temperature)
        used_augs = used_augs + extra
        batch_seconds.append(round(time.time() - t0 - sum(batch_seconds), 3))
        n_used = len(used_augs)
    t_gen = time.time() - t0

    uniq, index, aug_cand = [], {}, []
    for c in cands:
        if c is None:
            aug_cand.append(-1)
            continue
        k = _key(c)
        if k not in index:
            index[k] = len(uniq); uniq.append(c)
        aug_cand.append(index[k])
    size_ok = [expected is None or _shape(g) == expected for g in uniq]

    rec = {"unique": uniq, "aug_cand": aug_cand, "size_ok": size_ok, "expected_shape": expected,
           "batch_seconds": batch_seconds, "n_augs_used": n_used, "gen_seconds": round(t_gen, 2),
           "n_unique": len(uniq), "n_valid": sum(i >= 0 for i in aug_cand)}

    if poe and uniq:
        t1 = time.time()
        counts = Counter(i for i in aug_cand if i >= 0)
        top_all = [i for i, _ in counts.most_common(max_candidates)]
        top_ok = [i for i, _ in counts.most_common() if size_ok[i]][:max_candidates]
        keys_i = list(dict.fromkeys(top_all + top_ok))
        mat = poe_matrix(tok, model, demos, test_input, [_key(uniq[i]) for i in keys_i], used_augs, max_len)
        rec["poe_lp"] = {str(i): mat[_key(uniq[i])] for i in keys_i}
        rec["poe_seconds"] = round(time.time() - t1, 2)

    rec["vote"], rec["poe"] = rank_from_log(rec, size_filter=size_filter)
    if not poe:
        rec.pop("poe")
    return rec
