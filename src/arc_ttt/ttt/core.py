"""Per-puzzle test-time training (TTT) with checkpointed predictions and stopping signals.

For each puzzle:
  1. Optionally hold one demo pair out of training (holdout mode, needs >= 3 pairs).
  2. Build leave-one-out training examples from the remaining demo pairs: each pair
     in turn is the query, the others are the prompt's demos. Repeat under several
     augmentations (identity + random D8 symmetry with color permutation).
  3. Reset the LoRA adapter to the same initial weights, then train for up to
     `max_steps` steps at a constant learning rate.
  4. Every `signal_every` steps, log cheap stopping signals:
       loo_loss      mean loss on the un-augmented leave-one-out examples
       holdout_loss  loss on the held-out pair's output (holdout mode only)
  5. At each step in `predict_at` (e.g. 0, 8, 16, 32, 64), predict every test input
     (vote + product-of-experts top-2) and, in holdout mode, check whether the
     held-out pair is solved exactly.

One run therefore yields the whole fixed-budget sweep (one point per checkpoint),
and adaptive stopping rules can be evaluated offline from the logged signals.
"""
from __future__ import annotations

import random
import time

import torch

from ..augment import Augmentation, random_color_perm
from ..data import text_to_grid
from ..model_utils import amp_dtype, build_example, causal_lm_loss, generate_batch
from .predict import predict


def loo_examples(tok, pairs: list[dict], aug: Augmentation, max_len: int) -> list[dict]:
    pa = [{"input": aug.apply(p["input"]), "output": aug.apply(p["output"])} for p in pairs]
    out = []
    for i, q in enumerate(pa):
        ex = build_example(tok, pa[:i] + pa[i + 1:], q["input"], q["output"], max_len)
        if ex is not None:
            out.append(ex)
    return out


def task_augs(n: int, rng: random.Random) -> list[Augmentation]:
    """Identity first, then distinct D8 symmetries (cycled if n > 8), each with a random color permutation."""
    augs = [Augmentation()]
    for i in range(1, n):
        augs.append(Augmentation(d8=i % 8, colors=random_color_perm(rng)))
    return augs


class TestTimeTrainer:
    def __init__(self, tok, model, cfg: dict):
        self.tok, self.model, self.cfg = tok, model, cfg
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.init_state = [p.detach().clone() for p in self.params]
        self.amp = amp_dtype()

    def reset(self):
        with torch.no_grad():
            for p, p0 in zip(self.params, self.init_state):
                p.copy_(p0)

    @torch.no_grad()
    def _mean_loss(self, examples: list[dict]) -> float | None:
        if not examples:
            return None
        self.model.eval()
        tot = 0.0
        for ex in examples:
            ids = torch.tensor([ex["input_ids"]], device=self.model.device)
            with torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
                tot += causal_lm_loss(self.model, ids, ex["n_target"]).item()
        return tot / len(examples)

    def _train_step(self, opt, scaler, batch: list[dict]) -> float:
        self.model.train()
        tot = 0.0
        for ex in batch:
            ids = torch.tensor([ex["input_ids"]], device=self.model.device)
            with torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
                loss = causal_lm_loss(self.model, ids, ex["n_target"]) / len(batch)
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        return tot

    @torch.no_grad()
    def _solves_holdout(self, demos, ho, max_len) -> bool:
        self.model.eval()
        ex = build_example(self.tok, demos, ho["input"], None, max_len)
        if ex is None:
            return False
        text = generate_batch(self.tok, self.model, [ex["input_ids"]])[0]
        return text_to_grid(text) == ho["output"]

    def run_task(self, task: dict, seed: int) -> dict:
        c, tok = self.cfg, self.tok
        rng = random.Random(seed)
        max_len = c["max_seq_len"]
        pairs = list(task["train"])

        ho, ho_index = None, None
        if c.get("holdout") and len(pairs) >= 3:
            ho_index = rng.randrange(len(pairs))
            ho = pairs.pop(ho_index)

        train_augs = task_augs(c["n_train_augs"], rng)
        samples = [ex for a in train_augs for ex in loo_examples(tok, pairs, a, max_len)]
        rng.shuffle(samples)
        loo_identity = loo_examples(tok, pairs, Augmentation(), max_len)
        ho_example = build_example(tok, pairs, ho["input"], ho["output"], max_len) if ho else None
        pred_augs = task_augs(c["n_predict_augs"], rng)
        all_demos = task["train"]  # predictions always see every demo pair

        self.reset()
        opt = torch.optim.AdamW(self.params, lr=c["lr"], weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp is not None)
        bs = c.get("batch_size", 1)

        signals, checkpoints, train_s, losses = [], [], 0.0, []
        predict_at = set(c["predict_at"])
        max_steps = max(predict_at) if samples else 0
        for step in range(max_steps + 1):
            if step % c["signal_every"] == 0 or step in predict_at:
                t0 = time.time()
                signals.append({
                    "step": step,
                    "loo_loss": self._mean_loss(loo_identity),
                    "holdout_loss": self._mean_loss([ho_example] if ho_example else []),
                    "train_loss": sum(losses[-c["signal_every"]:]) / max(1, len(losses[-c["signal_every"]:])) if losses else None,
                    "train_seconds": round(train_s, 2),
                    "signal_seconds": round(time.time() - t0, 3),
                })
            if step in predict_at:
                self.model.eval()
                t0 = time.time()
                tests = [predict(tok, self.model, all_demos, t["input"], pred_augs, max_len,
                                 poe=c.get("poe", True), max_candidates=c.get("poe_max_candidates", 8))
                         for t in task["test"]]
                cp = {"step": step, "train_seconds": round(train_s, 2), "tests": tests,
                      "predict_seconds": round(time.time() - t0, 2)}
                if ho is not None:
                    t1 = time.time()
                    cp["holdout_exact"] = self._solves_holdout(pairs, ho, max_len)
                    cp["holdout_check_seconds"] = round(time.time() - t1, 2)
                checkpoints.append(cp)
            if step < max_steps:
                t0 = time.time()
                batch = [samples[(step * bs + j) % len(samples)] for j in range(bs)]
                losses.append(self._train_step(opt, scaler, batch))
                train_s += time.time() - t0

        return {
            "task": task["id"],
            "n_pairs": len(task["train"]),
            "holdout_index": ho_index,
            "n_train_samples": len(samples),
            "signals": signals,
            "checkpoints": checkpoints,
            "targets": [t.get("output") for t in task["test"]],
        }
