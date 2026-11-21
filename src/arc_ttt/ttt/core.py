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
  6. Optional weight averaging (`weight_avg: {kind: swa, start: 32}` or
     `{kind: ema, decay: 0.9, start: 0}`, optional `at: [64]` to limit the averaged
     predictions to those steps): keep a running average of the TTT adapter's
     weights from step `start` on, and at every checkpoint after `start` also predict with
     the averaged weights (logged under checkpoint["avg"], same format as the normal
     predictions). The normal predictions come from the same training run, so the two are
     a paired comparison.

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
        text = generate_batch(self.tok, self.model, [ex["input_ids"]],
                              constrained=self.cfg.get("constrained_decoding", False))[0]
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
        warm = c.get("warmup_steps", 0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp is not None)
        bs = c.get("batch_size", 1)

        wa = c.get("weight_avg")
        self._avg, self._n_avg = None, 0
        signals, checkpoints, train_s, losses = [], [], 0.0, []
        rule = self.parse_rule(c.get("stop_rule"))
        stopped_at = None
        predict_at = set(c["predict_at"])
        max_steps = max(predict_at) if samples else 0
        for step in range(max_steps + 1):
            if wa and step >= wa.get("start", 0):
                self._update_avg(wa)
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
            if step in predict_at and rule is not None:
                # Live adaptive mode: decide at each checkpoint; predict only once, at the stop.
                if rule["kind"] == "best":
                    improved = self._track_best(rule, signals[-1], step)
                    stop, use_best = (not improved) or step >= max_steps, True
                else:
                    stop, use_best = self._decide(rule, signals, step, max_steps)
                if not stop:
                    if step < max_steps:
                        t0 = time.time()
                        batch = [samples[(step * bs + j) % len(samples)] for j in range(bs)]
                        losses.append(self._train_step(opt, scaler, batch))
                        sched.step()
                        train_s += time.time() - t0
                    continue
                pred_step = step
                if use_best and rule.get("best_state") is not None:
                    with torch.no_grad():
                        for p_, b_ in zip(self.params, rule["best_state"]):
                            p_.copy_(b_)
                    pred_step = rule["best_step"]
                stopped_at = step
                checkpoints.append(self._predict_cp(task, all_demos, pred_augs, max_len, pred_step, train_s, ho, pairs))
                break
            if step in predict_at:
                checkpoints.append(self._predict_cp(task, all_demos, pred_augs, max_len, step, train_s, ho, pairs))
                if wa and self._n_avg >= 2 and step in set(wa.get("at", predict_at)):
                    live = self._load_avg()
                    avg_cp = self._predict_cp(task, all_demos, pred_augs, max_len, step, train_s, ho, pairs)
                    self._restore(live)
                    avg_cp["n_avg"] = self._n_avg
                    checkpoints[-1]["avg"] = avg_cp
            if step < max_steps:
                t0 = time.time()
                batch = [samples[(step * bs + j) % len(samples)] for j in range(bs)]
                losses.append(self._train_step(opt, scaler, batch))
                sched.step()
                train_s += time.time() - t0

        rec = {
            "task": task["id"],
            "n_pairs": len(task["train"]),
            "holdout_index": ho_index,
            "n_train_samples": len(samples),
            "signals": signals,
            "checkpoints": checkpoints,
            "targets": [t.get("output") for t in task["test"]],
        }
        if rule is not None:
            rec["stop_rule"] = c["stop_rule"]
            rec["stopped_at"] = stopped_at
            rec["predicted_at"] = checkpoints[-1]["step"]
        return rec

    def _predict_cp(self, task, all_demos, pred_augs, max_len, step, train_s, ho, pairs):
        c, tok = self.cfg, self.tok
        self.model.eval()
        t0 = time.time()
        tests = [predict(tok, self.model, all_demos, t["input"], pred_augs, max_len,
                                 poe=c.get("poe", True), max_candidates=c.get("poe_max_candidates", 8),
                                 constrained=c.get("constrained_decoding", False),
                                 size_filter=c.get("size_filter", False),
                                 adaptive=c.get("adaptive_infer"),
                         n_samples=c.get("n_sampled_candidates", 0),
                         temperature=c.get("sample_temperature", 0.7))
                         for t in task["test"]]
        cp = {"step": step, "train_seconds": round(train_s, 2), "tests": tests,
              "predict_seconds": round(time.time() - t0, 2)}
        if ho is not None:
            t1 = time.time()
            cp["holdout_exact"] = self._solves_holdout(pairs, ho, max_len)
            cp["holdout_check_seconds"] = round(time.time() - t1, 2)
        return cp

    # ---------------------------------------------------------- weight averaging
    @torch.no_grad()
    def _update_avg(self, wa):
        """Fold the current adapter weights into the running average (uniform for swa,
        exponential for ema)."""
        if self._avg is None:
            self._avg = [p.detach().float().clone() for p in self.params]
            self._n_avg = 1
            return
        self._n_avg += 1
        if wa.get("kind", "swa") == "ema":
            d = wa.get("decay", 0.9)
            for a, p in zip(self._avg, self.params):
                a.mul_(d).add_(p.detach().float(), alpha=1 - d)
        else:
            for a, p in zip(self._avg, self.params):
                a.add_(p.detach().float() - a, alpha=1.0 / self._n_avg)

    @torch.no_grad()
    def _load_avg(self):
        """Put the averaged weights into the adapter; returns the live weights for _restore."""
        live = [p.detach().clone() for p in self.params]
        for a, p in zip(self._avg, self.params):
            p.copy_(a.to(p.dtype))
        return live

    @torch.no_grad()
    def _restore(self, live):
        for b, p in zip(live, self.params):
            p.copy_(b)

    # ---------------------------------------------------------- live stopping rules
    @staticmethod
    def parse_rule(spec):
        """'plateau:0.05' | 'rel:0.05' | 'best:loo_loss:0.0' | 'best:holdout_loss:0.05' | None"""
        if not spec:
            return None
        parts = spec.split(":")
        if parts[0] == "plateau":
            return {"kind": "plateau", "delta": float(parts[1])}
        if parts[0] == "rel":
            return {"kind": "rel", "frac": float(parts[1])}
        if parts[0] == "best":
            return {"kind": "best", "key": parts[1], "tol": float(parts[2]) if len(parts) > 2 else 0.0}
        raise ValueError(f"unknown stop rule {spec}")

    def _decide(self, rule, signals, step, max_steps):
        """(stop now?, predict with the best saved adapter?) at a checkpoint."""
        if step >= max_steps:
            return True, rule["kind"] == "best"
        cur = signals[-1]
        cps = [s_ for s_ in signals if s_["step"] in set(self.cfg["predict_at"]) and s_["step"] < step]
        if rule["kind"] == "plateau":
            if not cps or cps[-1].get("loo_loss") is None or cur.get("loo_loss") is None:
                return False, False
            a, b = cps[-1]["loo_loss"], cur["loo_loss"]
            return (a > 0 and (a - b) / a < rule["delta"]), False
        if rule["kind"] == "rel":
            v0 = signals[0].get("loo_loss")
            return (step > 0 and v0 and cur.get("loo_loss") is not None and cur["loo_loss"] / v0 < rule["frac"]), False
        return False, False

    def _track_best(self, rule, sig, step) -> bool:
        """Save the adapter if this checkpoint's signal beats the best so far. Returns True if it did
        (or if the signal is missing, so training continues to the full budget)."""
        v = sig.get(rule["key"])
        if v is None:
            return True
        if rule.get("best_val") is None or v < rule["best_val"] * (1 - rule["tol"]):
            rule["best_val"], rule["best_step"] = v, step
            rule["best_state"] = [p_.detach().clone() for p_ in self.params]
            return True
        return False
