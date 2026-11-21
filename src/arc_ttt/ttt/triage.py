"""Triage: skip puzzles predicted hopeless entirely (no training, no prediction), give the
rest the full budget. Offline, from logged runs.

    # within one run: fit on half the puzzles, pick the threshold there, score on the other half
    python -m arc_ttt.ttt.triage eval runs/ttt-rearc-0.5b/full-early

    # freeze a predictor on one run, then apply it unchanged to another dataset
    python -m arc_ttt.ttt.triage fit runs/ttt-rearc-0.5b/full-early --out results/final/triage.json
    python -m arc_ttt.ttt.triage eval runs/conceptarc/final --predictor results/final/triage.json

The decision uses only what is known before any training: the demos (count, implied output
size) and the leave-one-out loss of the untrained adapter at step 0 (~0.3 s). A skipped
puzzle costs that signal and scores 0. A kept puzzle costs the signal plus its full budget.

Fair competitor: the best a fixed budget can do at the same average compute, where a fixed
budget may also be run on a random subset of the puzzles (the rest skipped). That is the
upper concave hull of (0 s, 0%) and every fixed budget, with vote or PoE answers.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

from .allocate import auc
from .analyze import fixed_budget_table, fixed_cost, load_runs, score, sig_at
from .early import _cp, features

TAUS = [round(x, 3) for x in np.arange(0.04, 0.42, 0.02)]


# ------------------------------------------------------------------ predictor

def fit(X, y, l2=5.0, iters=800, lr=0.1) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(b)}


def predict(model: dict, X) -> np.ndarray:
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(model["w"]) + model["b"])))


def design(recs) -> np.ndarray:
    return np.array([features(r, 0, None)["loss"] for r in recs], dtype=float)


# ------------------------------------------------------------------ costs and the competitor

def final_step(recs) -> int:
    return max(set.intersection(*[{c["step"] for c in r["checkpoints"]} for r in recs]))


def answer_cp(rec, step, use_avg):
    cp = _cp(rec, step)
    return cp["avg"] if use_avg and "avg" in cp else cp


def labels(recs, method, use_avg):
    F = final_step(recs)
    return np.array([score(answer_cp(r, F, use_avg), r["targets"], method) > 0 for r in recs], dtype=float)


def hull(recs) -> list[tuple[float, float]]:
    """Upper concave hull of (cost, score) over: skip everything, and every fixed budget (vote or PoE)."""
    pts = [(0.0, 0.0)]
    for m in ("vote", "poe"):
        pts += [(row["seconds_per_puzzle"], row["arc_score"]) for row in fixed_budget_table(recs, m)]
    pts.sort()
    h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0:
            h.pop()
        h.append(p)
    # keep only the non-decreasing part (more compute never needs to score less)
    out = [h[0]]
    for p in h[1:]:
        if p[1] >= out[-1][1]:
            out.append(p)
    return out


def hull_at(h, secs) -> float:
    if secs >= h[-1][0]:
        return h[-1][1]
    for (x0, y0), (x1, y1) in zip(h, h[1:]):
        if secs <= x1:
            return y0 + (y1 - y0) * (secs - x0) / (x1 - x0) if x1 > x0 else y1
    return h[-1][1]


def policy(recs, probs, tau, method, use_avg):
    F = final_step(recs)
    sc, cost = [], []
    for r, p in zip(recs, probs):
        sig = sig_at(r, 0).get("signal_seconds") or 0.3
        if p < tau:
            sc.append(0.0); cost.append(sig)
        else:
            cp = answer_cp(r, F, use_avg)
            # averaged-weight answers come after the same training, so the cost is that checkpoint's
            sc.append(score(cp, r["targets"], method)); cost.append(fixed_cost(_cp(r, F), method) + sig)
    return sum(sc) / len(sc), statistics.mean(cost), sum(p < tau for p in probs) / len(probs)


def gain(recs, probs, tau, method, use_avg, h=None):
    sc, c, skipped = policy(recs, probs, tau, method, use_avg)
    return sc - hull_at(h or hull(recs), c), sc, c, skipped


def oof(X, y, rng, folds=5):
    idx = list(range(len(y))); rng.shuffle(idx)
    p = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = [i for i in idx if i not in set(te)]
        p[te] = predict(fit(X[tr], y[tr]), X[te])
    return p


def choose_tau(recs, X, y, method, use_avg, rng) -> float:
    """Threshold with the best mean gain over out-of-fold probabilities (5 repeats), on these puzzles only."""
    h = hull(recs)
    ps = [oof(X, y, random.Random(rng.random())) for _ in range(5)]
    return max(TAUS, key=lambda t: statistics.mean(gain(recs, p, t, method, use_avg, h)[0] for p in ps))


# ------------------------------------------------------------------ commands

def describe_hull(recs, method):
    fx = fixed_budget_table(recs, method)
    return ("fixed budgets (" + method + "): " +
            "  ".join(f"{r['budget']}: {100 * r['arc_score']:.1f}% @ {r['seconds_per_puzzle']:.1f}s" for r in fx))


def cmd_fit(args):
    recs = load_runs(args.dirs)
    X, y = design(recs), labels(recs, args.method, args.avg)
    tau = choose_tau(recs, X, y, args.method, args.avg, random.Random(0))
    model = fit(X, y)
    model.update(tau=tau, method=args.method, use_avg=args.avg, n_train=len(recs),
                 trained_on=[str(d) for d in args.dirs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=1))
    p = predict(model, X)
    print(f"fit on {len(recs)} puzzles ({int(y.sum())} solved): in-sample AUC {auc(p, y):.2f}, "
          f"threshold {tau} (chosen on out-of-fold probabilities) -> {args.out}")


def cmd_eval(args):
    recs = load_runs(args.dirs)
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        recs = [r for r in recs if r["task"] in keep]
    X, y = design(recs), labels(recs, args.method, args.avg)
    h = hull(recs)
    out = []
    p_ = out.append
    tag = " (averaged weights)" if args.avg else ""
    p_(f"TRIAGE - {len(recs)} puzzles, {args.method.upper()}{tag}, answers at step {final_step(recs)}; "
       f"{int(y.sum())} solved")
    p_("=" * 84)
    p_(describe_hull(recs, args.method))
    p_("fair competitor (best fixed budget, possibly on a random subset): " +
       "  ".join(f"{100 * s:.1f}% @ {c:.1f}s" for c, s in h[1:]))
    p_("")
    if args.predictor:
        model = json.loads(Path(args.predictor).read_text())
        prob = predict(model, X)
        g, sc, c, sk = gain(recs, prob, model["tau"], args.method, args.avg, h)
        p_(f"FROZEN predictor from {', '.join(model['trained_on'])} ({model['n_train']} puzzles), "
           f"threshold {model['tau']}")
        p_(f"  AUC on these puzzles: {auc(prob, y):.2f}")
        p_(f"  skips {100 * sk:.0f}% of puzzles -> {100 * sc:.1f}% @ {c:.1f}s vs fair competitor "
           f"{100 * hull_at(h, c):.1f}%  -> gain {100 * g:+.1f} pts")
        p_("  (at other thresholds: " + ", ".join(
            f"{t}: {100 * gain(recs, prob, t, args.method, args.avg, h)[0]:+.1f}" for t in TAUS[::3]) + ")")
    else:
        rng = random.Random(0)
        prob = oof(X, y, random.Random(1))
        p_(f"out-of-fold AUC: {auc(prob, y):.2f}")
        p_("  tau   skipped   score   s/puzzle   competitor   gain   (out-of-fold, all puzzles; threshold NOT chosen)")
        for t in TAUS[::2]:
            g, sc, c, sk = gain(recs, prob, t, args.method, args.avg, h)
            p_(f"  {t:<5} {100 * sk:5.0f}%   {100 * sc:5.1f}%   {c:6.1f}     {100 * hull_at(h, c):5.1f}%    {100 * g:+5.1f}")
        held = []
        idx = list(range(len(recs)))
        for rep in range(args.repeats):
            rng.shuffle(idx)
            A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
            for S, T in ((A, B), (B, A)):
                rS, rT = [recs[i] for i in S], [recs[i] for i in T]
                tau = choose_tau(rS, X[S], y[S], args.method, args.avg, rng)
                model = fit(X[S], y[S])
                g, sc, c, sk = gain(rT, predict(model, X[T]), tau, args.method, args.avg)
                held.append((g, sc, c, sk, tau))
        gs = [x[0] for x in held]
        p_("")
        p_(f"HELD-OUT: fit + choose the threshold on half the puzzles, apply to the other half ({len(held)} splits)")
        p_(f"  gain {100 * statistics.mean(gs):+.1f} pts (sd {100 * statistics.pstdev(gs):.1f}), positive in "
           f"{100 * sum(g > 0 for g in gs) / len(gs):.0f}% of splits; "
           f"score {100 * statistics.mean(x[1] for x in held):.1f}% @ {statistics.mean(x[2] for x in held):.1f}s, "
           f"skipping {100 * statistics.mean(x[3] for x in held):.0f}%; median threshold "
           f"{statistics.median(x[4] for x in held)}")
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval"):
        s = sub.add_parser(name)
        s.add_argument("dirs", nargs="+")
        s.add_argument("--method", choices=["vote", "poe"], default="poe")
        s.add_argument("--avg", action="store_true", help="answer with the averaged (EMA/SWA) weights")
        s.add_argument("--out", default=None if name == "eval" else "triage.json")
    sub.choices["eval"].add_argument("--predictor", default=None, help="frozen predictor from `fit`")
    sub.choices["eval"].add_argument("--repeats", type=int, default=10)
    sub.choices["eval"].add_argument("--subset", default=None)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_eval)(args)


if __name__ == "__main__":
    main()
