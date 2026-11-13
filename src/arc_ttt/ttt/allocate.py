"""Budget allocation from features visible before TTT (offline, cross-validated).

A logistic model predicts from features that cost nothing to compute whether a task will be solved at
the largest logged budget. Policy: tasks with predicted probability < tau get a small budget
(`--low`, default 0 = no TTT), the rest get the largest logged budget. With logs that go
past 64 steps (e.g. configs/ttt_long.yaml, up to 128) this is *reallocation*: compute
saved on hopeless tasks pays for longer training on promising ones.

    python -m arc_ttt.ttt.allocate runs/ttt-0.5b/full [--low 0] [--repeats 50]

Each tau is fit on one random half of the tasks and scored on the other half, so no
number here was chosen on the tasks it is reported on.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics

import numpy as np

from .analyze import cp_at, fixed_budget_table, fixed_cost, fixed_score_at, load_runs, score, sig_at

FEATURES = ["n_pairs", "loo_loss_0", "has_size_rule", "log_expected_cells"]


def features(r) -> list[float]:
    """Only things known before spending any prediction compute: the number of demos, the
    step-0 leave-one-out loss (a few forward passes), and, in runs that log it, the output
    size implied by the demos. Step-0 *predictions* are not used: they would cost ~20 s each."""
    te = cp_at(r, 0)["tests"][0]
    exp = te.get("expected_shape")
    return [float(r.get("n_pairs") or 0), sig_at(r, 0).get("loo_loss") or 0.0,
            float(exp is not None), math.log(1 + exp[0] * exp[1]) if exp else 0.0]


def fit_logistic(X, y, l2=1.0, iters=500, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b)))


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def policy(recs, probs, tau, low, high, method):
    sc, cost = [], []
    for r, p in zip(recs, probs):
        cp = cp_at(r, low if p < tau else high)
        sc.append(score(cp, r["targets"], method))
        cost.append(fixed_cost(cp, method))
    return sum(sc) / len(sc), statistics.mean(cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--low", type=int, default=0, help="budget for tasks predicted hopeless")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--method", choices=["vote", "poe"], default="poe")
    args = ap.parse_args()
    recs = load_runs(args.dirs)
    high = max(cp["step"] for r in recs for cp in r["checkpoints"])
    X = np.array([features(r) for r in recs], dtype=float)
    y = np.array([score(cp_at(r, high), r["targets"], args.method) > 0 for r in recs], dtype=float)
    taus = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    rng = random.Random(0)
    res = {t: [] for t in taus}
    aucs = []
    for _ in range(args.repeats):
        idx = list(range(len(recs))); rng.shuffle(idx)
        A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
        for tr, te in ((A, B), (B, A)):
            model = fit_logistic(X[tr], y[tr])
            p = model(X[te])
            aucs.append(auc(p, y[te]))
            sub = [recs[i] for i in te]
            fixed = fixed_budget_table(sub, args.method)
            for t in taus:
                sc, secs = policy(sub, p, t, args.low, high, args.method)
                res[t].append((sc, secs, sc - fixed_score_at(fixed, secs)))
    full_fixed = {r["budget"]: r for r in fixed_budget_table(recs, args.method)}
    top = full_fixed[high]
    print(f"BUDGET ALLOCATION - {len(recs)} tasks, {args.method.upper()}, hopeless -> {args.low} steps, rest -> {high} steps")
    print("=" * 72)
    print(f"Predictor AUC on held-out halves: {np.nanmean(aucs):.2f} (0.5 = chance); "
          f"{int(y.sum())} of {len(y)} tasks solved at {high} steps")
    print(f"Fixed {high} steps: {100 * top['arc_score']:.1f}% at {top['seconds_per_puzzle']:.1f} s/puzzle")
    print()
    print(f"   tau    score    s/puzzle   compute vs fixed-{high}   gain vs best fixed at same compute")
    for t in taus:
        sc = statistics.mean(v[0] for v in res[t]); secs = statistics.mean(v[1] for v in res[t])
        g = statistics.mean(v[2] for v in res[t]); sd = statistics.pstdev(v[2] for v in res[t])
        print(f"   {t:<5}  {100 * sc:5.1f}%   {secs:6.1f}     {100 * secs / top['seconds_per_puzzle']:5.0f}%"
              f"              {100 * g:+5.1f} pts (sd {100 * sd:.1f})")
    print()
    print("Positive gain = the allocation beats simply using a smaller fixed budget for every task.")


if __name__ == "__main__":
    main()
