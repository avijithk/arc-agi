"""Is the early signal really missing? A learned predictor of "will this puzzle be solved",
using everything visible at decision step k (offline, cross-validated).

    python -m arc_ttt.ttt.early runs/ttt-rearc-0.5b/full-early [--method poe] [--repeats 20]

Our hand-made stopping rules failed. This asks whether *any* simple predictor built on the
signals available at step k can tell, early, which puzzles will be solved at the final
budget F (the largest logged step). Feature groups, each a superset of the previous one:

  free   known before any compute: number of demos, implied output size, training-set size
  loss   + the leave-one-out loss curve up to step k (level, ratio to step 0, recent slope)
  pred   + the step-k predictions: agreement of the 8 views, number of distinct answers,
           valid fraction, size-rule check, PoE confidence and margin, stability vs the
           previous checkpoint

Models: L2 logistic regression, and gradient-boosted trees (if scikit-learn is installed).
Every number is out-of-fold (repeated stratified 5-fold CV), so nothing is scored on the
puzzles it was fit on.

Policy at decision step k: puzzles predicted unlikely to be solved (p < tau) stop at k and
answer with their step-k prediction; the rest train on to F. Its cost includes the step-k
prediction for every puzzle. "gain" = score minus the best a plain fixed budget reaches at
the same compute (upper envelope). "held-out" picks (k, model, tau) on one random half of
the puzzles and reports the gain on the other half.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .allocate import auc, fit_logistic
from .analyze import fixed_budget_table, fixed_cost, fixed_score_at, load_runs, score, sig_at

GROUPS = ("free", "loss", "pred")


# ------------------------------------------------------------------ features

def _cp(rec, step):
    for cp in rec["checkpoints"]:
        if cp["step"] == step:
            return cp
    return None


def _loo(rec, step):
    v = sig_at(rec, step).get("loo_loss")
    return float(v) if v is not None else float("nan")


def _test_feats(te) -> list[float]:
    ac = te.get("aug_cand") or []
    n = max(1, len(ac))
    valid = [i for i in ac if i >= 0]
    counts = Counter(valid)
    top, top_n = (counts.most_common(1)[0] if counts else (None, 0))
    size_ok = te.get("size_ok") or []
    lp = te.get("poe_lp") or {}
    sums = sorted((sum(v for v in row if v is not None) / max(1, sum(v is not None for v in row))
                   for row in lp.values()), reverse=True)
    conf = sums[0] if sums else -50.0
    margin = (sums[0] - sums[1]) if len(sums) > 1 else 50.0
    return [top_n / n, len(counts) / n, len(valid) / n,
            float(bool(size_ok) and top is not None and size_ok[top]),
            max(conf, -50.0), min(margin, 50.0)]


def _top_vote(te):
    v = te.get("vote") or []
    return v[0] if v else None


def features(rec, k: int, prev_k: int | None) -> dict[str, list[float]]:
    """Feature vector per group for decision step k (only information available at k)."""
    cp = _cp(rec, k)
    te0 = cp["tests"][0]
    exp = te0.get("expected_shape")
    free = [float(rec.get("n_pairs") or 0), float(exp is not None),
            math.log(1 + exp[0] * exp[1]) if exp else 0.0,
            math.log(1 + (rec.get("n_train_samples") or 0))]
    l0, lk = _loo(rec, 0), _loo(rec, k)
    lprev = _loo(rec, max(0, k - 8))
    loss = [math.log(l0 + 1e-6) if not math.isnan(l0) else 0.0,
            math.log(lk + 1e-6) if not math.isnan(lk) else 0.0,
            (lk / l0) if l0 and not math.isnan(lk) else 1.0,
            ((lprev - lk) / l0) if l0 and not math.isnan(lk) and not math.isnan(lprev) else 0.0]
    tf = np.mean([_test_feats(te) for te in cp["tests"]], axis=0).tolist()
    stable = 0.0
    if prev_k is not None:
        pcp = _cp(rec, prev_k)
        stable = float(np.mean([_top_vote(a) is not None and _top_vote(a) == _top_vote(b)
                                for a, b in zip(cp["tests"], pcp["tests"])]))
    pred = tf + [stable]
    return {"free": free, "loss": free + loss, "pred": free + loss + pred}


# ------------------------------------------------------------------ models

def make_models():
    models = {"logistic": lambda X, y: fit_logistic(X, y, l2=5.0)}
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        def gbt(X, y):
            m = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                           subsample=0.8, random_state=0).fit(X, y)
            return lambda Xn: m.predict_proba(Xn)[:, 1]
        models["boosted trees"] = gbt
    except ImportError:
        pass
    return models


def oof_probs(X, y, fit, rng: random.Random, folds=5):
    """Out-of-fold probabilities from one stratified k-fold split."""
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    fold = {}
    for j, i in enumerate(pos + neg):
        fold[i] = j % folds
    p = np.zeros(len(y))
    for f in range(folds):
        tr = [i for i in range(len(y)) if fold[i] != f]
        te = [i for i in range(len(y)) if fold[i] == f]
        if y[tr].min() == y[tr].max():
            p[te] = y[tr].mean()
            continue
        p[te] = fit(X[tr], y[tr])(X[te])
    return p


# ------------------------------------------------------------------ policy

def policy(recs, probs, tau, k, final, method, needs_pred=True):
    """needs_pred: the features use the step-k predictions, so every puzzle pays for them;
    otherwise only the puzzles that stop at k predict there."""
    sc, cost = [], []
    for r, p in zip(recs, probs):
        ck, cf = _cp(r, k), _cp(r, final)
        pred_k = fixed_cost(ck, method) - ck["train_seconds"] if needs_pred else 0.0
        if p < tau:
            sc.append(score(ck, r["targets"], method)); cost.append(fixed_cost(ck, method))
        else:
            sc.append(score(cf, r["targets"], method)); cost.append(fixed_cost(cf, method) + pred_k)
    return sum(sc) / len(sc), statistics.mean(cost)


def gain(recs, probs, tau, k, final, method, group="pred"):
    sc, secs = policy(recs, probs, tau, k, final, method, needs_pred=(group == "pred"))
    return sc - fixed_score_at(fixed_budget_table(recs, method), secs), sc, secs


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--method", choices=["vote", "poe"], default="poe")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    recs = load_runs(args.dirs)
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        recs = [r for r in recs if r["task"] in keep]
    steps = sorted(set.intersection(*[{cp["step"] for cp in r["checkpoints"]} for r in recs]))
    final = steps[-1]
    decide = [s for s in steps if s < final]
    if not decide:
        raise SystemExit("needs a run with several prediction checkpoints (e.g. configs/ttt_early.yaml)")
    has_pred = "aug_cand" in recs[0]["checkpoints"][0]["tests"][0]
    groups = GROUPS if has_pred else ("free", "loss")   # older logs lack the per-view candidates
    y = np.array([score(_cp(r, final), r["targets"], args.method) > 0 for r in recs], dtype=float)
    models = make_models()
    rng = random.Random(0)
    taus = [0.05, 0.1, 0.15, 0.2, 0.3]

    out = []
    p = lambda s="": out.append(s)
    fixed = {row["budget"]: row for row in fixed_budget_table(recs, args.method)}
    p(f"EARLY PREDICTABILITY - {len(recs)} puzzles, {args.method.upper()}, label = solved at step {final} "
      f"({int(y.sum())} of {len(y)})")
    p("=" * 86)
    p("Fixed budgets: " + "  ".join(f"{s}: {100 * fixed[s]['arc_score']:.1f}% @ {fixed[s]['seconds_per_puzzle']:.0f}s"
                                    for s in steps))
    p()

    # AUC table and out-of-fold probabilities per (k, group, model, repeat)
    probs = {}
    p("AUC for 'solved at the final step', out-of-fold, mean over repeats (0.5 = chance)")
    header = "  k   " + "".join(f"{g + ' / ' + m[:8]:>22}" for g in groups for m in models)
    p(header)
    for i, k in enumerate(decide):
        prev = decide[i - 1] if i else None
        F = [features(r, k, prev) for r in recs]
        row = f"{k:>3}   "
        for g in groups:
            X = np.array([f[g] for f in F], dtype=float)
            for mname, fit in models.items():
                aucs, ps = [], []
                for rep in range(args.repeats):
                    pr = oof_probs(X, y, fit, random.Random(1000 * rep + k))
                    ps.append(pr); aucs.append(auc(pr, y))
                probs[(k, g, mname)] = ps
                row += f"{np.nanmean(aucs):>22.2f}"
        p(row)
    fin_agree = np.array([_test_feats(_cp(r, final)["tests"][0])[0] for r in recs]) if has_pred else None
    if has_pred:
        p(f"reference: agreement of the 8 views AT step {final} itself has AUC {auc(fin_agree, y):.2f} "
          f"(what you know only after paying for the full budget)")
    p()

    # Policy with the richest features, per k and tau (all puzzles, out-of-fold probabilities)
    for grp, mname in [(g_, m_) for g_ in groups if g_ != "free" for m_ in models]:
        p(f"POLICY, {grp} features ({mname}): stop at k if p < tau, else train to {final}"
          + ("  [every puzzle pays for the step-k prediction]" if grp == "pred" else ""))
        p("  k    tau    score   s/puzzle   gain vs best fixed budget at same compute")
        for k in decide:
            for t in taus:
                res = [gain(recs, pr, t, k, final, args.method, grp) for pr in probs[(k, grp, mname)]]
                g = [r_[0] for r_ in res]
                p(f"{k:>3}   {t:<5}  {100 * statistics.mean(r_[1] for r_ in res):5.1f}%   "
                  f"{statistics.mean(r_[2] for r_ in res):6.1f}     {100 * statistics.mean(g):+5.1f} pts "
                  f"(sd {100 * statistics.pstdev(g):.1f})")
        p()

    # Held-out selection: choose (k, group, model, tau) on half A, report gain on half B
    held = []
    idx = list(range(len(recs)))
    for rep in range(args.repeats):
        rng.shuffle(idx)
        A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
        for S, T in ((A, B), (B, A)):
            best, best_g = None, -1e9
            for (k, g, m), ps in probs.items():
                pr = ps[rep]
                for t in taus:
                    gA = gain([recs[i] for i in S], pr[S], t, k, final, args.method, g)[0]
                    if gA > best_g:
                        best, best_g = (k, g, m, t), gA
            k, g, m, t = best
            held.append((gain([recs[i] for i in T], probs[(k, g, m)][rep][T], t, k, final, args.method, g)[0], best))
    hg = [h[0] for h in held]
    picks = Counter((h[1][0], h[1][1], h[1][2]) for h in held).most_common(3)
    p(f"HELD-OUT: pick the best (k, features, model, tau) on half the puzzles, score it on the other half")
    p(f"  gain vs fixed budget: {100 * statistics.mean(hg):+.1f} pts (sd {100 * statistics.pstdev(hg):.1f}); "
      f"positive in {100 * sum(v > 0 for v in hg) / len(hg):.0f}% of splits")
    p("  most often picked: " + ", ".join(f"k={k} {g}/{m} ({c}x)" for (k, g, m), c in picks))

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
