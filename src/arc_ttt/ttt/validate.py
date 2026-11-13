"""Honest check of adaptive stopping rules: choose on one half of the tasks, score on the other.

Picking the best of ~20 rules on the same tasks you report on is optimistic. This splits
the tasks in half at random, picks the best rule (largest gain over the fixed-budget curve
at equal compute) on half A, scores that rule on half B, and repeats with many random
splits (and with A/B swapped).

    python -m arc_ttt.ttt.validate runs/ttt-0.5b/full [--repeats 50] [--method poe]

Reports the held-out gain of the selected rule (mean and spread), how often it beats the
fixed budgets by more than the noise threshold, and which rules get picked.
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter

from .analyze import (METHODS, adaptive_rules, fixed_budget_table, fixed_score_at, load_runs,
                      simulate, simulate_early_stop)


def candidate_rules(recs, method):
    """Every rule as (name, evaluate(recs) -> (arc_score, seconds)). Thresholds are taken
    from `recs` (the selection half), so the chosen rule is applied unchanged to the other half."""
    rules = []
    for name, fires, cost in adaptive_rules(recs):
        rules.append((name, lambda rs, f=fires, c=cost: simulate(rs, method, f, c)[:2]))
    for key in ("holdout_loss", "loo_loss"):
        if any(s.get(key) is not None for r in recs for s in r["signals"]):
            for tol in (0.0, 0.05, 0.15):
                rules.append((f"early stop on {key} (>{int(tol * 100)}%)",
                              lambda rs, k=key, t=tol: simulate_early_stop(rs, method, k, t)[:2]))
    return rules


def gain(recs, method, evaluate):
    sc, secs = evaluate(recs)
    return sc - fixed_score_at(fixed_budget_table(recs, method), secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--method", choices=METHODS, default="poe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_runs(args.dirs)
    n = len(recs)
    rng = random.Random(args.seed)
    noise = 4 / (n / 2)  # ~4 tasks in a half
    held, picked = [], Counter()
    for _ in range(args.repeats):
        idx = list(range(n)); rng.shuffle(idx)
        a = [recs[i] for i in idx[: n // 2]]
        b = [recs[i] for i in idx[n // 2:]]
        for sel, test in ((a, b), (b, a)):
            rules = candidate_rules(sel, args.method)
            name, ev = max(rules, key=lambda r: gain(sel, args.method, r[1]))
            held.append(gain(test, args.method, ev))
            label = name
            if "step-0" in name:
                label = name.split(" < ")[0] + " < k x its step-0 value"
            elif " < " in name:
                label = name.split(" < ")[0] + " < threshold"
            picked[label] += 1

    rules = candidate_rules(recs, args.method)
    best_all, ev_all = max(rules, key=lambda r: gain(recs, args.method, r[1]))
    full = {best_all: (gain(recs, args.method, ev_all),)}
    beats = sum(g >= noise for g in held) / len(held)
    worse = sum(g <= -noise for g in held) / len(held)
    pct = lambda v: f"{100 * v:+.1f} pts"
    print(f"RULE SELECTION CHECK - {n} tasks, method {args.method.upper()}, {len(held)} held-out evaluations")
    print("=" * 60)
    print(f"Best rule when chosen AND scored on all tasks: {best_all}")
    print(f"   apparent gain over fixed budgets: {pct(full[best_all][0])}  (optimistic: chosen on the same tasks)")
    print()
    print("Best rule chosen on one half, scored on the other half:")
    print(f"   mean held-out gain  {pct(statistics.mean(held))}   (spread: sd {100 * statistics.pstdev(held):.1f} pts)")
    print(f"   beats fixed budgets by > {100 * noise:.1f} pts (noise) in {100 * beats:.0f}% of splits;"
          f" worse by that much in {100 * worse:.0f}%")
    print()
    print("Rules most often picked:")
    for name, c in picked.most_common(5):
        print(f"   {c:>3}x  {name}")
    verdict = ("REAL: the gain survives on unseen tasks" if statistics.mean(held) >= noise / 2 and beats > 0.5
               else "NOT CONFIRMED: the gain mostly disappears on unseen tasks")
    print()
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
