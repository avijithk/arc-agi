"""How much could adaptive compute gain at best? Oracle policies vs fixed budgets (offline).

    python -m arc_ttt.ttt.oracle runs/ttt-0.5b/full [--method poe]

- skip hopeless: puzzles solved at the final step get the full budget, the rest stop at step 0
- stop at first solve: each puzzle stops at the first checkpoint where it is solved
Both use knowledge no real policy has, so they bound what any adaptive rule can gain.
Also prints where the compute goes (training vs prediction) and when puzzles first get solved.
"""
from __future__ import annotations

import argparse
import statistics as st

from .analyze import fixed_budget_table, fixed_cost, fixed_score_at, load_runs, score


def report(recs, m="poe") -> str:
    out = []
    steps = sorted(set.intersection(*[{c["step"] for c in r["checkpoints"]} for r in recs]))
    cp = lambda r, s: next(c for c in r["checkpoints"] if c["step"] == s)
    S = {s: [score(cp(r, s), r["targets"], m) for r in recs] for s in steps}
    n, first_s, last = len(recs), steps[0], steps[-1]
    fixed = fixed_budget_table(recs, m)
    out.append(f"ORACLE BOUNDS - {n} puzzles, {m.upper()}")
    out.append("=" * 72)
    out.append("Fixed budgets: " + "  ".join(f"{f['budget']}: {100 * f['arc_score']:.1f}% @ {f['seconds_per_puzzle']:.1f}s"
                                         for f in fixed))
    out.append("Cost per puzzle (training + one prediction):")
    for s in steps:
        tr = st.mean(cp(r, s)["train_seconds"] for r in recs)
        pr = st.mean(fixed_cost(cp(r, s), m) - cp(r, s)["train_seconds"] for r in recs)
        out.append(f"  step {s:>3}: train {tr:5.1f}s + predict {pr:5.1f}s")

    def line(label, sc, cost):
        f = fixed_score_at(fixed, cost)
        out.append(f"{label:<28} {100 * sc:5.1f}% @ {cost:5.1f}s | best fixed budget at that cost {100 * f:5.1f}% "
                   f"-> gain {100 * (sc - f):+5.1f} pts")

    sc = sum(S[last]) / n
    line("Oracle: skip hopeless", sc, st.mean(fixed_cost(cp(r, last if S[last][i] else first_s), m)
                                              for i, r in enumerate(recs)))
    tot, costs = 0.0, []
    for i, r in enumerate(recs):
        s = next((s for s in steps if S[s][i] > 0), first_s)
        tot += S[s][i]; costs.append(fixed_cost(cp(r, s), m))
    line("Oracle: stop at first solve", tot / n, st.mean(costs))
    first = {s: 0 for s in steps}
    for i in range(n):
        s = next((s for s in steps if S[s][i] > 0), None)
        if s is not None:
            first[s] += 1
    ever = sum(first.values())
    forgot = sum(1 for i in range(n) if any(S[s][i] for s in steps) and not S[last][i])
    out.append(f"First solved at step: {first}; ever solved {ever} of {n}, unsolved at step {last} again: {forgot}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--method", choices=["vote", "poe"], default="poe")
    args = ap.parse_args()
    print(report(load_runs(args.dirs), args.method))


if __name__ == "__main__":
    main()
