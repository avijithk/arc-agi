"""Weight averaging vs the live adapter, paired within the same training run.

    python -m arc_ttt.ttt.wavg runs/ttt-rearc-0.5b/dev-swa [--subset configs/dev_subset.txt]

For every checkpoint that also has averaged-weight predictions (configs/ttt_swa.yaml,
configs/ttt_ema.yaml), prints the ARC score of the live and averaged adapters, the paired
wins/losses with an exact McNemar test, and how many "forgotten" puzzles (solved at an
earlier live checkpoint, unsolved at this one) the average recovers.
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

from .analyze import METHODS, fixed_cost, load_runs, score


def mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def avg_view(recs: list[dict]) -> list[dict]:
    """Records whose checkpoints are the averaged-weight predictions (for compare.py / analyze)."""
    out = []
    for r in recs:
        cps = [dict(cp["avg"], step=cp["step"]) for cp in r["checkpoints"] if "avg" in cp]
        if cps:
            out.append(dict(r, checkpoints=cps))
    return out


def table(recs: list[dict]) -> str:
    steps = sorted({cp["step"] for r in recs for cp in r["checkpoints"] if "avg" in cp})
    lines = [f"WEIGHT AVERAGING vs LIVE ADAPTER - {len(recs)} puzzles "
             f"({recs[0].get('weight_avg_kind', '')})".rstrip(), "=" * 78,
             "step  method   live    avg    avg wins/losses   p      forgotten  recovered  s/puzzle live/avg"]
    for s in steps:
        for m in METHODS:
            live, avg, w, l, forg, rec_ = [], [], 0, 0, 0, 0
            cost_l, cost_a = [], []
            for r in recs:
                cps = {cp["step"]: cp for cp in r["checkpoints"]}
                if s not in cps or "avg" not in cps[s]:
                    continue
                a = score(cps[s], r["targets"], m)
                b = score(cps[s]["avg"], r["targets"], m)
                live.append(a); avg.append(b)
                w += b > a; l += b < a
                earlier = any(score(cp, r["targets"], m) > 0 for st, cp in cps.items() if st < s)
                if earlier and a == 0:
                    forg += 1
                    rec_ += b > 0
                cost_l.append(fixed_cost(cps[s], m)); cost_a.append(fixed_cost(cps[s]["avg"], m))
            n = len(live)
            lines.append(f"{s:>4}  {m:<6} {100 * sum(live) / n:5.1f}% {100 * sum(avg) / n:5.1f}%"
                         f"      +{w}/-{l}{'':<8} {mcnemar(w, l):.3f}   {forg:>5}      {rec_:>5}"
                         f"      {sum(cost_l) / n:5.1f}/{sum(cost_a) / n:5.1f}")
    lines.append("")
    lines.append("forgotten = solved at an earlier live checkpoint but not at this one; "
                 "recovered = of those, solved by the averaged weights.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    recs = load_runs(args.dirs)
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        recs = [r for r in recs if r["task"] in keep]
    recs = [r for r in recs if any("avg" in cp for cp in r["checkpoints"])]
    if not recs:
        raise SystemExit("no averaged-weight predictions found (run with configs/ttt_swa.yaml or ttt_ema.yaml)")
    text = table(recs)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
