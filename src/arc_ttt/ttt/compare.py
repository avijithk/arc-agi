"""Compare runs side by side: one table + one accuracy-vs-compute plot.

    python -m arc_ttt.ttt.compare base=runs/ttt-0.5b/full constrained=runs/ttt-0.5b/dev-constrained \\
        rearc=runs/ttt-rearc-0.5b/full baseline=runs/baseline-0.5b/eval-full \\
        --subset configs/dev_subset.txt --out results/compare

Each argument is NAME=FOLDER. A folder can be a TTT run (ttt.shard*.jsonl; fixed-budget or
live adaptive) or a baseline evaluation (predictions.shard*.jsonl). --subset restricts every
run to the same task ids, so runs on the dev subset and on all 400 can be compared fairly.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ..baseline.evaluate import load_done
from ..metrics import summarize
from .analyze import METHODS, fixed_budget_table, load_runs, score


def ttt_rows(name, recs):
    if all(r.get("stop_rule") for r in recs):
        row = {"run": name, "kind": "live adaptive", "n": len(recs),
               "seconds": statistics.mean(r["seconds"] for r in recs), "step": "adaptive"}
        for m in METHODS:
            row[m] = sum(score(r["checkpoints"][-1], r["targets"], m) for r in recs) / len(recs)
        return [row], {}
    curves, rows = {}, []
    tables = {m: fixed_budget_table(recs, m) for m in METHODS}
    for i, r in enumerate(tables["poe"]):
        v = tables["vote"][i]
        rows.append({"run": name, "kind": "ttt", "n": len(recs), "step": r["budget"],
                     "vote": v["arc_score"], "poe": r["arc_score"], "seconds": r["seconds_per_puzzle"]})
    curves = {m: [(r["seconds_per_puzzle"], r["arc_score"], r["budget"]) for r in tables[m]] for m in METHODS}
    return rows, curves


def baseline_rows(name, folder, keep):
    recs = {}
    for p in sorted(Path(folder).glob("predictions.shard*.jsonl")):
        recs.update(load_done(p))
    if keep:
        recs = {k: v for k, v in recs.items() if k in keep}
    recs = {k: v for k, v in recs.items() if all(t is not None for t in v["targets"])}
    m = summarize(recs)
    secs = statistics.mean(v["seconds"] for v in recs.values())
    return [{"run": name, "kind": "baseline", "n": len(recs), "step": "-", "vote": m["arc_score"],
             "poe": None, "seconds": secs}], {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=FOLDER")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--by-concept", action="store_true", help="ConceptARC: also print PoE score per concept at the largest budget")
    args = ap.parse_args()
    keep = set(Path(args.subset).read_text().split()) if args.subset else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for spec in args.runs:
        name, folder = spec.split("=", 1)
        if list(Path(folder).glob("ttt.shard*.jsonl")):
            recs = load_runs([folder])
            if keep:
                recs = [r for r in recs if r["task"] in keep]
            rr, cc = ttt_rows(name, recs)
        elif list(Path(folder).glob("predictions.shard*.jsonl")):
            rr, cc = baseline_rows(name, folder, keep)
        else:
            print(f"skipping {name}: no results in {folder}")
            continue
        rows += rr
        if cc:
            curves[name] = cc

    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"
    lines = [f"RUN COMPARISON{' (subset: ' + args.subset + ')' if args.subset else ''}", "=" * 72,
             f"{'run':<22}{'tasks':>6}{'step':>9}{'vote':>9}{'PoE':>9}{'s/puzzle':>11}"]
    last = None
    for r in rows:
        if last and last != r["run"]:
            lines.append("")
        lines.append(f"{r['run'] if r['run'] != last else '':<22}{r['n']:>6}{str(r['step']):>9}"
                     f"{pct(r.get('vote')):>9}{pct(r.get('poe')):>9}{r['seconds']:>10.1f}s")
        last = r["run"]
    if args.by_concept:
        import re
        for spec in args.runs:
            name, folder = spec.split("=", 1)
            if not list(Path(folder).glob("ttt.shard*.jsonl")):
                continue
            recs = load_runs([folder])
            groups = {}
            for r in recs:
                c = re.sub(r"\d+$", "", r["task"])
                groups.setdefault(c, []).append(score(r["checkpoints"][-1], r["targets"], "poe"))
            lines += ["", f"{name}: PoE by concept (last checkpoint)"]
            lines += [f"   {c:<22}{100 * sum(v) / len(v):5.1f}%  ({len(v)} tasks)" for c, v in sorted(groups.items())]
    text = "\n".join(lines) + "\n"
    (out / "compare.txt").write_text(text)
    json.dump(rows, open(out / "compare.json", "w"), indent=2)
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, cc in curves.items():
            xs, ys, steps = zip(*cc["poe"])
            ax.plot(xs, [100 * y for y in ys], "o-", label=name)
            for x, y, s in cc["poe"]:
                ax.annotate(str(s), (x, 100 * y), textcoords="offset points", xytext=(3, -9), fontsize=7)
        for r in rows:
            if r["kind"] != "ttt":
                y = r["poe"] if r["poe"] is not None else r["vote"]
                ax.scatter([r["seconds"]], [100 * y], marker="*", s=120, zorder=5, label=f"{r['run']} ({r['kind']})")
        ax.set_xlabel("GPU seconds per puzzle"); ax.set_ylabel("ARC score, PoE (%)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "compare.png", dpi=170)
        print(f"plot: {out / 'compare.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
