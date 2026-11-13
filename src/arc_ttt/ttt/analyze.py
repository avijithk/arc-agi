"""Analyze TTT runs: fixed-budget sweep, offline adaptive stopping, signal quality.

    python -m arc_ttt.ttt.analyze runs/ttt-0.5b/dev [more run dirs...] --out runs/ttt-0.5b/dev/analysis

Writes analysis.md (tables), analysis.json, and accuracy_vs_compute.png.

Fixed budget s: use the predictions logged at step s for every puzzle.
Adaptive rule:  for each puzzle, walk the checkpoints in order and stop at the first
                one where the rule fires (or the last one); use that checkpoint's
                predictions. Cost = training time up to the stop + the signal checks
                made along the way + one prediction.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ..metrics import cell_accuracy, size_match

METHODS = ("vote", "poe")


def load_runs(dirs: list[str]) -> list[dict]:
    recs = {}
    for d in dirs:
        for p in sorted(Path(d).glob("ttt.shard*.jsonl")):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    recs[r["task"]] = r
    return [r for r in recs.values() if all(t is not None for t in r["targets"])]


def steps_of(recs) -> list[int]:
    return sorted({cp["step"] for r in recs for cp in r["checkpoints"]})


def cp_at(rec, step):
    """Latest checkpoint at or before `step` (puzzles with no training data stop at 0)."""
    best = rec["checkpoints"][0]
    for cp in rec["checkpoints"]:
        if cp["step"] <= step:
            best = cp
    return best


def sig_at(rec, step):
    for s in rec["signals"]:
        if s["step"] == step:
            return s
    return {}


def score(cp, targets, method) -> float:
    vals = []
    for te, t in zip(cp["tests"], targets):
        guesses = te.get(method, [])[:2]
        vals.append(float(any(g == t for g in guesses)))
    return sum(vals) / len(vals)


def top1_stats(cp, targets, method):
    cell, size = [], []
    for te, t in zip(cp["tests"], targets):
        g = (te.get(method) or [None])[0]
        cell.append(cell_accuracy(g, t)); size.append(float(size_match(g, t)))
    return sum(cell) / len(cell), sum(size) / len(size)


def fixed_cost(cp, method="poe") -> float:
    """Training time to this checkpoint + one prediction (vote skips the PoE scoring time)."""
    pred = cp["predict_seconds"]
    if method == "vote":
        pred -= sum(te.get("poe_seconds", 0.0) for te in cp["tests"])
    return cp["train_seconds"] + pred


def fixed_budget_table(recs, method):
    rows = []
    for s in steps_of(recs):
        cps = [cp_at(r, s) for r in recs]
        sc = [score(cp, r["targets"], method) for cp, r in zip(cps, recs)]
        st = [top1_stats(cp, r["targets"], method) for cp, r in zip(cps, recs)]
        rows.append({
            "budget": s, "method": method, "arc_score": sum(sc) / len(sc),
            "cell_accuracy": sum(c for c, _ in st) / len(st), "size_match": sum(z for _, z in st) / len(st),
            "seconds_per_puzzle": statistics.mean(fixed_cost(cp, method) for cp in cps),
        })
    return rows


def simulate(recs, method, fires, check_cost=None):
    """fires(rec, prev_cp, cp) -> bool. Returns (arc_score, mean seconds, mean stop step)."""
    scores, costs, stops = [], [], []
    for r in recs:
        cps = r["checkpoints"]
        chosen, spent_checks, prev = cps[-1], 0.0, None
        for cp in cps:
            spent_checks += (check_cost(r, cp) if check_cost else 0.0)
            if fires(r, prev, cp):
                chosen = cp
                break
            prev = cp
        scores.append(score(chosen, r["targets"], method))
        costs.append(fixed_cost(chosen, method) + spent_checks)
        stops.append(chosen["step"])
    return sum(scores) / len(scores), statistics.mean(costs), statistics.mean(stops)


def simulate_early_stop(recs, method, key="holdout_loss", min_rel_improve=0.0):
    """Classic early stopping on a validation signal, evaluated at the prediction checkpoints.

    Keep the checkpoint with the lowest `key` seen so far; stop at the first checkpoint
    where the signal fails to improve on that best by more than `min_rel_improve`, and
    use the best checkpoint's predictions (in a live run: restore that adapter, predict once).
    Cost = training time up to the stopping checkpoint + signal checks + one prediction.
    Puzzles without the signal (e.g. < 3 demo pairs in holdout mode) use the full budget.
    """
    scores, costs, stops = [], [], []
    for r in recs:
        cps = r["checkpoints"]
        vals = [sig_at(r, cp["step"]).get(key) for cp in cps]
        if any(v is None for v in vals):
            chosen, stop_cp = cps[-1], cps[-1]
        else:
            best_i, stop_i = 0, len(cps) - 1
            for i in range(1, len(cps)):
                if vals[i] < vals[best_i] * (1 - min_rel_improve):
                    best_i = i
                else:
                    stop_i = i
                    break
            chosen, stop_cp = cps[best_i], cps[stop_i]
        checks = sum(signal_cost(r, cp) for cp in cps if cp["step"] <= stop_cp["step"])
        costs.append(stop_cp["train_seconds"] + fixed_cost(chosen, method) - chosen["train_seconds"] + checks)
        scores.append(score(chosen, r["targets"], method))
        stops.append(stop_cp["step"])
    return sum(scores) / len(scores), statistics.mean(costs), statistics.mean(stops)


def signal_cost(r, cp):
    return sum(s.get("signal_seconds", 0.0) for s in r["signals"] if s["step"] == cp["step"])


def adaptive_rules(recs):
    """Candidate stopping rules, each with a few thresholds taken from the data."""
    rules = []
    loo = sorted(s["loo_loss"] for r in recs for s in r["signals"] if s.get("loo_loss") is not None)
    has_ho = any(r.get("holdout_index") is not None for r in recs)
    qs = [0.1, 0.25, 0.5, 0.75]
    for q in qs:
        if loo:
            tau = loo[int(q * (len(loo) - 1))]
            rules.append((f"loo_loss < {tau:.3g} (q{int(q*100)})",
                          lambda r, p, cp, tau=tau: (sig_at(r, cp["step"]).get("loo_loss") or 1e9) < tau, signal_cost))
    for delta in (0.02, 0.05, 0.1):
        def plateau(r, prev, cp, delta=delta):
            if prev is None:
                return False
            a, b = sig_at(r, prev["step"]).get("loo_loss"), sig_at(r, cp["step"]).get("loo_loss")
            return a is not None and b is not None and a > 0 and (a - b) / a < delta
        rules.append((f"loo_loss plateau (<{int(delta*100)}% drop)", plateau, signal_cost))
    # Relative to each puzzle's own starting point: stop once the loss has fallen to a
    # fraction of its step-0 value (absolute losses aren't comparable across puzzles).
    def rel(key):
        def f(r, cp, frac):
            v0, v = sig_at(r, 0).get(key), sig_at(r, cp["step"]).get(key)
            return v0 is not None and v is not None and v0 > 0 and v / v0 < frac
        return f
    for frac in (0.05, 0.1, 0.2, 0.35, 0.5):
        rules.append((f"loo_loss < {frac:g} x its step-0 value",
                      lambda r, p, cp, frac=frac: rel("loo_loss")(r, cp, frac), signal_cost))
    if has_ho:
        for frac in (0.1, 0.25, 0.5):
            rules.append((f"holdout_loss < {frac:g} x its step-0 value",
                          lambda r, p, cp, frac=frac: rel("holdout_loss")(r, cp, frac), signal_cost))
        ho = sorted(s["holdout_loss"] for r in recs for s in r["signals"] if s.get("holdout_loss") is not None)
        for q in qs:
            if ho:
                tau = ho[int(q * (len(ho) - 1))]
                rules.append((f"holdout_loss < {tau:.3g} (q{int(q*100)})",
                              lambda r, p, cp, tau=tau: (sig_at(r, cp["step"]).get("holdout_loss") or 1e9) < tau,
                              signal_cost))
        rules.append(("holdout solved exactly", lambda r, p, cp: bool(cp.get("holdout_exact")),
                      lambda r, cp: signal_cost(r, cp) + cp.get("holdout_check_seconds", 0.0)))
    return rules


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(signal of a solved puzzle < signal of an unsolved one); lower loss should mean solved."""
    if not pos or not neg:
        return None
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def signal_quality(recs, method="vote"):
    rows = []
    for s in steps_of(recs):
        for key in ("loo_loss", "holdout_loss"):
            pos, neg = [], []
            for r in recs:
                v = sig_at(r, s).get(key)
                if v is None:
                    continue
                (pos if score(cp_at(r, s), r["targets"], method) > 0 else neg).append(v)
            a = auc(pos, neg)
            if a is not None:
                rows.append({"step": s, "signal": key, "auc": a, "n_solved": len(pos), "n_unsolved": len(neg)})
    return rows


def _test_cost(te, n_used, method):
    """Inference seconds for one test input if only the first n_used augmentations are used.
    Generation time is split evenly across augmentations within each logged batch; PoE time
    scales with the number of augmentations scored."""
    total = len(te["aug_cand"])
    per_batch = te.get("batch_seconds") or [te.get("gen_seconds", 0.0)]
    bsize = max(1, -(-total // len(per_batch)))
    gen = sum(sec * min(bsize, max(0, n_used - i * bsize)) / bsize for i, sec in enumerate(per_batch))
    poe = te.get("poe_seconds", 0.0) * n_used / total if method == "poe" else 0.0
    return gen + poe


def inference_replay(recs, step=None):
    """Offline replay of inference settings at one checkpoint (default: the largest step).
    Needs runs that logged every candidate (aug_cand). Returns rows for fixed numbers of
    augmentations and for adaptive stopping on agreement, each with and without the size filter."""
    from .predict import rank_from_log
    usable = [r for r in recs if all("aug_cand" in te for cp in r["checkpoints"] for te in cp["tests"])]
    if not usable:
        return []
    rows = []
    max_aug = min(len(te["aug_cand"]) for r in usable for cp in r["checkpoints"] for te in cp["tests"])
    settings = [("fixed", n, None) for n in (1, 2, 4, 8, 16, 32) if n <= max_aug]
    settings += [("adaptive", b, k) for b in (2,) for k in (2, 3, 4)]
    for filt in (False, True):
        for kind, a, k in settings:
            for m in METHODS:
                sc, cost, used = [], [], []
                for r in usable:
                    cp = cp_at(r, step if step is not None else max(c["step"] for c in r["checkpoints"]))
                    vals, csts, us = [], [], []
                    for te, t in zip(cp["tests"], r["targets"]):
                        n = len(te["aug_cand"])
                        if kind == "fixed":
                            n_used = a
                        else:
                            n_used = n
                            for end in range(a, n + a, a):
                                cnt = {}
                                ok = te.get("size_ok") or []
                                for i in te["aug_cand"][:min(end, n)]:
                                    if i >= 0 and (not filt or not ok or ok[i] or not any(ok)):
                                        cnt[i] = cnt.get(i, 0) + 1
                                if cnt and max(cnt.values()) >= k:
                                    n_used = min(end, n)
                                    break
                        vote, poe = rank_from_log(te, n_used, size_filter=filt)
                        guesses = vote if m == "vote" else poe
                        vals.append(float(any(g == t for g in guesses)))
                        csts.append(_test_cost(te, n_used, m))
                        us.append(n_used)
                    sc.append(sum(vals) / len(vals)); cost.append(sum(csts)); used.append(sum(us) / len(us))
                name = f"{a} augs" if kind == "fixed" else f"batches of {a}, stop at {k} agreeing"
                rows.append({"setting": name, "size_filter": filt, "method": m, "arc_score": sum(sc) / len(sc),
                             "infer_seconds": statistics.mean(cost), "mean_augs": statistics.mean(used)})
    return rows


def inference_text(rows) -> str:
    if not rows:
        return ""
    pct = lambda v: f"{100 * v:5.1f}%"
    L = ["", "5) Inference: how many candidates, and does the size filter help?",
         "   (replayed offline at the last checkpoint; seconds = inference only, per puzzle)", ""]
    for m in METHODS:
        L.append(f"   {m.upper()}")
        L.append("     setting                               no filter            size filter")
        names = list(dict.fromkeys(r["setting"] for r in rows))
        for nm in names:
            a = next(r for r in rows if r["setting"] == nm and r["method"] == m and not r["size_filter"])
            b = next(r for r in rows if r["setting"] == nm and r["method"] == m and r["size_filter"])
            L.append(f"     {nm:<36}  {pct(a['arc_score'])} {a['infer_seconds']:5.1f}s   {pct(b['arc_score'])} {b['infer_seconds']:5.1f}s"
                     f"   ({a['mean_augs']:.1f} augs)")
        L.append("")
    return "\n".join(L) + "\n"


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return head + "".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n" for r in rows)


def fixed_score_at(fixed_rows, secs):
    """Best ARC score a fixed budget reaches within this compute (upper envelope of the
    fixed-budget curve, linearly interpolated). If a smaller budget scores higher, that
    is what a fixed-budget user would pick, so it's the fair comparison."""
    pts, best = [], -1.0
    for x, y in sorted((r["seconds_per_puzzle"], r["arc_score"]) for r in fixed_rows):
        best = max(best, y)
        pts.append((x, best))
    if secs <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if secs <= x1:
            return y0 + (y1 - y0) * (secs - x0) / (x1 - x0) if x1 > x0 else y1
    return pts[-1][1]


def summary_text(n, fixed, adaptive, quality, has_holdout) -> str:
    pct = lambda v: f"{100 * v:5.1f}%"
    L = [f"TTT RESULTS - {n} puzzles", "=" * 60, "",
         "1) Fixed budgets: ARC score (GPU seconds per puzzle)", "",
         "   steps      vote              PoE"]
    by = {(r["method"], r["budget"]): r for r in fixed}
    for b in sorted({r["budget"] for r in fixed}):
        v, q = by.get(("vote", b)), by.get(("poe", b))
        L.append(f"   {b:>5}   {pct(v['arc_score'])} ({v['seconds_per_puzzle']:4.0f}s)   "
                 f"{pct(q['arc_score'])} ({q['seconds_per_puzzle']:4.0f}s)")
    L += ["", "2) Adaptive stopping vs. a fixed budget with the SAME compute",
          "   (gain > 0 means the rule beats fixed budgets; best 5 per method)", ""]
    for m in METHODS:
        fr = [r for r in fixed if r["method"] == m]
        top = max(fr, key=lambda r: r["budget"])
        rows = []
        for r in (x for x in adaptive if x["method"] == m):
            gain = r["arc_score"] - fixed_score_at(fr, r["seconds_per_puzzle"])
            rows.append((gain, r))
        rows.sort(key=lambda t: -t[0])
        L.append(f"   {m.upper()}  (max budget {top['budget']} steps: {pct(top['arc_score'])} at {top['seconds_per_puzzle']:.0f}s)")
        for gain, r in rows[:5]:
            L.append(f"     {pct(r['arc_score'])}  {100 * r['compute_vs_max_budget']:4.0f}% compute  "
                     f"gain {100 * gain:+5.1f} pts   {r['rule']}")
        L.append("")
    L += ["3) Does the stopping signal predict which puzzles get solved?",
          "   (AUC: 0.5 = no better than chance, 1.0 = perfect)", ""]
    for q in quality:
        L.append(f"   step {q['step']:>3}  {q['signal']:<13} AUC {q['auc']:.2f}   ({q['n_solved']} solved)")
    best = []
    for m in METHODS:
        fr = [r for r in fixed if r["method"] == m]
        cand = [(r["arc_score"] - fixed_score_at(fr, r["seconds_per_puzzle"]), r) for r in adaptive if r["method"] == m]
        if cand:
            best.append((m,) + max(cand, key=lambda t: t[0]))
    noise = 4 / n  # ~4 puzzles; smaller differences are within run-to-run noise
    L += ["", "4) Bottom line", f"   (1 puzzle = {100 / n:.2f} pts; gains under {100 * noise:.1f} pts count as a tie)", ""]
    for m, gain, r in best:
        verdict = "BEATS" if gain >= noise else ("roughly TIES" if gain > -noise else "does not beat")
        L.append(f"   {m.upper()}: best rule {verdict} fixed budgets ({100 * gain:+.1f} pts at equal compute): "
                 f"{r['rule']} -> {pct(r['arc_score']).strip()} using {100 * r['compute_vs_max_budget']:.0f}% of max compute")
    if not has_holdout:
        L.append("   (no held-out runs here; holdout rules need a HOLDOUT=1 run)")
    return "\n".join(L) + "\n"


def live_summary(recs) -> str:
    """Summary for a live adaptive run (one prediction per puzzle, at the stopping step)."""
    n = len(recs)
    rules = sorted({r["stop_rule"] for r in recs})
    L = [f"LIVE ADAPTIVE RUN - {n} puzzles, stop rule {', '.join(rules)}", "=" * 60, ""]
    for m in METHODS:
        sc = sum(score(r["checkpoints"][-1], r["targets"], m) for r in recs) / n
        L.append(f"   {m.upper():<5} ARC score {100 * sc:5.1f}%")
    secs = [r["seconds"] for r in recs if "seconds" in r]
    L += ["", f"   GPU time per puzzle (measured, everything included): {statistics.mean(secs):.1f} s",
          f"   total GPU time: {sum(secs) / 3600:.2f} h",
          f"   mean stopping step {statistics.mean(r['stopped_at'] for r in recs):.1f}; "
          f"mean step predicted from {statistics.mean(r['predicted_at'] for r in recs):.1f}",
          "", "   Compare with a fixed-budget run's analysis at the same settings (seconds_per_puzzle there",
          "   counts training + one prediction, which is what this run spends)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--adaptive-infer", action="store_true", help="(kept for old commands; replay now runs automatically)")
    ap.add_argument("--subset", default=None, help="only analyze task ids in this file, e.g. configs/dev_subset.txt")
    args = ap.parse_args()
    recs = load_runs(args.dirs)
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        recs = [r for r in recs if r["task"] in keep]
    out = Path(args.out or Path(args.dirs[0]) / "analysis"); out.mkdir(parents=True, exist_ok=True)
    if recs and all(r.get("stop_rule") for r in recs):
        txt = live_summary(recs)
        (out / "summary.txt").write_text(txt)
        print(txt)
        return

    fixed = [row for m in METHODS for row in fixed_budget_table(recs, m)]
    max_cost = {m: max(r["seconds_per_puzzle"] for r in fixed if r["method"] == m) for m in METHODS}
    adaptive = []
    for name, fires, cost in adaptive_rules(recs):
        for m in METHODS:
            sc, secs, stop = simulate(recs, m, fires, cost)
            adaptive.append({"rule": name, "method": m, "arc_score": sc, "seconds_per_puzzle": secs,
                             "compute_vs_max_budget": secs / max_cost[m], "mean_stop_step": stop})
    has_ho = any(r.get("holdout_index") is not None for r in recs)
    for key in (["holdout_loss"] if has_ho else []) + ["loo_loss"]:
        for tol in (0.0, 0.05, 0.15):
            for m in METHODS:
                sc, secs, stop = simulate_early_stop(recs, m, key, tol)
                adaptive.append({"rule": f"early stop on {key} (needs >{int(tol*100)}% improvement)", "method": m,
                                 "arc_score": sc, "seconds_per_puzzle": secs,
                                 "compute_vs_max_budget": secs / max_cost[m], "mean_stop_step": stop})
    quality = signal_quality(recs)

    json.dump({"n_puzzles": len(recs), "fixed": fixed, "adaptive": adaptive, "signal_quality": quality,
               "inference": inference_replay(recs)}, open(out / "analysis.json", "w"), indent=2)
    md = [f"# TTT analysis ({len(recs)} puzzles)\n",
          "## Fixed budgets\n", md_table(fixed, ["method", "budget", "arc_score", "cell_accuracy", "size_match", "seconds_per_puzzle"]),
          "\n## Adaptive stopping (offline)\n", md_table(adaptive, ["method", "rule", "arc_score", "seconds_per_puzzle", "compute_vs_max_budget", "mean_stop_step"]),
          "\n## Does the signal predict success? (AUC, 0.5 = no)\n", md_table(quality, ["step", "signal", "auc", "n_solved", "n_unsolved"])]
    (out / "analysis.md").write_text("\n".join(md))
    infer = inference_replay(recs)
    summ = summary_text(len(recs), fixed, adaptive, quality, any(r.get("holdout_index") is not None for r in recs))
    summ += inference_text(infer)
    if infer:
        md.append("\n## Inference replay\n" + md_table(infer, ["method", "setting", "size_filter", "arc_score", "infer_seconds", "mean_augs"]))
        (out / "analysis.md").write_text("\n".join(md))
    (out / "summary.txt").write_text(summ)
    print(summ)
    print(f"full tables: {out / 'analysis.md'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, m in zip(axes, METHODS):
            fr = [r for r in fixed if r["method"] == m]
            ax.plot([r["seconds_per_puzzle"] for r in fr], [100 * r["arc_score"] for r in fr], "o-", label="fixed budget")
            for r in fr:
                ax.annotate(str(r["budget"]), (r["seconds_per_puzzle"], 100 * r["arc_score"]),
                            textcoords="offset points", xytext=(4, -10), fontsize=8)
            ad = [r for r in adaptive if r["method"] == m]
            ax.scatter([r["seconds_per_puzzle"] for r in ad], [100 * r["arc_score"] for r in ad],
                       marker="x", label="adaptive rules")
            ax.set_title(f"{m}"); ax.set_xlabel("GPU seconds per puzzle"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("ARC score (%)"); axes[0].legend()
        fig.tight_layout(); fig.savefig(out / "accuracy_vs_compute.png", dpi=150)
        print(f"plot: {out / 'accuracy_vs_compute.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
