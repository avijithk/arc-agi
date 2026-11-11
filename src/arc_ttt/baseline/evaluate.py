"""Evaluate a model (optionally with a LoRA adapter) on ARC tasks, 2 attempts per test input.

Attempt 1: greedy decode on the task as given.
Attempt 2: greedy decode on the transposed task, mapped back; if that equals
attempt 1, the next D8 symmetry is tried (up to all 8).

Predictions are appended to <out_dir>/predictions.jsonl after every task, and
tasks already there are skipped, so a rerun resumes after a crash or spot reclaim.

    python -m arc_ttt.baseline.evaluate --adapter runs/baseline-0.5b/adapter \
        --subset configs/dev_subset.txt --out-dir runs/baseline-0.5b/eval-dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ..augment import Augmentation
from ..data import DATA_DIR, load_split
from ..metrics import summarize
from ..model_utils import load_model, predict_grid

SECOND_ATTEMPT_ORDER = [4, 1, 2, 3, 5, 6, 7]  # transpose first, then the other symmetries


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                rec = json.loads(line)
                done[rec["task"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model", default=None, help="base model; defaults to config model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for the raw base model)")
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--subset", default=None, help="file of task ids, one per line (e.g. configs/dev_subset.txt)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--time-limit-hours", type=float, default=None, help="stop before this long; rerun to resume")
    ap.add_argument("--constrained", action="store_true", help="decode only well-formed grids")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_len = args.max_len or cfg["train"]["max_seq_len"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"predictions.shard{args.shard}.jsonl"

    tasks = load_split(args.split, Path(cfg.get("data_dir", DATA_DIR)))
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        tasks = [t for t in tasks if t["id"] in keep]
    tasks = tasks[args.shard::args.num_shards][: args.limit]

    tok, model = load_model(args.model or cfg["model"])
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    done = load_done(pred_path)
    print(f"{len(tasks)} tasks, {len(done)} already done")
    f = open(pred_path, "a")
    start = time.time()
    for n, task in enumerate(tasks, 1):
        if task["id"] in done:
            continue
        if args.time_limit_hours and time.time() - start > args.time_limit_hours * 3600:
            print("time limit reached; rerun to resume")
            break
        t0 = time.time()
        attempts, gen_seconds = [], 0.0
        for test in task["test"]:
            first = predict_grid(tok, model, task["train"], test["input"], Augmentation(), max_len, args.constrained)
            guesses = [first["grid"]]
            gen_seconds += first["seconds"]
            for k in SECOND_ATTEMPT_ORDER:
                second = predict_grid(tok, model, task["train"], test["input"], Augmentation(d8=k), max_len, args.constrained)
                gen_seconds += second["seconds"]
                if second["grid"] != first["grid"] or first["grid"] is None:
                    break
            guesses.append(second["grid"])
            attempts.append(guesses)
        rec = {
            "task": task["id"],
            "attempts": attempts,
            "targets": [t.get("output") for t in task["test"]],
            "seconds": round(time.time() - t0, 2),
            "gen_seconds": round(gen_seconds, 2),
        }
        f.write(json.dumps(rec) + "\n"); f.flush()
        done[task["id"]] = rec
        solved = sum(any(g == t for g in a if g is not None) for a, t in zip(attempts, rec["targets"]))
        print(f"[{n}/{len(tasks)}] {task['id']}: {solved}/{len(attempts)} solved ({rec['seconds']}s)", flush=True)

    summary = summarize_dir(out, adapter=args.adapter)
    print(json.dumps(summary, indent=2))


def summarize_dir(out: str | Path, adapter: str | None = None) -> dict:
    """Metrics over every predictions.shard*.jsonl in a folder; writes metrics.json."""
    out = Path(out)
    all_done = {}
    for p in sorted(out.glob("predictions.shard*.jsonl")):
        all_done.update(load_done(p))
    scored = {k: v for k, v in all_done.items() if all(t is not None for t in v["targets"])}
    summary = summarize(scored) if scored else {}
    summary["total_seconds"] = round(sum(v["seconds"] for v in all_done.values()), 1)
    summary["adapter"] = adapter
    json.dump(summary, open(out / "metrics.json", "w"), indent=2)
    return summary


if __name__ == "__main__":
    main()
