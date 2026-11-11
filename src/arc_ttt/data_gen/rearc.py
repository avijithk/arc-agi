"""Build ARC-format training tasks from re-arc (Hodel, 2024).

re-arc ships 1000 verified generated input/output pairs for each of the 400 ARC-AGI-1
*training* tasks (no evaluation tasks, so no leakage). We sample `--per-task` pairs per
task and group them into synthetic tasks of `--pairs` pairs each (the last pair is the
test pair), written as data/rearc/<task>_<k>.json in the same format as ARC.

    bash scripts/download_rearc.sh                       # once: fetch + unzip re-arc
    python -m arc_ttt.data_gen.rearc --per-task 100 --out data/rearc
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..data import DATA_DIR


def build(src: Path, out: Path, per_task: int, pairs: int, seed: int, max_side: int = 30) -> int:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_tasks = 0
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"no re-arc task files in {src}; run scripts/download_rearc.sh first")
    for f in files:
        examples = [e for e in json.load(open(f))
                    if len(e["input"]) <= max_side and len(e["output"]) <= max_side
                    and len(e["input"][0]) <= max_side and len(e["output"][0]) <= max_side]
        rng.shuffle(examples)
        examples = examples[:per_task]
        for k in range(len(examples) // pairs):
            group = examples[k * pairs:(k + 1) * pairs]
            task = {"train": [{"input": e["input"], "output": e["output"]} for e in group[:-1]],
                    "test": [{"input": group[-1]["input"], "output": group[-1]["output"]}]}
            json.dump(task, open(out / f"{f.stem}_{k:03d}.json", "w"))
            n_tasks += 1
    return n_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA_DIR / "re-arc" / "re_arc" / "tasks"))
    ap.add_argument("--out", default=str(DATA_DIR / "rearc"))
    ap.add_argument("--per-task", type=int, default=100, help="generated pairs sampled per original task")
    ap.add_argument("--pairs", type=int, default=4, help="pairs per synthetic task (last one is the test pair)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = build(Path(args.src), Path(args.out), args.per_task, args.pairs, args.seed)
    print(f"wrote {n} synthetic tasks to {args.out}")


if __name__ == "__main__":
    main()
