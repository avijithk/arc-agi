"""Run test-time training over ARC tasks, one JSON line per task.

    python -m arc_ttt.ttt.run --config configs/ttt.yaml --subset configs/dev_subset.txt \
        --out-dir runs/ttt-0.5b/dev

Starts from the base model with the baseline LoRA merged in (`baseline_adapter` in
the config), then trains a fresh TTT adapter per puzzle. Results are appended after
every task; rerunning skips finished tasks. Analyze with `python -m arc_ttt.ttt.analyze`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from ..baseline.evaluate import load_done
from ..data import DATA_DIR, load_split
from ..model_utils import load_model
from .core import TestTimeTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ttt.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline-adapter", default=None, help="overrides config; 'none' = raw base model")
    ap.add_argument("--holdout", action="store_true", help="hold one demo pair out of TTT for the stopping signal")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--time-limit-hours", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc = dict(cfg["ttt"])
    if args.holdout:
        tc["holdout"] = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"ttt.shard{args.shard}.jsonl"
    json.dump({"config": cfg, "ttt": tc, "args": vars(args)}, open(out / "run_config.json", "w"), indent=2)

    tasks = load_split(args.split, Path(cfg.get("data_dir", DATA_DIR)))
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        tasks = [t for t in tasks if t["id"] in keep]
    tasks = tasks[args.shard::args.num_shards][: args.limit]

    tok, model = load_model(cfg["model"], gradient_checkpointing=True, training=True)
    adapter = args.baseline_adapter or cfg.get("baseline_adapter")
    if adapter and adapter != "none":
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"merged baseline adapter {adapter}")
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    trainer = TestTimeTrainer(tok, model, tc)

    done = load_done_ttt(pred_path)
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
        rec = trainer.run_task(task, seed=cfg["seed"] + hash_id(task["id"]))
        rec["seconds"] = round(time.time() - t0, 2)
        if torch.cuda.is_available():
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        f.write(json.dumps(rec) + "\n"); f.flush()
        last = rec["checkpoints"][-1]
        solved = sum(any(g == t for g in te["vote"]) for te, t in zip(last["tests"], rec["targets"]))
        print(f"[{n}/{len(tasks)}] {task['id']}: step {last['step']} vote {solved}/{len(rec['targets'])} "
              f"({rec['seconds']}s)", flush=True)


def hash_id(task_id: str) -> int:
    return int(task_id, 16) % 100_000 if all(ch in "0123456789abcdef" for ch in task_id) else len(task_id)


def load_done_ttt(path: Path) -> dict:
    return load_done(path)


if __name__ == "__main__":
    main()
