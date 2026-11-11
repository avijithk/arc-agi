"""Shared warm-start: one short fine-tune on the *demonstration pairs* of all evaluation
tasks at once, before per-task TTT. Test inputs' answers are never used (the same
information per-task TTT already uses), so this is legitimate transductive training.

    python -m arc_ttt.ttt.warmstart --config configs/ttt.yaml --out runs/warmstart-0.5b/dev \
        --subset configs/dev_subset.txt
    # then TTT with:  warmstart_adapter: runs/warmstart-0.5b/dev/adapter   (configs/ttt_warmstart.yaml)

Starts from the baseline (merged), trains a fresh LoRA on leave-one-out examples of each
task's demos under random augmentations, and saves it. TTT merges it before training
each per-task adapter.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from ..augment import Augmentation, random_color_perm
from ..data import DATA_DIR, load_split
from ..model_utils import amp_dtype, build_example, causal_lm_loss, load_model


def make_samples(tasks, tok, per_task, max_len, seed):
    rng = random.Random(seed)
    out = []
    for t in tasks:
        pairs = t["train"]
        if len(pairs) < 2:
            continue
        for _ in range(per_task):
            aug = Augmentation(d8=rng.randrange(8), colors=random_color_perm(rng))
            pa = [{"input": aug.apply(p["input"]), "output": aug.apply(p["output"])} for p in pairs]
            rng.shuffle(pa)
            ex = build_example(tok, pa[1:], pa[0]["input"], pa[0]["output"], max_len)
            if ex is not None:
                out.append(ex)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ttt.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--samples-per-task", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = load_split(args.split, Path(cfg.get("data_dir", DATA_DIR)))
    if args.subset:
        keep = set(Path(args.subset).read_text().split())
        tasks = [t for t in tasks if t["id"] in keep]

    tok, model = load_model(cfg["model"], gradient_checkpointing=True, training=True)
    if cfg.get("baseline_adapter") and cfg["baseline_adapter"] != "none":
        model = PeftModel.from_pretrained(model, cfg["baseline_adapter"]).merge_and_unload()
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    amp = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp is not None)
    max_len = cfg["ttt"]["max_seq_len"]

    t0, step, log = time.time(), 0, open(out / "warmstart_log.jsonl", "a")
    model.train()
    for epoch in range(args.epochs):
        samples = make_samples(tasks, tok, args.samples_per_task, max_len, cfg["seed"] + epoch)
        print(f"epoch {epoch}: {len(samples)} samples from {len(tasks)} tasks", flush=True)
        run = []
        for i, ex in enumerate(samples):
            ids = torch.tensor([ex["input_ids"]], device=model.device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = causal_lm_loss(model, ids, ex["n_target"]) / args.grad_accum
            scaler.scale(loss).backward()
            run.append(loss.item() * args.grad_accum)
            if (i + 1) % args.grad_accum and i + 1 < len(samples):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": epoch, "loss": sum(run) / len(run), "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                run = []
    model.save_pretrained(out / "adapter")
    json.dump({"tasks": len(tasks), "steps": step, "seconds": round(time.time() - t0, 1), "args": vars(args)},
              open(out / "warmstart_summary.json", "w"), indent=2)
    print(f"saved warm-start adapter to {out / 'adapter'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
