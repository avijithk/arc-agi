"""Baseline: LoRA fine-tune a small LLM on serialized ARC training tasks.

Each training sample picks one pair of a task as the query and uses the other
pairs as demos, under a random augmentation (D8 symmetry + color permutation).
Loss is on the query's output grid only.

    python -m arc_ttt.baseline.train --config configs/baseline.yaml

Checkpoints go to <out_dir>/checkpoint every `save_every` optimizer steps; rerunning
the same command resumes from there (useful on spot instances).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from ..augment import Augmentation, random_color_perm
from ..data import DATA_DIR, load_split
from ..model_utils import amp_dtype, build_example, causal_lm_loss, load_model


def make_samples(tasks: list[dict], tok, per_task: int, max_len: int, seed: int) -> list[dict]:
    """One epoch of samples, deterministic given the seed."""
    rng = random.Random(seed)
    samples, skipped = [], 0
    for task in tasks:
        pairs = [p for p in task["train"] + task["test"] if "output" in p]
        if len(pairs) < 2:
            continue
        for _ in range(per_task):
            aug = Augmentation(d8=rng.randrange(8), colors=random_color_perm(rng))
            pairs_a = [{"input": aug.apply(p["input"]), "output": aug.apply(p["output"])} for p in pairs]
            rng.shuffle(pairs_a)
            query, demos = pairs_a[0], pairs_a[1:]
            ex = build_example(tok, demos, query["input"], query["output"], max_len)
            if ex is None:
                skipped += 1
                continue
            ex["task"] = task["id"]
            samples.append(ex)
    rng.shuffle(samples)
    if skipped:
        print(f"skipped {skipped} samples longer than {max_len} tokens")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out-dir", default=None, help="overrides config out_dir")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (for smoke tests)")
    ap.add_argument("--time-limit-hours", type=float, default=None,
                    help="checkpoint and exit after this long; rerun to resume (Kaggle sessions cap at 12h)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc, lc = cfg["train"], cfg["lora"]
    out = Path(args.out_dir or cfg["out_dir"])
    ckpt = out / "checkpoint"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    tok, model = load_model(cfg["model"], gradient_checkpointing=tc.get("gradient_checkpointing", True), training=True)
    resume = (ckpt / "state.pt").exists()
    if resume:
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()

    data_dir = Path(cfg.get("data_dir", DATA_DIR))
    tasks = []
    for split in tc["splits"]:
        tasks += load_split(split, data_dir)
    print(f"{len(tasks)} training tasks")

    epochs, accum = tc["epochs"], tc["grad_accum"]
    samples_per_epoch = len(make_samples(tasks, tok, tc["samples_per_task"], tc["max_seq_len"], cfg["seed"]))
    total_steps = math.ceil(samples_per_epoch * epochs / accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    warmup = int(tc.get("warmup_frac", 0.03) * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )

    amp = amp_dtype()  # fp16 autocast on T4/P100; None on bf16 GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=amp is not None)
    step = 0
    if resume:
        state = torch.load(ckpt / "state.pt", weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        print(f"resumed at optimizer step {step}/{total_steps}")

    log = open(out / "train_log.jsonl", "a")
    model.train()
    t0, running = time.time(), []
    done = step >= total_steps
    for epoch in range(epochs):
        if done:
            break
        samples = make_samples(tasks, tok, tc["samples_per_task"], tc["max_seq_len"], cfg["seed"] + epoch)
        start = max(0, step * accum - epoch * samples_per_epoch)  # skip what a resumed run already did
        for i in range(start, len(samples)):
            ex = samples[i]
            ids = torch.tensor([ex["input_ids"]], device=model.device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = causal_lm_loss(model, ids, ex["n_target"]) / accum
            scaler.scale(loss).backward()
            running.append(loss.item() * accum)
            if (i + 1) % accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tc.get("max_grad_norm", 1.0))
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % tc["log_every"] == 0 or step == total_steps:
                rec = {"step": step, "epoch": epoch, "loss": sum(running) / len(running),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                running = []
            out_of_time = args.time_limit_hours and time.time() - t0 > args.time_limit_hours * 3600
            if step % tc["save_every"] == 0 or step == total_steps or out_of_time:
                model.save_pretrained(ckpt)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                            "step": step}, ckpt / "state.pt")
            if out_of_time:
                print(f"time limit reached at step {step}/{total_steps}; checkpoint saved, rerun to resume")
                return
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    json.dump({"config": cfg, "steps": step, "train_seconds": round(time.time() - t0, 1)},
              open(out / "train_summary.json", "w"), indent=2)
    print(f"saved adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()
