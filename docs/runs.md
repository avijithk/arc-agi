# Run guide (current as of Nov 21)

Every command runs **on an instance, inside tmux**, after:

```bash
cd ~/arc-agi && git pull && source /opt/pytorch/bin/activate && python -m pytest -q
tmux new -s run        # detach: Ctrl-b then d; back: tmux attach -t run
```

Every run is resumable: rerun the same command after a crash, disconnect or spot reclaim.
"dev" = the 100-puzzle subset in `configs/dev_subset.txt`. Times are A10G estimates.

## Instance A (Avi)

| Run | Command | Time | Output |
|---|---|---|---|
| Data prep (CPU) | `bash scripts/download_rearc.sh && python -m arc_ttt.data_gen.rearc --per-task 100 --out data/rearc` | 10 min | `data/rearc/` |
| **R9a** constrained decoding + size filter (dev) | `CONFIG=configs/ttt_constrained.yaml OUT=runs/ttt-0.5b/dev-constrained bash scripts/run_ttt.sh` | 3 h | `runs/ttt-0.5b/dev-constrained` |
| Warm-start adapter (dev) | `python -m arc_ttt.ttt.warmstart --config configs/ttt.yaml --out runs/warmstart-0.5b/dev --subset configs/dev_subset.txt` | 15 min | `runs/warmstart-0.5b/dev/adapter` |
| **R9b** TTT after warm-start (dev) | `CONFIG=configs/ttt_warmstart.yaml OUT=runs/ttt-0.5b/dev-warmstart bash scripts/run_ttt.sh` | 3 h | `runs/ttt-0.5b/dev-warmstart` |
| **R2** re-arc baseline | `python -m arc_ttt.baseline.train --config configs/baseline_rearc.yaml` | 5–6 h | `runs/baseline-rearc-0.5b/adapter` |
| **R3** TTT on re-arc baseline, all 400 | `CONFIG=configs/ttt_rearc.yaml OUT=runs/ttt-rearc-0.5b/full FULL=1 bash scripts/run_ttt.sh` | 7–9 h | `runs/ttt-rearc-0.5b/full` |
| **R6** 1.5B baseline + TTT (dev) | `python -m arc_ttt.baseline.train --config configs/baseline_1.5b.yaml && CONFIG=configs/ttt_1.5b.yaml OUT=runs/ttt-1.5b/dev bash scripts/run_ttt.sh` | 5 h | `runs/ttt-1.5b/dev` |
| **R8** ConceptARC, best setup | `bash scripts/download_conceptarc.sh && SPLIT=conceptarc CONFIG=<best config> OUT=runs/conceptarc/best FULL=1 bash scripts/run_ttt.sh` | 3 h | `runs/conceptarc/best` |

Before R3: if R9a/R9b showed a gain, keep `constrained_decoding`/`size_filter` on in `configs/ttt_rearc.yaml` and add a `warmstart_adapter:` line. A warm-start for all 400 tasks: same warmstart command without `--subset`, `--out runs/warmstart-rearc/full`, and `--config configs/ttt_rearc.yaml`.

## Instance B (partner)

Instance B first needs its own baseline (34 min, same seed as A's):
`python -m arc_ttt.baseline.train --config configs/baseline.yaml`

| Run | Command | Time | Output |
|---|---|---|---|
| **R5** long budget (dev) | `CONFIG=configs/ttt_long.yaml OUT=runs/ttt-0.5b/dev-long bash scripts/run_ttt.sh` | 4 h | `runs/ttt-0.5b/dev-long` |
| **R5** learning rate × warmup (dev) | `for c in lr3e-4 warmup lr3e-4_warmup; do CONFIG=configs/ttt_$c.yaml OUT=runs/ttt-0.5b/dev-$c bash scripts/run_ttt.sh; done` | 6 h | `runs/ttt-0.5b/dev-<c>` |
| **R4** adaptive inference, 16 views logged (dev) | `CONFIG=configs/ttt_adaptive_infer.yaml OUT=runs/ttt-0.5b/dev-adaptive-infer bash scripts/run_ttt.sh` | 5 h | `runs/ttt-0.5b/dev-adaptive-infer` |
| **R7** ablations (dev) | `bash scripts/run_ablations.sh` | 8 h | `runs/ablations/<name>` |

## After Check-in 2: learned early predictor + weight averaging

All three use the final setup (re-arc baseline + constrained decoding + size filter), so they need
`runs/baseline-rearc-0.5b/adapter` (on Avi's instance).

| Run | Command | Time | Output |
|---|---|---|---|
| **E1** early checkpoints (0,4,8,16,32,64), all 400 | `FULL=1 CONFIG=configs/ttt_early.yaml OUT=runs/ttt-rearc-0.5b/full-early bash scripts/run_ttt.sh` | ~16 h (or 2 instances: add `SHARD=0 NUM_SHARDS=2` / `SHARD=1 NUM_SHARDS=2`, ~8 h each) | `runs/ttt-rearc-0.5b/full-early` |
| **W1** SWA weight averaging (dev) | `CONFIG=configs/ttt_swa.yaml OUT=runs/ttt-rearc-0.5b/dev-swa bash scripts/run_ttt.sh` | ~5.5 h | `runs/ttt-rearc-0.5b/dev-swa` |
| **W2** EMA weight averaging (dev) | `CONFIG=configs/ttt_ema.yaml OUT=runs/ttt-rearc-0.5b/dev-ema bash scripts/run_ttt.sh` | ~5.5 h | `runs/ttt-rearc-0.5b/dev-ema` |

Analysis (laptop):
- `python -m arc_ttt.ttt.early <E1 folder>`: can a learned predictor tell early which puzzles will be solved, and does that save compute? (`pip install scikit-learn` adds boosted trees)
- `python -m arc_ttt.ttt.wavg <W1 folder>`: averaged vs live weights, paired, with recovered "forgotten" puzzles
- `python -m arc_ttt.ttt.compare live=<W1> swa=<W1>@avg`: on the accuracy-vs-compute plot

## Final runs on the partner's account (Nov 21 on)

One script runs everything (~26 h on a g5.xlarge or g6.xlarge, ~$27) and shuts the instance down:

| Step | What | Time | Output |
|---|---|---|---|
| Timing | answer cost with 1/2/4/8 views (no training, 30 dev puzzles) | ~40 min | `runs/timing/views<k>` |
| **ConceptARC** | 160 unseen puzzles, re-arc + EMA, answers at 0/16/64: frozen triage test | ~10-12 h | `runs/conceptarc/final` |
| **All 400 with EMA** | EMA on the full set + a second early-checkpoint run | ~14 h | `runs/ttt-rearc-0.5b/full-ema` |

Setup on a fresh instance (same AMI as before: Deep Learning OSS Nvidia Driver AMI GPU PyTorch, Ubuntu):
```bash
git clone https://github.com/avijithk/arc-agi.git && cd arc-agi && bash scripts/setup_aws.sh
mkdir -p runs/baseline-rearc-0.5b      # then copy the adapter folder in (see below)
tmux new -s run
bash scripts/run_final_partner.sh 2>&1 | tee -a runs/final.log
```
The re-arc adapter goes to `runs/baseline-rearc-0.5b/adapter/` (adapter_config.json + adapter_model.safetensors).
Progress: `wc -l runs/timing/*/ttt.shard0.jsonl runs/conceptarc/final/ttt.shard0.jsonl runs/ttt-rearc-0.5b/full-ema/ttt.shard0.jsonl; tail -n 2 runs/final.log`

Analysis (laptop):
- `python -m arc_ttt.ttt.triage fit results/final/full-early --out results/final/triage.json` (predictor frozen on ARC eval)
- `python -m arc_ttt.ttt.triage eval results/final/conceptarc --predictor results/final/triage.json` (unseen puzzles)
- `python -m arc_ttt.ttt.triage eval <run>` (fit and threshold on half the puzzles, scored on the other half)
- `python -m arc_ttt.ttt.wavg <run>` (EMA vs live weights)

## Final runs (after Check-in 2)

| Run | Command | Time |
|---|---|---|
| **F1** live adaptive, all 400 | `python -m arc_ttt.ttt.run --config configs/ttt_final.yaml --stop-rule <rule> --out-dir runs/final/adaptive` then `python -m arc_ttt.ttt.analyze runs/final/adaptive` | 5 h |
| **F2** fixed budget, same settings | `CONFIG=configs/ttt_final.yaml OUT=runs/final/fixed FULL=1 bash scripts/run_ttt.sh` | 5 h |

Stop rules: `plateau:0.05`, `rel:0.05`, `best:loo_loss`, `best:holdout_loss` (holdout needs `--holdout`).
For live adaptive *inference* instead, add `adaptive_infer: {batch: 2, min_votes: 3}` under `ttt:` in the config.

## Optional (only if time allows)

| Run | Command |
|---|---|
| Sampled candidates (dev) | `CONFIG=configs/ttt_sampled.yaml OUT=runs/ttt-0.5b/dev-sampled bash scripts/run_ttt.sh` |
| Qwen3-0.6B (dev) | `python -m arc_ttt.baseline.train --config configs/baseline_qwen3.yaml && CONFIG=configs/ttt_qwen3.yaml OUT=runs/ttt-qwen3/dev bash scripts/run_ttt.sh` |

## Analysis (no GPU; run on the laptop after copying results)

| Question | Command |
|---|---|
| One run's summary | `python -m arc_ttt.ttt.analyze <run folder>` |
| Same, only the dev puzzles of a full run | `python -m arc_ttt.ttt.analyze runs/ttt-0.5b/full --subset configs/dev_subset.txt` |
| Several runs side by side + combined plot | `python -m arc_ttt.ttt.compare base=runs/ttt-0.5b/full constrained=runs/ttt-0.5b/dev-constrained ... --subset configs/dev_subset.txt --out results/compare` |
| ConceptARC by concept | add `--by-concept` to `compare` |
| Do stopping rules survive on unseen puzzles? | `python -m arc_ttt.ttt.validate <run folder> --method poe` |
| Budget allocation / reallocation (use R5's 128-step logs for reallocation) | `python -m arc_ttt.ttt.allocate <run folder> --low 8` |

Copy a run to the laptop (Mac Terminal): `scp -i ~/.ssh/arc-agi-2.pem -r ubuntu@<DNS>:arc-agi/runs/<folder> ~/Desktop/Projects/arc-agi/results/`
(skip `checkpoint/` and `adapter/` folders if you only need results: they are large).
