# Run guide (current as of Nov 12)

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
