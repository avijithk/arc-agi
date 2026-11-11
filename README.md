# Adaptive Test-Time Training for ARC-AGI

10-417/617 (Intro to Deep Learning, Fall 2026) course project — CMU.

ARC-AGI puzzles show a few input→output grid examples and ask the model to infer the rule and apply it to a new grid. Every evaluation puzzle uses an unseen rule. We fine-tune a small LLM on serialized grids, add per-puzzle test-time training (TTT), and study **adaptive TTT budgets**: the model's performance on one of the puzzle's own examples decides how long each puzzle trains, so easy puzzles stop early.

**Claim:** adaptive TTT matches fixed-budget TTT accuracy with less GPU time. Published TTT systems use a fixed budget per puzzle, even though the budget drives accuracy and dominates compute.

## Plan

| Stage | What |
|---|---|
| Baseline | Grids → text, LoRA fine-tune Qwen2.5-0.5B on ARC training puzzles (+ augmentations), predict directly |
| TTT | Per puzzle: fresh LoRA adapter on its demos (+ rotations, flips, recolorings), combine predictions across augmentations |
| Ours (headline) | Adaptive per-puzzle TTT budget from the held-out / leave-one-out demo loss |
| Component (prior work) | Product-of-experts scoring across augmentations ([Franzen et al., ICML 2025](https://arxiv.org/abs/2505.07859)), cited, not claimed |
| Side result | Does held-out demo performance predict solving the puzzle? |

**Evaluation:** official ARC score (exact match, 2 attempts) on the 400 public eval puzzles; fixed 100-puzzle dev subset (`configs/dev_subset.txt`) for iterating. Also cell accuracy, size match, accuracy vs. GPU-minutes per puzzle, ablations, error analysis by puzzle type.

Full checkpoint-by-checkpoint plan, compute budget, risks and possible add-ons: [`docs/plan.md`](docs/plan.md).

## Repo layout

```
src/arc_ttt/      data loading, grid serialization, augmentations, metrics
  data.py         load ARC tasks, grid <-> text
  augment.py      D8 symmetries + color permutations (and their inverses)
  metrics.py      pass@2, cell accuracy, size match
  model_utils.py  model loading, prompt tokenization, greedy grid prediction
  baseline/       train.py + evaluate.py              (Person A)
  ttt/            core.py, predict.py, run.py, analyze.py (Person B)
configs/          experiment configs (YAML)
scripts/          download data, run jobs
notebooks/        exploration and figures
tests/            unit tests (pytest)
report/           LaTeX for midway + final reports (NeurIPS format)
docs/             proposal and course requirements
data/, checkpoints/   git-ignored
```

## Running on AWS (g5.xlarge, A10G 24 GB; needs the Paid plan, the Free plan has no GPU instances)

1. Launch a **g5.xlarge** with a **Deep Learning AMI (PyTorch, Ubuntu)**, 60 GB gp3 disk. Spot is about half price.
2. SSH in, clone, set up (one time):
   ```bash
   git clone https://github.com/avijithk/arc-agi.git && cd arc-agi
   bash scripts/setup_aws.sh          # installs deps, downloads ARC data, caches Qwen2.5-0.5B, runs tests
   ```
   GitHub will ask for a password: use a personal access token (Settings → Developer settings → Tokens).
3. Run inside tmux so it survives disconnects:
   ```bash
   tmux new -s run
   bash scripts/run_midway.sh A       # or B on the other account; see "Midway runs" below
   # Ctrl-b d to detach; tmux attach -t run to come back
   ```
4. Results land in `runs/baseline-0.5b/`: `train_log.jsonl` (loss curve), `eval-dev/metrics.json`, `eval-dev/predictions.shard0.jsonl`.
5. **Stop the instance** when the run finishes.

If the instance dies (spot reclaim, crash), rerun the same command: training resumes from `runs/.../checkpoint`, evaluation skips finished puzzles. Copy results off before terminating an instance, e.g. `scp -r ubuntu@<ip>:arc-agi/runs/baseline-0.5b/eval-dev .`

Useful flags: `--time-limit-hours H` on train/evaluate checkpoints and exits cleanly; `FULL=1 bash scripts/run_baseline.sh` also evaluates all 400 eval puzzles; `python -m arc_ttt.baseline.evaluate --limit 5 ...` for a quick check; `--max-steps N` on training for a short test run.

## Test-time training

```bash
bash scripts/run_ttt.sh                  # dev subset, TTT on top of runs/baseline-0.5b/adapter
FULL=1 bash scripts/run_ttt.sh           # all 400 eval puzzles
HOLDOUT=1 bash scripts/run_ttt.sh        # hold one demo pair out to log the stopping signal
```

Each puzzle gets a fresh LoRA adapter trained on leave-one-out examples from its own demos (x16 augmentations), at a constant learning rate. Predictions (vote and product-of-experts top-2 over 8 augmentations) are logged at steps 0, 8, 16, 32, 64, and stopping signals every 4 steps. So **one run gives the whole fixed-budget sweep**, and adaptive stopping rules are evaluated offline from the logs:

```bash
python -m arc_ttt.ttt.analyze runs/ttt-0.5b/dev     # -> analysis/analysis.md, accuracy_vs_compute.png
```

Step 0 is the baseline with augmentation voting / PoE and no TTT.

## Block 1 additions (Nov 11)

| Feature | How to use it |
|---|---|
| Constrained decoding (only well-formed grids) | `constrained_decoding: true` in a TTT config; `--constrained` on `arc_ttt.baseline.evaluate` |
| Size filter (drop candidates whose size breaks the demos' pattern) | `size_filter: true`; verdicts are always logged, so analyze replays it on/off |
| Full candidate logging + PoE log-probs per augmentation | automatic in every TTT run |
| Adaptive inference | log many candidates (`configs/ttt_adaptive_infer.yaml`), analyze replays stop-on-agreement settings; live mode `adaptive_infer: {batch: 2, min_votes: 3}` |
| TTT learning-rate warmup | `warmup_steps: 4` |
| re-arc data | `bash scripts/download_rearc.sh && python -m arc_ttt.data_gen.rearc --per-task 100`; train with `configs/baseline_rearc.yaml` |
| Shared warm-start on eval demos | `python -m arc_ttt.ttt.warmstart --out runs/warmstart-0.5b/dev --subset configs/dev_subset.txt`; then `configs/ttt_warmstart.yaml` |
| Run configs | `configs/ttt_{constrained,warmstart,rearc,adaptive_infer,long,lr3e-4,warmup,lr3e-4_warmup,1.5b}.yaml`, `configs/baseline_{rearc,1.5b}.yaml`, `configs/ablations/*.yaml` |
| Ablations | `bash scripts/run_ablations.sh` |
| Analysis on a subset of a run | `python -m arc_ttt.ttt.analyze runs/ttt-0.5b/full --subset configs/dev_subset.txt` |

## Midway runs (two accounts in parallel)

| Instance | Command | What it produces | Rough time (A10G) |
|---|---|---|---|
| A | `bash scripts/run_midway.sh A` | baseline: train, dev + all 400; then TTT sweep on the dev subset | ~5–7 h |
| B | `bash scripts/run_midway.sh B` | baseline train, then TTT sweep on all 400 | ~10–14 h |

Both train the same baseline (same seed), so nothing has to be copied between accounts. Times are estimates; check the per-puzzle seconds in the first log lines.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
bash scripts/download_data.sh      # ARC-AGI-1 training + evaluation JSON into data/
pytest -q
```

## Deadlines

| Due | Deliverable | Weight |
|---|---|---|
| Oct 26 | Proposal ✅ | 0% |
| Nov 4 | Check-in 1 — baseline numbers (A), TTT started (B) | 1% |
| Nov 11 | Midway Report (≤3 pp, NeurIPS) — baseline + TTT on full eval | 3% |
| Nov 24 | Check-in 2 — adaptive TTT + scoring (B), ablations + compute (A) | 1% |
| Dec 2 | Final Report (≤8 pp, NeurIPS) | 20% |

Citations use the alphabetic BibLaTeX style. Page limits are strict.

## Division of labor

- **A — data & baseline:** loading, serialization, augmentation code, baseline training, eval script, ablations, compute comparison.
- **B — test-time side:** per-puzzle training loop, voting and consistency scoring, adaptive training logic.
- **Shared:** ablations, analysis, writing.

## Workflow

Branch off `main` for each piece of work (`a/baseline-train`, `b/ttt-loop`, …), open a PR, and have the other person review. Don't commit data, checkpoints, or W&B runs.
