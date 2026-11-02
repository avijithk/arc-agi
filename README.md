# Adaptive Test-Time Training for ARC-AGI

10-417/617 (Intro to Deep Learning, Fall 2026) course project — CMU.

ARC-AGI puzzles show a few input→output grid examples and ask the model to infer the rule and apply it to a new grid. Every evaluation puzzle uses an unseen rule. We fine-tune a small open-source LLM on serialized grids, add per-puzzle test-time training (TTT), and then study **adaptive** TTT: a held-out demo example decides how long each puzzle trains, and the final answer is picked by how consistently the model scores it across augmentations, not just by vote count.

## Plan

| Stage | What |
|---|---|
| Baseline | Grids → text, LoRA fine-tune a 0.5–1.5B LLM on ARC training + synthetic puzzles, predict directly |
| TTT | Per puzzle: fresh LoRA adapter on its demos (+ rotations, flips, recolorings), vote across augmented predictions |
| Ours | Adaptive TTT budget via held-out demo; augmentation-consistency scoring instead of plain voting |

**Evaluation:** official ARC score (exact match, 2 attempts) on the 400 public eval puzzles; 100-puzzle fixed dev subset while iterating. Also cell accuracy, size-match rate, ablations, accuracy vs. GPU-time, error analysis by puzzle type.

## Repo layout

```
src/arc_ttt/      data loading, grid serialization, augmentations, metrics
  data.py         load ARC tasks, grid <-> text
  augment.py      D8 symmetries + color permutations (and their inverses)
  metrics.py      pass@2, cell accuracy, size match
  baseline/       baseline LoRA fine-tuning          (Person A)
  ttt/            per-puzzle TTT, voting, adaptive    (Person B)
configs/          experiment configs (YAML)
scripts/          download data, run jobs
notebooks/        exploration and figures
tests/            unit tests (pytest)
report/           LaTeX for midway + final reports (NeurIPS format)
docs/             proposal and course requirements
data/, checkpoints/   git-ignored
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
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
