# Project plan

Adaptive Test-Time Training for ARC-AGI · 10-417/617 · as of Nov 2, 2026

Person A = data and baseline. Person B = test-time training and the adaptive method. Ablations, analysis and writing are shared.

## Contribution

- **Headline:** adaptive per-puzzle TTT budget. A stopping signal from the held-out (or leave-one-out) demo loss decides how many steps each puzzle gets. Claim: same accuracy as fixed-budget TTT, less GPU time.
- **Component, cited:** product-of-experts scoring across augmentations ([Franzen et al., ICML 2025](https://arxiv.org/abs/2505.07859)). Already published, so not claimed as new.
- **Side result:** whether held-out demo performance predicts solving the puzzle (reportable either way).
- **Motivation:** TTT budget strongly affects ARC accuracy and dominates compute, but all published systems use a fixed budget ([budget study, Oct 2026](https://arxiv.org/abs/2610.13660)).

## Compute

- AWS g5.xlarge (A10G 24 GB, bf16) on two accounts, $100 credits each (not pooled), Paid plan with a spend limit around $100. About $1.01/hr on-demand, ~$0.49/hr spot → roughly 300 usable spot GPU-hours.
- Stop instances after every run. Predictions save per puzzle; training checkpoints every 50 steps; everything resumes.

## Check-in 1 — Nov 3/4 (1%)

Goal: a working pipeline and the sharpened contribution, not strong numbers.

- A: data, serialization, augmentations, tests ✅; dev subset fixed ✅; baseline train/eval code ✅; first run started, numbers if it finishes
- B: per-puzzle TTT loop running on a few puzzles; plan for the stopping signal and PoE scoring
- Tell the TA: contribution update (PoE is prior work → adaptive budget is the headline); compute setup
- Ask: is "same accuracy for less compute" strong enough? Is a 100-puzzle dev subset enough? Which ablations matter most?
- After: Gradescope summary (goal shifts + steps to the Midway)

## Midway Report — Nov 11 (3%, ≤3 pages, NeurIPS)

Experiments (Nov 4–10):
- A: baseline on all 400 eval puzzles; loss curves
- B: fixed-budget TTT + voting on the dev subset, then all 400
- B: PoE scoring vs. voting on the dev subset
- B: log held-out demo loss and per-puzzle accuracy in every TTT run (the adaptive method's training data)
- A: accuracy by puzzle size and output-size change

| Section | Pts | Owner |
|---|---|---|
| Title | 1 | A |
| Introduction | 2 | A |
| Literature review (Chollet 2019; Akyürek et al. 2024; Franzen et al. 2025; Oct 2026 budget study) | 3 | B |
| Methods/Model | 4 | A + B |
| Preliminary results | 4 | A |
| Evaluation of preliminary work | 2 | B |
| Future work (adaptive budget plan) | 2 | B |
| Timeline and division of labor | 1 | A |
| References (biblatex, alphabetic) | 1 | B |

Drafts Nov 9, swap and edit Nov 10, submit Nov 11.

## Check-in 2 — Nov 24 (1%)

- B: stopping signal checked every k steps; compare stopping rules (loss threshold, plateau, held-out exact match); PoE integrated; dev result: adaptive vs. fixed at matched GPU-minutes
- A: dev-subset ablations (synthetic data, augmentations, voting vs. PoE, LoRA rank 8/32/128); fixed-budget sweep (16/32/64/128 steps) for the accuracy-vs-compute curve; 1.5B on the dev subset; GPU-time logging everywhere
- Both: first answer on whether the held-out signal predicts success; bring the accuracy-vs-compute plot; Gradescope summary

## Final Report — Dec 2 (20%, ≤8 pages, NeurIPS)

- Final runs (Nov 25–28): full 400 for baseline, fixed TTT, adaptive TTT (0.5B); best fixed + adaptive at 1.5B
- Figures (Nov 25–30): main results table; accuracy vs. GPU-time curve with the adaptive points; steps spent per puzzle (solved vs. unsolved); held-out signal calibration; error analysis by puzzle type; loss curves
- Writing: Intro (A, Nov 29), Lit review (B, Nov 29), Methods (A+B, Nov 30), Results (A, Nov 30), Discussion (B, Dec 1), title/contribution statement/references (A, Dec 1); read-through and cut to 8 pages Dec 1; submit Dec 2

## Experiment budget (A10G GPU-hours, rough)

| Experiment | Runs | GPU-h |
|---|---|---|
| Baseline fine-tune 0.5B / 1.5B | 1–2 / 1 | 8 |
| Full eval 0.5B (baseline, fixed TTT, adaptive TTT) | 5 | 35 |
| Full eval 1.5B (best fixed + adaptive) | 2 | 40 |
| Dev-subset runs (ablations, budget sweep, stopping rules) | ~50 | 100 |
| Synthetic-data fine-tune (re-arc) | 1 | 25 |
| **Total** | | **~210 of ~300** |

Cut order if short: synthetic-data fine-tune, then 1.5B full evals.

## Add-ons if the TA wants more scope

| Add-on | Build time | Extra GPU-h | Notes |
|---|---|---|---|
| 1. Reallocate a fixed total budget (steps from easy puzzles go to hard ones) | 2–3 days | ~15 | Turns the claim into "more accuracy, same compute". Lead with this. |
| 2. Predict each puzzle's budget up front (grid size, #demos, initial loss) | 1–2 days | ~0 | Uses already-logged runs. |
| 3. Adaptive inference (number of augmentations per puzzle) | 1–2 days | ~10 | Save dev adapters once, sweep 2/4/8/16. |
| 4. Transfer to ConceptARC | 1 day | ~10 | Eval only. |
| 5. Does better pretraining (re-arc) reduce the TTT needed? | 1–2 days | ~25 | Builds on the budget study. |
| 6. Budget vs. model size (0.5B/1.5B/3B) | 1 day | 40+ | Dev subset only. |
| 7. Meta-learned adapter init (Reptile) | 4–6 days | ~30–45 | Riskiest; replaces 1.5B full evals rather than adding to them. |

Recommended if asked: 1 + 3 (about 25 GPU-h, 3–5 days, fits Nov 12–24).

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| g5 capacity unavailable in a zone | Try another availability zone, or g6.xlarge (L4 24 GB) |
| Spot reclaim / session cap | Per-puzzle saves, checkpoints, `--time-limit-hours`; rerun resumes |
| Held-out loss doesn't predict success | Report the calibration as a finding; try leave-one-out loss or cross-augmentation agreement |
| Adaptive saves little compute | Frame as analysis of budget vs. difficulty; the fixed-budget sweep still stands |
| Holding out a demo hurts (2–3 demos per puzzle) | Use the leave-one-out loss TTT already computes |
| Long puzzles (>8k tokens) | Demos dropped to fit; skipped count reported |
