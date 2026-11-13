# Project plan

Adaptive Test-Time Training for ARC-AGI · 10-417/617 · updated Nov 10, 2026

Person A = data and baseline. Person B = test-time training and the adaptive method. Ablations, analysis and writing are shared.
Commands for every step: see the ARC-AGI Runbook doc. GPU costs at on-demand g5.xlarge (~$1.01/h); spot is about half.

## Contribution

- **Headline:** adaptive per-puzzle compute for TTT: stop training once a puzzle looks learned, and stop generating candidates once they agree. Claim: same accuracy as fixed budgets with less GPU time.
- **Components, cited:** TTT (Akyürek et al. 2024), product-of-experts scoring (Franzen et al. 2025).
- **Side result:** whether the stopping signal predicts solving the puzzle.

## Results so far (400 eval puzzles, Qwen2.5-0.5B)

| Setup | Vote | PoE |
|---|---|---|
| Baseline, direct (2 attempts) | 3.1% | — |
| TTT step 0 (8 views) | 3.1% | 4.5% |
| TTT 32 steps | 12.4% | 15.5% |
| TTT 64 steps | 13.4% | 16.6% |
| Best adaptive rule (offline) | ties fixed budgets | ties fixed budgets |

Signal AUC 0.58 (step 0) → 0.76 (step 64). One prediction costs about as much as 64 training steps. Baseline size match only 60%; many direct predictions hit the length cap.

## Action items

| # | When | Item | Who | Status | Your time | GPU time | Cost |
|---|---|---|---|---|---|---|---|
| 1 | Oct 26 | Project proposal | A + B | ✅ | — | — | $0 |
| 2 | Nov 1–2 | Repo, private GitHub, partner invited, plan | A | ✅ | — | — | $0 |
| 3 | Nov 2 | Code: data, augmentations, metrics, tests | — | ✅ | — | — | $0 |
| 4 | Nov 2 | Code: baseline training + evaluation | — | ✅ | — | — | $0 |
| 5 | Nov 3–4 | Check-in 1 + Gradescope summary | A + B | ✅ | — | — | $0 |
| 6 | Nov 8 | Code: TTT, vote + PoE, signals, analysis, run scripts | — | ✅ | — | — | $0 |
| 7 | Nov 8–9 | AWS: Paid plan, spend limit, quota, instance, setup fixes | A | ✅ | ~2 h | — | $0 |
| 8 | Nov 9 | Baseline fine-tune (0.5B) | A | ✅ | — | 34 min | ~$0.60 |
| 9 | Nov 9–10 | TTT sweep, all 400 (steps 0–64) | A | ✅ | — | ~16.5 h | ~$17 |
| 10 | Nov 10 | Baseline eval, dev + all 400 | A | ✅ | — | ~2 h | ~$2 |
| 11 | Nov 10 | Results to laptop; Midway draft; runbook; readable analysis summary | A | ✅ | — | — | $0 |
| 12 | Nov 10 | **R1:** held-out-signal run (dev); stop instance after | A | ⏳ | 5 min | ~3.5 h | ~$3.50 |
| 13 | Nov 10 | Compile report on Overleaf, fill placeholders | A | | 30 min | — | $0 |
| 14 | This week | Partner: quota, instance, setup, spend limit (early cost controls off) | B | | 1 h | 10 min | ~$0.20 |
| 15 | Nov 11 | Add held-out results, read-through, **submit Midway** | A + B | | 2–3 h | — | $0 |
| 16 | Nov 11–12 | **New:** constrained decoding (digits/newline/end only, equal row lengths, ≤30 rows) in baseline + TTT prediction | B | | ½ day | — | $0 |
| 17 | Nov 11–12 | **New:** size filter on candidates (drop answers whose size breaks the demos' size pattern); offline replay on the existing full run | A | | 2–3 h | — | $0 |
| 18 | Nov 11–13 | Build re-arc pipeline + configs (baseline trained 3–4 epochs, per idea "train longer") | A | | 1 day | — | $0 |
| 19 | Nov 11–13 | Small configs: ablations, 1.5B, 128 steps, learning rate, **new:** lr warmup option | A + B | | ½ day | — | $0 |
| 20 | Nov 12 | **New R9a:** baseline + TTT on dev with constrained decoding (measures its effect) | B | | 15 min | ~3 h | ~$3 |
| 21 | Nov 12–13 | **New:** shared warm-start on the eval tasks' demonstration pairs (no test answers), before per-puzzle TTT | A | | ½ day | — | $0 |
| 22 | Nov 11–15 | Build adaptive inference | B | | 1–2 days | — | $0 |
| 23 | Nov 13–14 | **R2:** re-arc baseline fine-tune | A | | 15 min | ~6 h | ~$6 |
| 24 | Nov 14 | **New R9b:** warm-start on/off, TTT on dev | A | | 15 min | ~4 h | ~$4 |
| 25 | Nov 14–15 | **R3:** TTT on re-arc baseline, all 400 (with whichever of constrained decoding / warm-start won on dev) | A | | 15 min | ~7 h | ~$7 |
| 26 | Nov 15–16 | **R5:** 128 steps + learning rate **× warmup on/off** (dev) | B | | 15 min | ~6 h | ~$6 |
| 27 | Nov 15–17 | **R4:** adaptive inference experiments (dev) | B | | 15 min | ~5 h | ~$5 |
| 28 | Nov 15–20 | Build online early stopping + per-puzzle stopping rules | B | | 1½ days | — | $0 |
| 29 | Nov 16–18 | **R6:** 1.5B baseline + TTT (dev) | A | | 15 min | ~5 h | ~$5 |
| 30 | Nov 18–21 | **R7:** ablations (dev) | B | | 30 min | ~8 h | ~$8 |
| 31 | Nov 19–21 | ConceptARC loader + offline extras (budget predictor, reallocation) | A | | 1 day | — | $0 |
| 32 | Nov 21–22 | **R8:** ConceptARC, best setup | A | | 15 min | ~3 h | ~$3 |
| 33 | Nov 22–23 | Gather results, analyses, Check-in 2 plots | A + B | | 2–3 h | — | $0 |
| 34 | **Nov 24** | **Check-in 2** + Gradescope; pick final config | A + B | | 1–2 h | — | $0 |
| 35 | Nov 25–26 | **F1:** real adaptive run, all 400 | A | | 15 min | ~5 h | ~$5 |
| 36 | Nov 25–26 | **F2:** fixed-budget comparison, all 400 | B | | 15 min | ~5 h | ~$5 |
| 37 | Nov 26–28 | **F3:** TA requests, reruns | either | | varies | ~5 h | ~$5 |
| 38 | Nov 28 | Copy all results; stop both instances | A + B | | 30 min | — | $0 |
| 39 | Nov 29 | Write Intro + Literature review | A / B | | ½ day | — | $0 |
| 40 | Nov 30 | Write Methods + Results, figures | A + B | | 1 day | — | $0 |
| 41 | Dec 1 | Discussion, contribution statement, cut to 8 pages | A + B | | ½ day | — | $0 |
| 42 | **Dec 2** | **Submit final report**; terminate instances, delete volumes | A + B | | 30 min | — | $0 |

### Optional, only if time allows

| Item | Who | Your time | GPU time | Cost |
|---|---|---|---|---|
| More diverse candidates (sampling / top-k paths instead of greedy) | B | 1 day | ~3 h | ~$3 |
| Qwen3-0.6B in place of Qwen2.5-0.5B (dev) | A | ½ day | ~6 h | ~$6 |
| re-arc extra-large data (more examples per task) | A | — | 10–15 h | $10–15 |

### Totals

| | GPU time | Cost |
|---|---|---|
| Spent | ~19 h | ~$20 |
| Running (R1) | ~3.5 h | ~$3.50 |
| Remaining (required) | ~62 h | ~$62 (~$31 spot) |
| Optional | ~9–24 h | ~$9–24 |
| **Project total (required)** | **~85 h** | **~$86** of $220 combined credits |

Plus disk storage: ~$0.16/day per account while instances exist.

## Cut order if time runs short

1. Optional items
2. ConceptARC (R8)
3. 1.5B (R6)
4. Half the ablations (keep attention vs. MLP and no-augmentation)

Never cut: re-arc (R2/R3), adaptive inference (R4), constrained decoding (16/R9a), the real adaptive run (F1).
