"""Test-time training (Person B).

core.py     per-puzzle TTT: leave-one-out examples, adapter reset, stopping signals, checkpointed predictions
predict.py  batched generation under augmentations, voting, product-of-experts scoring
run.py      python -m arc_ttt.ttt.run --config configs/ttt.yaml --out-dir runs/ttt-0.5b/dev --subset configs/dev_subset.txt
analyze.py  python -m arc_ttt.ttt.analyze runs/ttt-0.5b/dev   (fixed-budget sweep, offline adaptive stopping, signal AUC, plot)
"""
