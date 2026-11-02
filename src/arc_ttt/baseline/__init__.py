"""Baseline (Person A): LoRA fine-tuning on serialized ARC training tasks.

train.py     -> python -m arc_ttt.baseline.train --config configs/baseline.yaml
evaluate.py  -> python -m arc_ttt.baseline.evaluate --adapter runs/baseline-0.5b/adapter --subset configs/dev_subset.txt --out-dir ...
"""
