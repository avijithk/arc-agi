"""Test-time training (Person B).

TODO
- per task: fresh LoRA adapter, train on leave-one-out demo pairs + augmentations
- predict under N augmentations, invert, vote (top-2)
- ours: adaptive step budget from a held-out demo; augmentation-consistency scoring
"""
