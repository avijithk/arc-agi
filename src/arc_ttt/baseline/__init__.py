"""Baseline: LoRA fine-tuning on serialized ARC training + synthetic tasks (Person A).

TODO
- build (prompt, target) pairs with data.format_prompt over training tasks + augmentations
- LoRA fine-tune (peft) a 0.5-1.5B model, config in configs/baseline.yaml
- greedy decode on eval tasks -> text_to_grid -> metrics.summarize
"""
