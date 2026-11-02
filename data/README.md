# Data

Not tracked in git. Run `bash scripts/download_data.sh` to fetch ARC-AGI-1 into `data/training/` and `data/evaluation/`.

Each task is a JSON file: `{"train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid, "output": grid}]}`, where a grid is a list of rows of ints 0–9 (colors), up to 30×30.

The fixed 100-puzzle dev subset is listed in `configs/dev_subset.txt` (generate with `python -m arc_ttt.data --make-dev-subset`).
