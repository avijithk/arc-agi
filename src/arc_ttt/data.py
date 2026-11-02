"""Loading ARC tasks and converting grids to/from text."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

Grid = list[list[int]]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_task(path: str | Path) -> dict:
    with open(path) as f:
        task = json.load(f)
    task["id"] = Path(path).stem
    return task


def load_split(split: str = "training", data_dir: Path = DATA_DIR) -> list[dict]:
    """Load every task in data/<split>/ (split = 'training' or 'evaluation')."""
    return [load_task(p) for p in sorted((data_dir / split).glob("*.json"))]


def grid_to_text(grid: Grid) -> str:
    """One row per line, one digit per cell, e.g. [[1,0],[0,1]] -> '10\\n01'."""
    return "\n".join("".join(str(c) for c in row) for row in grid)


def text_to_grid(text: str) -> Grid | None:
    """Inverse of grid_to_text. Returns None if the text isn't a valid rectangular grid."""
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if not rows or any(not r.isdigit() for r in rows):
        return None
    if len({len(r) for r in rows}) != 1:
        return None
    return [[int(c) for c in r] for r in rows]


def format_prompt(train_pairs: list[dict], test_input: Grid) -> str:
    """Serialize demo pairs + test input into a single prompt string."""
    parts = []
    for i, ex in enumerate(train_pairs, 1):
        parts.append(f"Example {i}\nInput:\n{grid_to_text(ex['input'])}\nOutput:\n{grid_to_text(ex['output'])}")
    parts.append(f"Test\nInput:\n{grid_to_text(test_input)}\nOutput:\n")
    return "\n\n".join(parts)


def make_dev_subset(n: int = 100, seed: int = 0, out: Path = Path("configs/dev_subset.txt")) -> list[str]:
    ids = sorted(p.stem for p in (DATA_DIR / "evaluation").glob("*.json"))
    random.Random(seed).shuffle(ids)
    chosen = sorted(ids[:n])
    out.write_text("\n".join(chosen) + "\n")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-dev-subset", action="store_true")
    args = ap.parse_args()
    if args.make_dev_subset:
        print(f"wrote {len(make_dev_subset())} task ids to configs/dev_subset.txt")
