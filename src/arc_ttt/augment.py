"""Invertible augmentations: the 8 dihedral symmetries (D8) and color permutations.

Each augmentation is applied to every grid in a task; predictions made on an
augmented task are mapped back with `invert` before voting/scoring.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

Grid = list[list[int]]


def _d8(a: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rotate by 90*(k%4) degrees, then transpose if k >= 4."""
    a = np.rot90(a, k % 4)
    return a.T if k >= 4 else a


def _d8_inv(a: np.ndarray, k: int) -> np.ndarray:
    if k >= 4:
        a = a.T
    return np.rot90(a, -(k % 4))


@dataclass(frozen=True)
class Augmentation:
    d8: int = 0                                   # 0 = identity
    colors: tuple[int, ...] = tuple(range(10))    # colors[c] = new color for c

    def apply(self, grid: Grid) -> Grid:
        a = np.asarray(grid)
        a = np.asarray(self.colors)[a]
        return _d8(a, self.d8).tolist()

    def invert(self, grid: Grid) -> Grid:
        a = _d8_inv(np.asarray(grid), self.d8)
        inv = np.argsort(self.colors)
        return inv[a].tolist()

    def apply_task(self, task: dict) -> dict:
        def pair(ex):
            out = {"input": self.apply(ex["input"])}
            if "output" in ex:
                out["output"] = self.apply(ex["output"])
            return out
        return {**task, "train": [pair(e) for e in task["train"]], "test": [pair(e) for e in task["test"]]}


def random_color_perm(rng: random.Random, keep_background: bool = True) -> tuple[int, ...]:
    """Random permutation of colors 0-9; by default color 0 (black background) stays fixed."""
    rest = list(range(1, 10)) if keep_background else list(range(10))
    rng.shuffle(rest)
    return tuple([0] + rest) if keep_background else tuple(rest)


def sample_augmentations(n: int, seed: int = 0, include_identity: bool = True) -> list[Augmentation]:
    rng = random.Random(seed)
    augs = [Augmentation()] if include_identity else []
    while len(augs) < n:
        augs.append(Augmentation(d8=rng.randrange(8), colors=random_color_perm(rng)))
    return augs
