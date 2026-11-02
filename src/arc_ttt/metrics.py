"""Evaluation metrics: official ARC score (exact match, top-2), cell accuracy, size match."""
from __future__ import annotations

Grid = list[list[int]]


def exact_match(pred: Grid | None, target: Grid) -> bool:
    return pred is not None and pred == target


def size_match(pred: Grid | None, target: Grid) -> bool:
    if not pred:
        return False
    return len(pred) == len(target) and all(len(r) == len(target[0]) for r in pred)


def cell_accuracy(pred: Grid | None, target: Grid) -> float:
    """Fraction of cells correct; 0 if the shape is wrong."""
    if not size_match(pred, target):
        return 0.0
    total = sum(len(r) for r in target)
    right = sum(p == t for pr, tr in zip(pred, target) for p, t in zip(pr, tr))
    return right / total


def task_score(attempts: list[list[Grid | None]], targets: list[Grid]) -> float:
    """ARC score for one task.

    attempts[i] holds up to 2 guesses for test input i. Test input i counts as
    solved if any guess matches exactly; the task score is the fraction of its
    test inputs solved (most tasks have exactly one).
    """
    solved = [any(exact_match(g, t) for g in guesses[:2]) for guesses, t in zip(attempts, targets)]
    return sum(solved) / len(targets)


def summarize(results: dict[str, dict]) -> dict[str, float]:
    """results[task_id] = {'attempts': [...], 'targets': [...]} -> aggregate metrics."""
    n = len(results)
    score = cell = size = 0.0
    for r in results.values():
        score += task_score(r["attempts"], r["targets"])
        firsts = [g[0] if g else None for g in r["attempts"]]
        cell += sum(cell_accuracy(p, t) for p, t in zip(firsts, r["targets"])) / len(r["targets"])
        size += sum(size_match(p, t) for p, t in zip(firsts, r["targets"])) / len(r["targets"])
    return {"arc_score": score / n, "cell_accuracy": cell / n, "size_match": size / n, "n_tasks": n}
