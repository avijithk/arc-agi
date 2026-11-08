from arc_ttt.ttt.analyze import fixed_budget_table, simulate, auc
from arc_ttt.ttt.core import task_augs
from arc_ttt.ttt.predict import poe_top2, vote_top2
import json
import random

A, B, C = [[1]], [[2]], [[3]]


def test_vote_ranks_by_count_then_first_seen():
    assert vote_top2([B, A, A, None, C, B, A]) == [A, B]
    assert vote_top2([C, B]) == [C, B]          # tie -> first seen
    assert vote_top2([None, None]) == []


def test_poe_ranks_by_score():
    scores = {json.dumps(A): -5.0, json.dumps(B): -1.0, json.dumps(C): -3.0}
    assert poe_top2(scores) == [B, C]


def test_task_augs_identity_first_and_distinct():
    augs = task_augs(8, random.Random(0))
    assert augs[0].d8 == 0 and augs[0].colors == tuple(range(10))
    assert sorted(a.d8 for a in augs) == list(range(8))


def _rec(task, steps_correct, loo):
    """Synthetic TTT record: solved at the listed steps, loo_loss per step."""
    cps, sigs = [], []
    for i, (step, ok) in enumerate(steps_correct):
        cps.append({"step": step, "train_seconds": float(step), "predict_seconds": 1.0,
                    "tests": [{"vote": [A] if ok else [B], "poe": [A] if ok else [C]}]})
        sigs.append({"step": step, "loo_loss": loo[i], "signal_seconds": 0.0})
    return {"task": task, "checkpoints": cps, "signals": sigs, "targets": [A], "holdout_index": None}


def test_fixed_and_adaptive():
    recs = [_rec("easy", [(0, False), (8, True), (64, True)], [1.0, 0.1, 0.05]),
            _rec("hard", [(0, False), (8, False), (64, True)], [1.0, 0.8, 0.2])]
    rows = {r["budget"]: r for r in fixed_budget_table(recs, "vote")}
    assert rows[0]["arc_score"] == 0 and rows[8]["arc_score"] == 0.5 and rows[64]["arc_score"] == 1
    assert rows[64]["seconds_per_puzzle"] == 65.0
    # stop when loo_loss < 0.3: easy stops at 8, hard at 64 -> full accuracy, less compute
    sig = lambda r, s: next(x for x in r["signals"] if x["step"] == s)
    acc, secs, stop = simulate(recs, "vote", lambda r, p, cp: sig(r, cp["step"])["loo_loss"] < 0.3)
    assert acc == 1.0 and secs == (9 + 65) / 2 and stop == 36


def test_auc():
    assert auc([0.1, 0.2], [0.5, 0.9]) == 1.0
    assert auc([0.5], [0.5]) == 0.5
