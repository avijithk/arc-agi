import numpy as np

from arc_ttt.ttt.triage import fit, hull, hull_at, predict


def _rec(task, solved_steps, n=3):
    te_ok = {"aug_cand": [0], "unique": [[[1]]], "size_ok": [True], "expected_shape": [1, 1],
             "vote": [[[1]]], "poe": [[[1]]], "poe_lp": {"0": [-1.0]}}
    te_bad = dict(te_ok, vote=[[[2]]], poe=[[[2]]])
    cps = [{"step": s, "train_seconds": float(s), "predict_seconds": 10.0, "tests": [te_ok if s in solved_steps else te_bad]}
           for s in (0, 8, 64)]
    return {"task": task, "n_pairs": n, "n_train_samples": 16, "targets": [[[1]]],
            "signals": [{"step": 0, "loo_loss": 1.0, "signal_seconds": 0.1}], "checkpoints": cps}


def test_hull_includes_skipping_and_is_concave():
    recs = [_rec("a", {64}), _rec("b", set())]
    h = hull(recs)
    assert h[0] == (0.0, 0.0)
    assert abs(hull_at(h, h[-1][0]) - 0.5) < 1e-9          # half the puzzles solved at the full budget
    assert hull_at(h, h[-1][0] / 2) <= 0.25 + 1e-9        # a random half at the full budget


def test_fit_predict_separates():
    X = np.array([[0.0], [0.1], [1.0], [1.1]] * 5); y = np.array([0, 0, 1, 1] * 5, dtype=float)
    p = predict(fit(X, y), X)
    assert p[y == 1].min() > p[y == 0].max()
