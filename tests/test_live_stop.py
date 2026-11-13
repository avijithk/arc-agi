from arc_ttt.ttt.core import TestTimeTrainer


class Fake:
    """Exercise the decision logic without a model."""
    cfg = {"predict_at": [0, 8, 16, 32, 64]}
    params = []
    _decide = TestTimeTrainer._decide
    _track_best = TestTimeTrainer._track_best


def sig(step, loo):
    return {"step": step, "loo_loss": loo}


def test_parse():
    assert TestTimeTrainer.parse_rule(None) is None
    assert TestTimeTrainer.parse_rule("plateau:0.05") == {"kind": "plateau", "delta": 0.05}
    assert TestTimeTrainer.parse_rule("best:holdout_loss")["tol"] == 0.0


def test_plateau_and_rel():
    f = Fake()
    sigs = [sig(0, 1.0), sig(8, 0.5), sig(16, 0.49)]
    assert f._decide({"kind": "plateau", "delta": 0.05}, sigs[:2], 8, 64) == (False, False)
    assert f._decide({"kind": "plateau", "delta": 0.05}, sigs, 16, 64) == (True, False)
    assert f._decide({"kind": "rel", "frac": 0.6}, sigs[:2], 8, 64) == (True, False)
    assert f._decide({"kind": "rel", "frac": 0.4}, sigs[:2], 8, 64) == (False, False)
    assert f._decide({"kind": "rel", "frac": 0.4}, sigs, 64, 64)[0] is True    # budget exhausted


def test_best_tracks_and_stops():
    f, rule = Fake(), {"kind": "best", "key": "loo_loss", "tol": 0.0}
    assert f._track_best(rule, sig(0, 1.0), 0) is True
    assert f._track_best(rule, sig(8, 0.7), 8) is True and rule["best_step"] == 8
    assert f._track_best(rule, sig(16, 0.8), 16) is False      # got worse -> stop, use step 8
    assert rule["best_step"] == 8
