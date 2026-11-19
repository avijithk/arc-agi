import torch

from arc_ttt.ttt.core import TestTimeTrainer
from arc_ttt.ttt.early import _test_feats, features
from arc_ttt.ttt.wavg import avg_view, mcnemar


class FakeTrainer:
    _update_avg = TestTimeTrainer._update_avg
    _load_avg = TestTimeTrainer._load_avg
    _restore = TestTimeTrainer._restore

    def __init__(self):
        self.params = [torch.nn.Parameter(torch.zeros(3))]
        self._avg, self._n_avg = None, 0


def test_swa_is_uniform_mean_and_restore_is_exact():
    t = FakeTrainer()
    for v in (1.0, 2.0, 6.0):
        with torch.no_grad():
            t.params[0].fill_(v)
        t._update_avg({"kind": "swa"})
    assert t._n_avg == 3
    live = t._load_avg()
    assert torch.allclose(t.params[0], torch.full((3,), 3.0))
    t._restore(live)
    assert torch.allclose(t.params[0], torch.full((3,), 6.0))
    assert torch.allclose(t._avg[0], torch.full((3,), 3.0))     # average untouched by the swap


def test_ema():
    t = FakeTrainer()
    for v in (0.0, 10.0):
        with torch.no_grad():
            t.params[0].fill_(v)
        t._update_avg({"kind": "ema", "decay": 0.9})
    assert torch.allclose(t._avg[0], torch.full((3,), 1.0))


def test_mcnemar():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 5) == 1.0
    assert mcnemar(10, 0) < 0.01


def _te(aug_cand, size_ok, lp, vote):
    return {"aug_cand": aug_cand, "unique": [[[1]], [[2]]], "size_ok": size_ok, "poe_lp": lp,
            "vote": vote, "expected_shape": [1, 1]}


def test_avg_view_and_features():
    te = _te([0, 0, 1, -1], [True, False], {"0": [-1.0, -1.0], "1": [-3.0, None]}, [[[1]]])
    rec = {"task": "x", "n_pairs": 3, "n_train_samples": 10, "targets": [[[1]]],
           "signals": [{"step": 0, "loo_loss": 1.0}, {"step": 4, "loo_loss": 0.5}],
           "checkpoints": [{"step": 0, "tests": [te], "train_seconds": 0, "predict_seconds": 1},
                           {"step": 4, "tests": [te], "train_seconds": 1, "predict_seconds": 1,
                            "avg": {"step": 4, "tests": [te], "train_seconds": 1, "predict_seconds": 1}}]}
    v = avg_view([rec])
    assert len(v) == 1 and [c["step"] for c in v[0]["checkpoints"]] == [4]
    agree, uniq, valid, size_ok, conf, margin = _test_feats(te)
    assert agree == 0.5 and uniq == 0.5 and valid == 0.75 and size_ok == 1.0
    assert conf == -1.0 and margin == 2.0
    f = features(rec, 4, 0)
    assert len(f["free"]) < len(f["loss"]) < len(f["pred"])
    assert f["pred"][-1] == 1.0          # top vote unchanged since step 0
