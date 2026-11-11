from arc_ttt.ttt.analyze import inference_replay
from arc_ttt.ttt.predict import rank_from_log, size_rule

A, B, C = [[1, 1]], [[2, 2]], [[3]]          # A, B are 1x2; C is 1x1


def demo(i, o):
    return {"input": i, "output": o}


def test_size_rules():
    same = size_rule([demo([[1, 2]], [[2, 1]]), demo([[1], [2]], [[3], [4]])])
    assert same([[0, 0, 0]]) == (1, 3)
    const = size_rule([demo([[1, 2]], [[5]]), demo([[1], [2], [3]], [[6]])])
    assert const([[0] * 7]) == (1, 1)
    up = size_rule([demo([[1]], [[1, 1], [1, 1]]), demo([[1, 2]], [[1, 1, 2, 2], [1, 1, 2, 2]])])
    assert up([[0, 0, 0]]) == (2, 6)
    assert size_rule([demo([[1]], [[1, 1]]), demo([[1]], [[1], [1]])]) is None


def _te(order, size_ok, poe_lp=None):
    return {"unique": [A, B, C], "aug_cand": order, "size_ok": size_ok, "poe_lp": poe_lp or {},
            "batch_seconds": [1.0] * (len(order) // 2), "poe_seconds": 0.0}


def test_rank_from_log_prefix_and_filter():
    te = _te([2, 2, 0, 1, 0, 2, -1, 0], [True, True, False],
             {"0": [-1] * 8, "1": [-5] * 8, "2": [-0.5] * 8})
    assert rank_from_log(te, 2)[0] == [C]                    # first 2 augs: only C
    assert rank_from_log(te)[0] == [C, A]                    # 3 votes each: first seen (C) ranks first
    assert rank_from_log(te, size_filter=True)[0] == [A, B]  # C has the wrong size
    assert rank_from_log(te)[1] == [C, A]                    # PoE: C scores highest
    assert rank_from_log(te, size_filter=True)[1] == [A, B]


def test_replay_adaptive_stops_early():
    rec = {"task": "t", "targets": [A], "holdout_index": None, "signals": [],
           "checkpoints": [{"step": 64, "train_seconds": 10.0, "predict_seconds": 4.0,
                            "tests": [_te([0, 0, 0, 1, 2, 2, 2, 2], [True, True, True])]}]}
    rows = inference_replay([rec])
    get = lambda name, m="vote", f=False: next(r for r in rows if r["setting"] == name and r["method"] == m and r["size_filter"] == f)
    full, stop = get("8 augs"), get("batches of 2, stop at 2 agreeing")
    assert full["mean_augs"] == 8 and stop["mean_augs"] == 2       # A,A agree after the first batch
    assert full["arc_score"] == 1.0 and stop["arc_score"] == 1.0   # A is in the top 2 either way
    assert stop["infer_seconds"] == 1.0 and full["infer_seconds"] == 4.0
