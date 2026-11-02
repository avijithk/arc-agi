from arc_ttt.model_utils import build_example


class CharTok:
    """Minimal stand-in tokenizer: one token per character."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


DEMO = {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}


def test_loss_only_on_output():
    ex = build_example(CharTok(), [DEMO], [[5, 6]], [[6, 5]], max_len=10_000)
    target = [ord(c) for c in "65"] + [0]
    assert ex["input_ids"][-3:] == target
    assert [l for l in ex["labels"] if l != -100] == target
    assert len(ex["labels"]) == len(ex["input_ids"])


def test_drops_demos_to_fit():
    full = build_example(CharTok(), [DEMO] * 3, [[5]], [[6]], max_len=10_000)
    short = build_example(CharTok(), [DEMO] * 3, [[5]], [[6]], max_len=len(full["input_ids"]) - 1)
    assert short["n_demos"] < 3
    assert build_example(CharTok(), [DEMO], [[5]], [[6]], max_len=5) is None
