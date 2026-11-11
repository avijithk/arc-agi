import torch

from arc_ttt.model_utils import GridConstraint


class Tok:
    """Digits 0-9 -> ids 0-9, newline -> 10, EOS -> 11, vocab of 20."""
    eos_token_id = 11
    pad_token_id = 11

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 if c == "\n" else int(c) for c in text]}


def allowed(gen, prompt=(15, 16), max_side=30):
    c = GridConstraint(Tok(), prompt_len=len(prompt), max_side=max_side)
    ids = torch.tensor([list(prompt) + gen])
    out = c(ids, torch.zeros(1, 20))
    return {i for i in range(20) if out[0, i] > float("-inf")}


DIG = set(range(10))


def test_first_row_free_then_fixed_width():
    assert allowed([]) == DIG                       # must start with a digit
    assert allowed([1]) == DIG | {10, 11}           # can continue, end row, or stop
    assert allowed([1, 2, 10]) == DIG               # new row: digits only
    assert allowed([1, 2, 10, 3]) == DIG            # row 2 shorter than width 2: digits only
    assert allowed([1, 2, 10, 3, 4]) == {10, 11}    # row 2 full: newline or stop


def test_never_allows_other_tokens_and_caps_rows():
    assert allowed([5] * 30) == {10, 11}            # 30 cells: row must end
    gen = sum([[1, 10] for _ in range(29)], []) + [1]
    assert allowed(gen) == {11}                     # 30th row full: only stop
    assert allowed([1, 11]) == {11}                 # finished stays finished
