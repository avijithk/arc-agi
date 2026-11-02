import itertools

from arc_ttt.augment import Augmentation, sample_augmentations
from arc_ttt.data import grid_to_text, text_to_grid
from arc_ttt.metrics import cell_accuracy, task_score

G = [[1, 2, 0], [0, 3, 4]]


def test_text_roundtrip():
    assert text_to_grid(grid_to_text(G)) == G
    assert text_to_grid("12\n3") is None  # ragged


def test_all_augmentations_invert():
    for k, aug in itertools.product(range(8), sample_augmentations(5, seed=1)):
        a = Augmentation(d8=k, colors=aug.colors)
        assert a.invert(a.apply(G)) == G


def test_metrics():
    assert cell_accuracy([[1, 2, 0], [0, 3, 9]], G) == 5 / 6
    assert cell_accuracy([[1, 2]], G) == 0.0
    assert task_score([[[[0]], G]], [G]) == 1.0
