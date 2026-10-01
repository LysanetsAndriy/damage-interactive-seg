import numpy as np
import torch

from dmgseg.eval.metrics import SegMetrics


def test_global_metrics_on_toy_example():
    true = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 1, 1, 1]])
    m = SegMetrics(2)
    m.update(pred, true)
    r = m.compute()
    np.testing.assert_allclose(r["global"]["iou"], [1 / 2, 2 / 3])
    # globally, F1 and Dice are the same quantity
    np.testing.assert_allclose(r["global"]["f1"], r["global"]["dice"])


def test_paper_iou_is_mean_of_per_image_values():
    m = SegMetrics(2)
    m.update(np.array([[0, 0]]), np.array([[0, 0]]))   # class 1 absent -> NaN, ignored
    m.update(np.array([[1, 1]]), np.array([[0, 1]]))   # IoU0 = 0, IoU1 = 1/2
    r = m.compute()
    np.testing.assert_allclose(r["paper"]["iou"], [0.5, 0.5])    # (1 + 0)/2, nan ignored
    np.testing.assert_allclose(r["global"]["iou"], [2 / 3, 1 / 2])


def test_torch_and_numpy_inputs_agree():
    rng = np.random.default_rng(0)
    pred, true = rng.integers(0, 6, (3, 8, 8)), rng.integers(0, 6, (3, 8, 8))
    a, b = SegMetrics(6), SegMetrics(6)
    a.update(pred, true)
    b.update(torch.from_numpy(pred), torch.from_numpy(true))
    np.testing.assert_array_equal(a.confusion, b.confusion)
