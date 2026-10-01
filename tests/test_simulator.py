import numpy as np

from dmgseg.eval.simulator import ClickTrace, next_click, simulate_object, summarize


def square(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_first_click_is_deepest_inside_object():
    gt = square(50, 50, 10, 31, 10, 31)
    x, y, positive = next_click(None, gt)
    assert positive and (x, y) == (20, 20)


def test_next_click_targets_larger_error():
    gt = square(60, 60, 10, 40, 10, 40)
    pred = gt.copy()
    pred[10:40, 25:40] = False          # big missed part -> positive click inside it
    pred[50:53, 50:53] = True           # small false positive
    x, y, positive = next_click(pred, gt)
    assert positive and 25 <= x < 40 and gt[y, x]

    pred2 = gt.copy()
    pred2[45:60, 0:60] = True           # big false positive -> negative click outside gt
    x, y, positive = next_click(pred2, gt)
    assert not positive and not gt[y, x]


class BoxClicker:
    """Fake predictor: a positive click fills a box around the click inside the
    object, a negative click erases a box. The first click also leaks a bit of
    background, which a negative click must fix."""

    def __init__(self, h, w, r, gt):
        self.h, self.w, self.r, self.gt = h, w, r, gt

    def reset_object(self):
        self.mask = np.zeros((self.h, self.w), bool)

    def click(self, x, y, positive=True):
        r = self.r
        box = np.zeros_like(self.mask)
        box[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        if positive:
            first = not self.mask.any()
            self.mask |= box & self.gt
            if first:
                self.mask[:4, :] = True  # leak
        else:
            self.mask &= ~box
        return self.mask.copy()


def test_simulation_improves_and_noc():
    gt = square(80, 80, 20, 60, 20, 60)
    trace = simulate_object(BoxClicker(80, 80, 10, gt), gt, max_clicks=20)
    assert not all(p for _, _, p in trace.clicks)  # the leak needed a negative click
    assert len(trace.ious) == 20
    assert trace.ious[-1] > trace.ious[0]
    assert trace.ious[-1] > 0.9
    assert trace.noc(0.9, 20) <= 20


def test_summarize_counts_failures():
    good, bad = ClickTrace(ious=[0.95] * 5), ClickTrace(ious=[0.5] * 5)
    s = summarize({0: [good, bad]}, ["A"], max_clicks=5)
    assert s["A"]["noc@90"] == (1 + 5) / 2
    assert s["A"]["fail@90"] == 0.5
    assert s["all"]["n"] == 2
