import numpy as np
import torch

from dmgseg.app.engine import Session


class FakeClicker:
    """Square masks around the click: 3 candidates of radius 3, 6, 12."""

    def __init__(self, h, w):
        self.h, self.w = h, w

    def set_image(self, image):
        self.image_embedding = torch.zeros(256, 64, 64)

    def reset_object(self):
        self.points, self.labels, self.logits = [], [], None

    def _square(self, x, y, r):
        m = np.zeros((self.h, self.w), bool)
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        return m

    def click(self, x, y, positive=True):
        self.points.append((x, y)); self.labels.append(int(positive))
        if self.logits is None:
            self.candidates = (np.stack([self._square(x, y, r) for r in (3, 6, 12)]), np.array([0.2, 0.9, 0.5]))
            self.first_logits = np.zeros((3, 4, 4))
            self.logits, self.last_score = self.first_logits[1], 0.9
            return self.candidates[0][1]
        self.last_score = 0.8
        m = self._square(x, y, 4)
        return (m | self.candidates[0][1]) if positive else (self.candidates[0][1] & ~m)

    def choose_index(self, k):
        self.logits, self.last_score = self.first_logits[k], float(self.candidates[1][k])
        return self.candidates[0][k]


def make_prior(h, w, cls):
    p = np.zeros((h, w, 6), np.float32)
    p[..., cls] = 1.0
    return p


def test_pending_then_classified_paint_order_undo_and_export(tmp_path):
    h, w = 60, 80
    s = Session(np.zeros((h, w, 3), np.uint8), FakeClicker(h, w), head=None)
    a = s.new_object(20, 20)                       # before the prior: class pending
    assert s.objects[a].pending
    prior = make_prior(h, w, 1)
    prior[15:26, 15:26, :] = 0
    prior[15:26, 15:26, 4] = 1.0                   # a "window" around (20, 20)
    s.set_prior(prior)
    assert not s.objects[a].pending and s.objects[a].label == 4
    b = s.new_object(60, 30)                       # Building area
    assert s.objects[b].label == 1
    s.next_class(b)                                # right click -> second-ranked class
    assert s.objects[b].label != 1
    s.undo()
    assert s.objects[b].label == 1
    s.set_class(b, 3)
    assert s.objects[b].label == 3 and s.objects[b].manual == 3
    s.refine(60, 40, True, b)                      # grow
    assert s.objects[b].mask[40, 60]
    lm = s.label_map()
    assert lm[20, 20] == 4 and lm[30, 60] == 3 and lm[0, 0] == 0
    png, js = s.export(tmp_path / "m.png", source_name="x.png")
    assert png.exists() and js.exists()
    s.delete(a)
    assert len(s.objects) == 1
