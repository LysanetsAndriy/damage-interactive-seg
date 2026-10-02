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
    assert s.objects[a].pending and s.objects[a].label is None
    assert s.label_map()[20, 20] == 0              # pending objects are not painted
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
    # a "tree" in front of the building: marked Other, it cuts its area out
    t = s.new_object(62, 32)
    s.set_class(t, 0)
    lm = s.label_map()
    assert lm[32, 62] == 0 and s.other_mask()[32, 62] and lm[20, 20] == 4
    assert s.counts()[0] == 1
    # right-click cycling ends with Other
    assert s.objects[b].ranking[-1] == 0
    s.delete(a)
    assert len(s.objects) == 2


def _session_with_objects(h=60, w=80):
    s = Session(np.zeros((h, w, 3), np.uint8), FakeClicker(h, w), head=None)
    s.set_prior(make_prior(h, w, 1))
    b = s.new_object(30, 30)                 # Building square r=6 (SAM's top-scored candidate)
    s.set_class(b, 1)
    win = s.new_object(32, 32)               # a window on it
    s.set_class(win, 4)
    tree = s.new_object(27, 27)              # a tree in front, marked Other
    s.set_class(tree, 0)
    return s


def test_state_roundtrip():
    s = _session_with_objects()
    state = s.to_state()
    s2 = Session(np.zeros((60, 80, 3), np.uint8), FakeClicker(60, 80), head=None)
    s2.load_state(state)
    assert len(s2.objects) == len(s.objects)
    assert np.array_equal(s2.label_map(), s.label_map())
    assert [o.label for o in s2.objects] == [o.label for o in s.objects]


def test_cvat_export_roundtrip_through_dataset_parser(tmp_path):
    from PIL import Image
    from dmgseg.app.project import FolderProject, export_cvat
    from dmgseg.data.cvat import parse_annotations, semantic_mask
    Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(tmp_path / "b.png")
    proj = FolderProject(tmp_path)
    s = _session_with_objects()
    proj.save_state("a.png", s.to_state())
    for shape in ("mask", "polygon"):
        out = tmp_path / f"cvat_{shape}.xml"
        assert export_cvat(proj, out, shape=shape) == 1
        anns = parse_annotations(out)
        assert [a.name for a in anns] == ["a.png"]
        back = semantic_mask(anns[0])
        if shape == "mask":
            assert np.array_equal(back, s.label_map())          # exact, incl. the tree cut-out
        else:
            assert (back == s.label_map()).mean() > 0.95
