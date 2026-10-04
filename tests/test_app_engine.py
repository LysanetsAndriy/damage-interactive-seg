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


# ------------------------------------------------------------------ drawing tools
class DrawClicker(FakeClicker):
    """+ box prompts (the box itself) and multi-point prompts (squares around the
    positive points, minus squares around the negative ones)."""

    def box_click(self, box, x, y):
        self.points, self.labels = [box[:2], box[2:], (x, y)], [2, 3, 1]
        m = np.zeros((self.h, self.w), bool)
        m[box[1]:box[3], box[0]:box[2]] = True
        self.logits, self.last_score = np.zeros((4, 4)), 0.9
        return m

    def predict_raw(self, points, labels, multimask=False):
        """Box-only prompts: the box shrunk by 0, 3, 6 px (3 candidates)."""
        (x0, y0), (x1, y1) = points[0], points[1]
        out = []
        for d in ((0, 3, 6) if multimask else (3,)):
            m = np.zeros((self.h, self.w), bool)
            m[y0 + d:y1 - d, x0 + d:x1 - d] = True
            out.append(m)
        return np.stack(out), np.full(len(out), 0.8), np.zeros((len(out), 4, 4))

    def set_state(self, points, labels, logits, score):
        self.points, self.labels, self.logits, self.last_score = list(points), list(labels), logits, score

    def prompt(self, points, labels, mask_input=None):
        self.points, self.labels = list(points), list(labels)
        m = np.zeros((self.h, self.w), bool)
        for (x, y), l in zip(points, labels):
            if l == 1:
                m |= self._square(int(x), int(y), 4)
        for (x, y), l in zip(points, labels):
            if l == 0:
                m &= ~self._square(int(x), int(y), 4)
        self.logits, self.last_score = np.zeros((4, 4)), 0.8
        return m


def two_windows_prior(h, w):
    prior = make_prior(h, w, 1)                  # Building everywhere
    for x0 in (15, 40):
        prior[15:26, x0:x0 + 11, :] = 0
        prior[15:26, x0:x0 + 11, 4] = 1.0        # two "windows"
    return prior


def test_lasso_grabs_objects_inside_not_the_context():
    from dmgseg.tool.lasso import is_loop
    h, w = 60, 80
    s = Session(np.zeros((h, w, 3), np.uint8), DrawClicker(h, w), head=None)
    s.set_prior(two_windows_prior(h, w))
    loop = [(10, 10), (56, 10), (56, 31), (10, 31), (11, 11)]
    assert is_loop(loop) and not is_loop([(10, 10), (30, 12), (56, 30)])
    new = s.lasso(loop, group=True)
    assert [s.objects[i].label for i in new] == [4, 4]     # the windows, not a facade piece
    assert s.lasso(loop, group=True) == []                 # already labeled: nothing new
    s.undo()
    assert s.objects == []


def test_lasso_fallback_scribble_and_exact_edits():
    h, w = 60, 80
    s = Session(np.zeros((h, w, 3), np.uint8), DrawClicker(h, w), head=None)
    s.set_prior(make_prior(h, w, 1))             # nothing but Building: no candidates in a small loop
    new = s.lasso([(30, 30), (40, 30), (40, 40), (30, 40)], group=True)
    assert len(new) == 1 and s.objects[new[0]].mask[35, 35]
    i = s.scribble([(5, 50), (20, 50), (35, 50)])          # an object along the line
    assert s.objects[i].mask[50, 5] and s.objects[i].mask[50, 35]
    s.edit_area([(0, 0), (10, 0), (10, 10), (0, 10)], add=True, index=i)
    assert s.objects[i].mask[5, 5]
    s.edit_area([(0, 44), (79, 44), (79, 59), (0, 59)], add=False)   # cut everywhere
    assert not s.objects[i].mask[50, 20] and s.objects[i].mask[5, 5]
    n = len(s.objects)
    s.edit_area([(28, 28), (42, 28), (42, 42), (28, 42)], add=False)  # empties the lasso object
    assert len(s.objects) == n - 1


def test_single_loop_local_edits_and_multi_delete():
    h, w = 60, 80
    s = Session(np.zeros((h, w, 3), np.uint8), DrawClicker(h, w), head=None)
    s.set_prior(two_windows_prior(h, w))
    [i] = s.lasso([(10, 10), (40, 10), (40, 40), (10, 40)])        # one object, picked by the loop
    o = s.objects[i]
    assert o.sam_valid and o.mask[25, 25] and not o.mask[5, 5]
    # a pre-label-like object (not SAM's mask): Option+click cuts only the piece under the cursor
    blob = np.zeros((h, w), bool)
    blob[5:55, 45:78] = True
    s.objects.append(type(o)(mask=blob, ranking=[1, 2, 3, 4, 5, 0], sam_valid=False, auto=True))
    j = len(s.objects) - 1
    s.refine(60, 30, False, j)
    m = s.objects[j].mask
    assert not m[30, 60] and m[10, 50] and m.sum() > 0.7 * blob.sum()
    s.set_classes([i, j], 3)
    assert s.objects[i].label == s.objects[j].label == 3
    s.delete_many([i, j])
    assert s.objects == []
    s.undo()
    assert len(s.objects) == 2


def test_user_class_beats_automatic_window(tmp_path):
    """A Damage object the user sets over an automatic Broken Window blob shows as
    Damage (also in the CVAT export); a user-confirmed Building does not hide it."""
    from dmgseg.app.engine import Obj
    from dmgseg.app.project import FolderProject, export_cvat
    from dmgseg.data.cvat import parse_annotations, semantic_mask
    from PIL import Image
    h, w = 40, 50
    s = Session(np.zeros((h, w, 3), np.uint8), DrawClicker(h, w), head=None)
    win = np.zeros((h, w), bool); win[10:20, 10:20] = True
    dmg = np.zeros((h, w), bool); dmg[5:25, 5:25] = True
    bld = np.zeros((h, w), bool); bld[0:35, 0:45] = True
    s.objects = [Obj(mask=win, ranking=[4, 1, 2, 3, 5, 0], auto=True, sam_valid=False),
                 Obj(mask=bld, ranking=[1, 2, 3, 4, 5, 0]),
                 Obj(mask=dmg, ranking=[4, 3, 1, 2, 5, 0])]
    s.set_classes([1], 1)                     # user confirms the building
    assert s.label_map()[15, 15] == 4         # the window still shows
    s.set_class(2, 3)                         # user: the bigger object is Damage
    lm = s.label_map()
    assert lm[15, 15] == 3 and lm[30, 30] == 1
    assert s.object_at(15, 15) == 2
    # the export reproduces it
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(tmp_path / "a.png")
    proj = FolderProject(tmp_path)
    proj.save_state("a.png", s.to_state())
    export_cvat(proj, tmp_path / "x.xml")
    assert np.array_equal(semantic_mask(parse_annotations(tmp_path / "x.xml")[0]), lm)
