import numpy as np
import torch

from dmgseg import paths
from dmgseg.data.cvat import PAPER_KINDS, parse_annotations, semantic_mask
from dmgseg.data.split import load_split, paper_split
from dmgseg.prior.losses import CombinedLoss
from dmgseg.prior.patches import patch_grid

ANNS = parse_annotations(paths.ANNOTATIONS_XML)


def test_split_matches_paper_and_saved_file():
    split = paper_split(ANNS)
    assert (len(split["train"]), len(split["val"])) == (246, 44)
    assert split == load_split()


def test_brush_masks_decode_to_their_bbox():
    for ann in ANNS:
        for s in ann.shapes_of(("mask",)):
            left, top, w, h = s.bbox
            assert sum(s.rle) == w * h
            m = s.render(ann.height, ann.width)
            assert m.sum() > 0 and not m[:, :left].any() and not m[:top].any()


def test_paper_mode_reproduces_table2():
    counts = np.zeros(6)
    for ann in ANNS:
        counts += np.bincount(semantic_mask(ann, PAPER_KINDS).ravel(), minlength=6)
    np.testing.assert_allclose(np.round(counts / counts.sum() * 100, 2),
                               [66.10, 19.42, 1.24, 8.66, 2.66, 1.92])


def test_patch_counts_match_table3():
    split = load_split()
    by_name = {a.name: a for a in ANNS}
    for size, stride, expected_train in [(224, 180, 11042), (384, 256, 4659), (640, 160, 5887)]:
        n = 0
        for name in split["train"]:
            a = by_name[name]
            w, h = a.width, a.height
            if w < size or h < size:
                s = max(size / w, size / h)
                w, h = int(round(w * s)), int(round(h * s))
            grid = patch_grid(w, h, size, stride)
            assert all(0 <= x <= w - size and 0 <= y <= h - size for x, y in grid)
            n += len(grid)
        assert n == expected_train


def test_loss_is_small_for_perfect_prediction():
    target = torch.randint(0, 6, (2, 16, 16))
    logits = torch.nn.functional.one_hot(target, 6).permute(0, 3, 1, 2).float() * 50
    weights = torch.tensor([0.3, 1.0, 2.5, 1.5, 3.0, 2.8])
    assert CombinedLoss(class_weights=weights)(logits, target).item() < 1e-3
