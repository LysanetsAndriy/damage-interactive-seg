"""Automatic first draft: DINOv2 prior blobs snapped to object shapes by SAM.

For each clickable class, the prior's argmax map is split into connected blobs
(tiny ones dropped); SAM is prompted with each blob's box and its deepest point.
If SAM's mask barely overlaps the blob, the blob itself is kept. Large classes go
first (Building, Roof) so smaller objects are painted on top later anyway.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from dmgseg.eval.simulator import iou
from dmgseg.tool.assign import CLICKABLE

ORDER = [1, 2, 5, 3, 4]          # Building, Roof, Damaged roof, Damage, Broken Window


@dataclass
class Proposal:
    mask: np.ndarray
    prior_class: int
    sam_score: float
    points: list
    labels: list
    logits: np.ndarray | None
    from_sam: bool


def prior_blobs(prior, min_frac=2e-4, min_px=150):
    """[(class, blob mask)] from the prior's argmax map, biggest first within a class."""
    h, w = prior.shape[:2]
    lm = prior.argmax(-1)
    out = []
    for c in ORDER:
        m = (lm == c).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        blobs = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= max(min_px, min_frac * h * w)]
        for _, i in sorted(blobs, reverse=True):
            out.append((c, comp == i))
    return out


def snap(clicker, blob, min_agreement=0.3):
    """SAM shape for a blob: prompt = blob box + its deepest point."""
    ys, xs = np.nonzero(blob)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    dt = ndimage.distance_transform_edt(np.pad(blob, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    mask = clicker.box_click(box, int(x), int(y))
    if iou(mask, blob) >= min_agreement:
        return Proposal(mask, 0, clicker.last_score, list(clicker.points), list(clicker.labels),
                        clicker.logits.copy(), True)
    return Proposal(blob, 0, 0.0, list(clicker.points), list(clicker.labels), None, False)


def prelabel(prior, clicker, max_objects=150, dedupe_iou=0.7):
    """-> list of Proposals (prior_class set); duplicates of earlier proposals dropped."""
    props = []
    for c, blob in prior_blobs(prior):
        p = snap(clicker, blob)
        p.prior_class = c
        if any(q.prior_class == c and iou(q.mask, p.mask) > dedupe_iou for q in props):
            continue
        props.append(p)
        if len(props) >= max_objects:
            break
    return props
