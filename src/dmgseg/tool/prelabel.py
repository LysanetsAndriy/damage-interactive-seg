"""Automatic first draft: DINOv2 prior blobs snapped to object shapes by SAM.

For each clickable class, the prior's argmax map is split into connected blobs;
SAM is prompted with each blob's box and deepest point. SAM's mask is accepted if
it roughly agrees with the blob (precision >= 0.5, recall >= 0.3) and then may only
ADD unlabeled pixels to the blob ("grow-only"): it snaps objects out to their
real edges but never takes pixels the prior gave to another class.

On 10 validation images: prior map 0.6705 mIoU -> pre-label 0.6856 (Building +3.5,
Roof +2.8, Other +2.0). Without grow-only, SAM's spills cost Roof 12 points.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

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


def bbox(mask):
    """(y0, y1, x0, x1) of a non-empty mask."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def deepest_point(mask):
    """(x, y) farthest from the mask's edge. The distance transform runs on the
    mask's bounding box only (same result, much faster on big images)."""
    y0, y1, x0, x1 = bbox(mask)
    dt = ndimage.distance_transform_edt(np.pad(mask[y0:y1, x0:x1], 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return int(x + x0), int(y + y0)


def box_iou(a, b):
    """IoU of two masks, computed on the union of their bounding boxes."""
    ay0, ay1, ax0, ax1 = bbox(a)
    by0, by1, bx0, bx1 = bbox(b)
    if ay1 <= by0 or by1 <= ay0 or ax1 <= bx0 or bx1 <= ax0:
        return 0.0
    y0, y1, x0, x1 = min(ay0, by0), max(ay1, by1), min(ax0, bx0), max(ax1, bx1)
    a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def prior_blobs(prior, min_frac=2e-4, min_px=150, opening=False):
    """[(class, blob mask)] from the prior's argmax map, biggest first within a class.
    No morphological opening by default: it erased thin roof strips."""
    h, w = prior.shape[:2]
    lm = prior.argmax(-1)
    out = []
    for c in ORDER:
        m = (lm == c).astype(np.uint8)
        if not m.any():
            continue
        if opening:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        blobs = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= max(min_px, min_frac * h * w)]
        for _, i in sorted(blobs, reverse=True):
            out.append((c, comp == i))
    return out


def snap(clicker, blob, min_precision=0.7, min_recall=0.5):
    """SAM shape for a blob: prompt = blob box + its deepest point. SAM's mask is kept
    only if it agrees with the blob both ways: most of it lies inside the blob
    (precision: no spilling into sky or the neighbour) and it covers most of the
    blob (recall); otherwise the blob itself is kept."""
    y0, y1, x0, x1 = bbox(blob)
    box = (x0, y0, x1, y1)
    x, y = deepest_point(blob)
    mask = clicker.box_click(box, x, y)
    inter = np.logical_and(mask, blob).sum()
    if mask.any() and inter / mask.sum() >= min_precision and inter / blob.sum() >= min_recall:
        return Proposal(mask, 0, clicker.last_score, list(clicker.points), list(clicker.labels),
                        clicker.logits.copy(), True)
    return Proposal(blob, 0, 0.0, list(clicker.points), list(clicker.labels), None, False)


def prelabel(prior, clicker, max_objects=400, dedupe_iou=0.7, min_precision=0.5, min_recall=0.3,
             keep_small_px=1, use_sam=True, grow_only=True):
    """-> list of Proposals (prior_class set). Blobs big enough are snapped by SAM;
    smaller ones (>= keep_small_px) are kept as they are, so the draft never loses
    what the prior found. Duplicates of earlier proposals are dropped."""
    props = []
    lm = prior.argmax(-1)
    for c, blob in prior_blobs(prior, min_px=keep_small_px, min_frac=0):
        if use_sam and blob.sum() >= max(150, 2e-4 * blob.size):
            p = snap(clicker, blob, min_precision, min_recall)
            if p.from_sam and grow_only:
                # SAM may only add unlabeled ("Other") pixels to the blob, never take
                # pixels the prior gave to another class (that cost Roof 12 points)
                p.mask = blob | (p.mask & (lm == 0))
        else:
            p = Proposal(blob, 0, 0.0, [], [], None, False)
            if blob.sum() >= 4:   # keep a box + point prompt so a later click can snap it
                y0, y1, x0, x1 = bbox(blob)
                x, y = deepest_point(blob)
                p.points = [(x0, y0), (x1, y1), (x, y)]
                p.labels = [2, 3, 1]
        p.prior_class = c
        if any(q.prior_class == c and box_iou(q.mask, p.mask) > dedupe_iou for q in props):
            continue
        props.append(p)
        if len(props) >= max_objects:
            break
    return props
