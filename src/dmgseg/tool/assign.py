"""Combining SAM masks with the semantic prior (shared by the evaluation and the app).

- class_ranking: which class a clicked object gets (and the right-click order).
- choose_first_mask: which of SAM's 3 first-click masks to keep.

The "Other" class is never assigned to a clicked object: unlabeled pixels are Other.
"""
import cv2
import numpy as np
from scipy import ndimage

OTHER = 0
CLICKABLE = (1, 2, 3, 4, 5)  # Building, Roof, Damage, Broken Window, Damaged roof


def fit_prior(prior, height, width):
    """Resize a (h, w, C) prior to the image size if inference used an upscaled copy."""
    if prior.shape[:2] == (height, width):
        return prior
    return np.stack([cv2.resize(prior[..., c], (width, height), interpolation=cv2.INTER_LINEAR)
                     for c in range(prior.shape[-1])], axis=-1)


def class_scores(prior, mask):
    """Mean prior probability of each clickable class inside the mask, renormalized.
    -> dict {class_id: score}."""
    if not mask.any():
        return {c: 1 / len(CLICKABLE) for c in CLICKABLE}
    mean = prior[mask].mean(0)
    sub = np.array([mean[c] for c in CLICKABLE])
    sub = sub / max(sub.sum(), 1e-8)
    return dict(zip(CLICKABLE, sub.tolist()))


def class_ranking(prior, mask):
    """Clickable classes, most likely first: left click gets [0], each right click
    moves to the next."""
    scores = class_scores(prior, mask)
    return sorted(scores, key=scores.get, reverse=True)


def prior_region(prior, x, y):
    """The object as the prior sees it: the connected region around the click where
    the click's most likely clickable class wins."""
    c = max(CLICKABLE, key=lambda k: prior[y, x, k])
    region = prior.argmax(-1) == c
    if not region[y, x]:  # click pixel itself labelled Other/another class: soften
        region = prior[..., c] >= 0.5 * prior[y, x, c]
    labels, _ = ndimage.label(region)
    return labels == labels[y, x] if labels[y, x] else region


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def choose_first_mask(masks, sam_scores, prior, x, y, method="region"):
    """Index of the first-click mask to keep.

    score:  SAM's own predicted IoU (plain SAM).
    purity: the mask whose pixels the prior agrees on most (max mean clickable prob).
    region: the mask that best overlaps the prior's own region around the click.
    """
    if method == "score":
        return int(np.argmax(sam_scores))
    if method == "purity":
        return int(np.argmax([max(prior[m].mean(0)[list(CLICKABLE)]) if m.any() else 0 for m in masks]))
    if method == "region":
        region = prior_region(prior, x, y)
        return int(np.argmax([_iou(m, region) for m in masks]))
    raise ValueError(method)
