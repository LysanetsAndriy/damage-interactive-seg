"""Drawing tools: a loop around a thing selects it, a scribble over a thing selects it.

A drawn stroke is a LOOP when it ends near where it started (relative to its
size), otherwise a SCRIBBLE.

loop_object(loop): the one object the user encircled. SAM proposes shapes (box of
the loop alone with 3 candidates; + points inside the loop; + "not this" points
between the loop and its box) and the loop picks: the candidate that matches the
loop best (IoU, computed before clipping, so a mask that spills far outside the
loop loses). The winner is clipped to the slightly widened loop. If no candidate
matches at all, the loop's own shape is the object.

grab(loop) (Cmd/Ctrl + loop): all the objects the prior finds inside the loop.
    1. Candidates: every prior class blob inside the loop, snapped by SAM (box +
       deepest point, grow-only, as in the automatic pre-label).
    2. Containers (Building, Roof: the things other classes sit on) belong to the
       loop only if most of their full prior blob lies inside it: circling one
       window on a facade must not create a "Building" piece shaped like the loop
       (the building is context, not a target). Contents (Damage, Broken Window,
       Damaged roof) are kept even if their blob continues outside: the prior
       often merges a row of windows into one long blob.
    3. Candidates already labeled with the same class are skipped.
    4. If nothing inside the loop was recognized (the prior missed what the user
       circled, e.g. damage it calls "Building"), the loop itself becomes one
       object: SAM on the loop's box + interior points, or the loop's own shape
       when SAM disagrees with it. A facade between circled windows is context,
       so circling a group of windows gives just the windows.
    Everything is clipped to the loop, slightly widened (people draw roughly).

scribble: SAM with positive points sampled along the stroke (one object; good for
long thin things a single click cannot pin down, e.g. a roof strip).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from dmgseg.tool.prelabel import ORDER, snap

CONTAINERS = (1, 2)              # Building, Roof
SPLIT = (4,)                     # Broken Window: countable, the prior often merges neighbours


def split_instances(blob, core=0.5):
    """Split a blob of merged instances at its thin bridges: the thick cores
    (far from the blob's edge) are the seeds; every blob pixel joins the nearest
    core."""
    dt = ndimage.distance_transform_edt(blob)
    cores, n = ndimage.label(dt >= core * dt.max())
    if n <= 1:
        return [blob]
    # cores are only thick parts of the blob; grow them back over the blob
    _, idx = ndimage.distance_transform_edt(cores == 0, return_indices=True)
    lab = cores[idx[0], idx[1]] * blob
    return [lab == k for k in range(1, n + 1) if (lab == k).any()]


@dataclass
class Grab:
    mask: np.ndarray
    prior_class: int | None          # None: the class head decides (the loop itself)
    sam_score: float
    points: list
    labels: list


def is_loop(points, close_frac=0.35, min_size=6):
    """True if the stroke ends near its start (relative to the stroke's size)."""
    p = np.asarray(points, np.float32)
    if len(p) < 4:
        return False
    size = (p.max(0) - p.min(0)).max()
    return size >= min_size and np.linalg.norm(p[-1] - p[0]) <= close_frac * size


def polygon_mask(points, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(points, np.float32)).astype(np.int32)], 1)
    return m.astype(bool)


def widen(region, frac=0.03, min_px=3):
    """The region grown by ~3 % of its size: room for SAM to reach the real edge."""
    r = max(min_px, int(frac * np.sqrt(region.sum())))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(region.astype(np.uint8), k) > 0


def interior_points(region, n=3):
    """The deepest point plus up to n-1 more deep points, spread out."""
    dt = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    pts = [(int(x), int(y))]
    deep = np.argwhere(dt >= 0.5 * dt.max())[:, ::-1]
    if len(deep) > 4000:
        deep = deep[np.linspace(0, len(deep) - 1, 4000).astype(int)]
    for _ in range(n - 1):
        d = np.min([np.hypot(*(deep - p).T) for p in pts], axis=0)
        i = int(d.argmax())
        if d[i] < dt.max():               # the region is too small for another point
            break
        pts.append(tuple(int(v) for v in deep[i]))
    return pts


def stroke_points(stroke, n=6):
    """n points evenly spaced along the stroke (by length)."""
    p = np.asarray(stroke, np.float32)
    if len(p) == 1:
        return [tuple(map(int, p[0]))]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    at = np.linspace(0, seg[-1], n)
    xs, ys = np.interp(at, seg, p[:, 0]), np.interp(at, seg, p[:, 1])
    out = []
    for x, y in zip(xs, ys):
        q = (int(round(x)), int(round(y)))
        if q not in out:
            out.append(q)
    return out


def grab(region, prior, clicker, label_map=None, min_inside=0.5, covered=0.7, min_frac=0.01, min_px=30,
         loop_iou=0.5, max_objects=200):
    """-> list of Grab (see the module docstring)."""
    area = int(region.sum())
    wide = widen(region)
    out, found = [], 0
    if prior is not None:
        lm = prior.argmax(-1)
        for c in ORDER:
            full = (lm == c).astype(np.uint8)
            if not (full.astype(bool) & region).any():
                continue
            n_full, comp_full = cv2.connectedComponents(full, connectivity=8)
            full_area = np.bincount(comp_full.ravel(), minlength=n_full)
            n, comp, stats, _ = cv2.connectedComponentsWithStats((full.astype(bool) & region).astype(np.uint8),
                                                                 connectivity=8)
            for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
                a = stats[i, cv2.CC_STAT_AREA]
                if a < max(min_px, min_frac * area):
                    break
                blob = comp == i
                # the blob's full prior component (it may continue outside the loop)
                ids = np.unique(comp_full[blob])
                if c in CONTAINERS and a < min_inside * full_area[ids].sum():
                    continue                          # mostly outside the loop: context
                found += 1
                for part in (split_instances(blob) if c in SPLIT else [blob]):
                    if part.sum() < min_px:
                        continue
                    p = snap(clicker, part, 0.5, 0.3)
                    mask = (part | (p.mask & (lm == 0))) if p.from_sam else part
                    mask &= wide
                    if label_map is not None and (label_map[mask] == c).sum() >= covered * mask.sum():
                        continue                      # already labeled
                    out.append(Grab(mask, c, p.sam_score, p.points, p.labels))
                    if len(out) >= max_objects:
                        return out
    if not found:
        # the loop itself as an object
        ys, xs = np.nonzero(region)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        pts = interior_points(region)
        m = clicker.prompt([box[:2], box[2:]] + pts, [2, 3] + [1] * len(pts))
        inter, union = (m & region).sum(), (m | region).sum()
        if union and inter / union >= loop_iou:
            mask, score = m & wide, clicker.last_score
        else:
            mask, score = region.copy(), 0.0
        out.append(Grab(mask, None, score, list(clicker.points), list(clicker.labels)))
    return out


def _iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def loop_object(region, clicker, min_fit=0.2):
    """-> (mask, sam_score, points, labels, from_sam). Sets the clicker's state to
    the winning prompt so later clicks refine it."""
    ys, xs = np.nonzero(region)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    box, box_l = [(x0, y0), (x1, y1)], [2, 3]
    wide = widen(region)
    pos = interior_points(region, 3)
    outside = np.zeros_like(region)
    outside[y0:y1, x0:x1] = True
    outside &= ~wide
    neg = interior_points(outside, 4) if outside.sum() >= 50 else []
    prompts = [(box, box_l, True), (box + pos, box_l + [1] * len(pos), False)]
    if neg:
        prompts.append((box + pos + neg, box_l + [1] * len(pos) + [0] * len(neg), False))
    best = None
    for pts, lab, multi in prompts:
        masks, scores, logits = clicker.predict_raw(pts, lab, multi)
        for m, sc, lg in zip(masks, scores, logits):
            fit = _iou(m, region) + 0.05 * float(sc)
            if best is None or fit > best[0]:
                best = (fit, m, float(sc), lg, pts, lab)
    fit, m, sc, lg, pts, lab = best
    clicker.set_state(pts, lab, lg, sc)
    if _iou(m, region) < min_fit:
        return region.copy(), 0.0, pts, lab, False
    clipped = m & wide
    return clipped, sc, pts, lab, clipped.sum() >= 0.97 * m.sum()
