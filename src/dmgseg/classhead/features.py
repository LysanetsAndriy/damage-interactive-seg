"""Description cards: one fixed-length vector per candidate mask (class head, version A).

Card layout (288 float32):
    prior_in    12  mean prior probability per class (6) + share of mask pixels per
                    prior argmax class (6)
    prior_ring   6  mean prior probability per class in a ring around the mask
    sam        256  SAM 2 image embedding (256 x 64 x 64) averaged over the mask
    geometry     8  log relative area, bbox width/height (relative), log aspect,
                    fill ratio (area / bbox area), center x/y (relative), compactness
    sam_info     6  SAM predicted IoU, candidate type one-hot (first-click mask 0/1/2,
                    refined mask), click number / 3
"""
import cv2
import numpy as np

N_CLASSES = 6
GROUPS = {"prior_in": 12, "prior_ring": 6, "sam": 256, "geometry": 8, "sam_info": 6}
CARD_SIZE = sum(GROUPS.values())
REFINED = 3  # candidate type for masks after clicks 2, 3, ...


def group_slices():
    out, start = {}, 0
    for name, n in GROUPS.items():
        out[name] = slice(start, start + n)
        start += n
    return out


def _ring(mask):
    """Pixels within r of the mask but outside it, r ~ 15% of the object's size."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    r = int(max(3, 0.15 * np.sqrt(len(ys))))
    h, w = mask.shape
    y0, y1 = max(0, ys.min() - r), min(h, ys.max() + r + 1)
    x0, x1 = max(0, xs.min() - r), min(w, xs.max() + r + 1)
    sub = mask[y0:y1, x0:x1]
    # distance to the mask <= r: the same disk dilation, but its cost does not grow
    # with r (a 230 px elliptical kernel took ~60 ms on a large building)
    dist = cv2.distanceTransform((~sub).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ring = np.zeros_like(mask)
    ring[y0:y1, x0:x1] = (dist <= r) & ~sub
    return ring


def _geometry(mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area = len(ys)
    if area == 0:
        return np.zeros(8, np.float32)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    edges = mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    perimeter = max(edges.sum(), 1)
    return np.array([
        np.log(area / (h * w)),
        bw / w, bh / h,
        np.log(bw / bh),
        area / (bw * bh),
        xs.mean() / w, ys.mean() / h,
        perimeter ** 2 / (4 * np.pi * area),
    ], np.float32)


def _adaptive_weights(mask, out_h, out_w):
    """F.adaptive_avg_pool2d of a 2-D mask to (out_h, out_w), in numpy: cell (i, j)
    averages rows floor(i*H/out_h) .. ceil((i+1)*H/out_h) (same for columns)."""
    m = mask.astype(np.float64)
    H, W = m.shape
    ii = np.zeros((H + 1, W + 1))
    ii[1:, 1:] = m.cumsum(0).cumsum(1)                       # integral image
    r = np.arange(out_h)
    c = np.arange(out_w)
    r0, r1 = (r * H) // out_h, -((-(r + 1) * H) // out_h)
    c0, c1 = (c * W) // out_w, -((-(c + 1) * W) // out_w)
    s = ii[r1][:, c1] - ii[r0][:, c1] - ii[r1][:, c0] + ii[r0][:, c0]
    return s / ((r1 - r0)[:, None] * (c1 - c0)[None, :])


def sam_pool(sam_embed, mask):
    """Average SAM's (256, 64, 64) embedding over the mask (mask resized with area
    weighting, so tiny objects still contribute). sam_embed: numpy array or torch
    tensor (PyTorch backend)."""
    E = sam_embed if isinstance(sam_embed, np.ndarray) else sam_embed.float().cpu().numpy()
    weights = _adaptive_weights(mask, E.shape[-2], E.shape[-1]).astype(np.float32)
    total = weights.sum()
    if total <= 0:
        return np.zeros(E.shape[0], np.float32)
    return ((E * weights).sum((-2, -1)) / total).astype(np.float32)


def card(mask, prior, sam_embed, sam_score, cand_type, click_no):
    """mask: bool (H, W); prior: float (H, W, 6); sam_embed: tensor (256, 64, 64)."""
    if mask.any():
        p_in = prior[mask]
        mean_in = p_in.mean(0)
        frac_in = np.bincount(p_in.argmax(1), minlength=N_CLASSES) / len(p_in)
    else:
        mean_in = frac_in = np.zeros(N_CLASSES)
    ring = _ring(mask)
    mean_ring = prior[ring].mean(0) if ring.any() else mean_in
    info = np.zeros(6, np.float32)
    info[0] = sam_score
    info[1 + cand_type] = 1
    info[5] = click_no / 3
    return np.concatenate([mean_in, frac_in, mean_ring, sam_pool(sam_embed, mask),
                           _geometry(mask), info]).astype(np.float32)
