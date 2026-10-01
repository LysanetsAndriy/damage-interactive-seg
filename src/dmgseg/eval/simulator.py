"""Simulated user for click-based interactive segmentation (RITM / SAM protocol).

Click 1 is placed at the point deepest inside the object. Each next click goes to
the point deepest inside the largest error region: positive if that region is a
missed part of the object, negative if it is an extra region. Clicks stop at
max_clicks; NoC@t is the number of clicks needed to reach IoU >= t (max_clicks if
never reached).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def iou(pred, gt):
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def _deepest_point(region):
    """(x, y, depth) of the pixel farthest from the region's border."""
    dt = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return int(x), int(y), float(dt[y, x])


def next_click(pred, gt):
    """-> (x, y, is_positive) for the next corrective click."""
    if pred is None or not pred.any():
        x, y, _ = _deepest_point(gt)
        return x, y, True
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    fx, fy, fd = _deepest_point(fn) if fn.any() else (0, 0, -1.0)
    px, py, pd = _deepest_point(fp) if fp.any() else (0, 0, -1.0)
    return (fx, fy, True) if fd >= pd else (px, py, False)


@dataclass
class ClickTrace:
    """Result of simulating one object."""
    ious: list = field(default_factory=list)      # IoU after click 1..n
    clicks: list = field(default_factory=list)    # (x, y, positive)

    def noc(self, threshold, max_clicks):
        for k, v in enumerate(self.ious, start=1):
            if v >= threshold:
                return k
        return max_clicks


def simulate_object(predictor, gt, max_clicks=20, stop_iou=None):
    """predictor must already have the image set. stop_iou: stop early once reached
    (the remaining IoU values are filled with the last one, as is standard)."""
    trace = ClickTrace()
    predictor.reset_object()
    pred = None
    for _ in range(max_clicks):
        x, y, positive = next_click(pred, gt)
        trace.clicks.append((x, y, positive))
        pred = predictor.click(x, y, positive)
        trace.ious.append(iou(pred, gt))
        if stop_iou is not None and trace.ious[-1] >= stop_iou:
            break
    trace.ious += [trace.ious[-1]] * (max_clicks - len(trace.ious))
    return trace


def summarize(traces_by_class, class_names, max_clicks=20, thresholds=(0.85, 0.90)):
    """traces_by_class: {class_id: [ClickTrace]}. Returns per-class and overall
    mean IoU@k, NoC@t and the share of objects never reaching t (failure rate)."""
    def stats(traces):
        if not traces:
            return None
        ious = np.array([t.ious for t in traces])
        out = {"n": len(traces), "miou@k": ious.mean(0).round(4).tolist()}
        for t in thresholds:
            nocs = np.array([tr.noc(t, max_clicks) for tr in traces])
            reached = ious.max(1) >= t
            out[f"noc@{int(t * 100)}"] = round(float(nocs.mean()), 3)
            out[f"fail@{int(t * 100)}"] = round(float(1 - reached.mean()), 4)
        return out

    every = [t for ts in traces_by_class.values() for t in ts]
    result = {"all": stats(every)}
    for c, name in enumerate(class_names):
        result[name] = stats(traces_by_class.get(c, []))
    return result
