"""How well do the drawing tools work on the validation objects?

Each object gets three single gestures on an empty image (fresh state each time):
    click     one left click at the object's deepest point (the baseline)
    loop      a rough loop around it: the object's outline widened by ~8 % of its
              size, simplified and jittered like a hand-drawn line (one object)
    scribble  a line along the object's main axis through its interior
-> IoU with the object (best matching new object; and the union of all new
objects for the loop), class correct, how many objects the loop created.

Groups: for each Broken Window with >= 1 other broken window nearby, one
Cmd+loop (group mode) around the group -> share of the group's windows found (IoU >= 0.5) and the
number of extra objects.

    python scripts/lasso_eval.py --max-objects 400
"""
import argparse
import functools
import json
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from dmgseg import paths
from dmgseg.app.engine import ClassHead, Session
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.data.split import load_split
from dmgseg.prior.kfold import load_prior
from dmgseg.sam.predictor import SamClicker

print = functools.partial(print, flush=True)  # noqa: A001
WINDOW = CLASS_NAMES.index("Broken Window")


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def rough_loop(mask, rng, widen=0.08, jitter=0.4):
    """A hand-drawn-like loop around a mask (or around several masks)."""
    r = int(widen * np.sqrt(mask.sum())) + 3
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    big = cv2.dilate(mask.astype(np.uint8), k)
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea) if len(cnts) == 1 else cv2.convexHull(np.concatenate(cnts))
    c = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)[:, 0, :].astype(np.float32)
    c += rng.uniform(-jitter * r, jitter * r, c.shape)
    h, w = mask.shape
    c[:, 0] = c[:, 0].clip(0, w - 1)
    c[:, 1] = c[:, 1].clip(0, h - 1)
    return [tuple(p) for p in c]


def axis_stroke(mask):
    """A line along the object's main axis, through its deep interior."""
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    pts = np.argwhere(dt >= 0.5 * dt.max())[:, ::-1].astype(np.float32)
    if len(pts) < 2:
        return [tuple(pts[0])] * 2
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    proj = (pts - mean) @ vt[0]
    a, b = pts[proj.argmin()], pts[proj.argmax()]
    return [tuple(a + t * (b - a)) for t in np.linspace(0, 1, 12)]


def deepest(mask):
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return int(x), int(y)


def fresh(s):
    s.objects, s.active, s._undo = [], None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-objects", type=int, default=400)
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--out", default=str(paths.ARTIFACTS / "lasso_eval.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    names = load_split()["val"]
    objs = [(n, o) for n in names for o in image_objects(anns[n], args.min_area)
            if o.label != 0]
    pick = set(rng.choice(len(objs), min(args.max_objects, len(objs)), replace=False).tolist())
    by_image = {}
    for i, (n, o) in enumerate(objs):
        if i in pick:
            by_image.setdefault(n, []).append(o)
    sam_ft = paths.ARTIFACTS / "sam" / "finetuned_decoder.pt"
    clicker = SamClicker("small", "cpu", decoder_weights=sam_ft)
    head = ClassHead(paths.ARTIFACTS / "classhead" / "head_a_samft.pt")
    rows, groups, t0 = [], [], time.time()
    for k, name in enumerate(names):
        image = np.asarray(Image.open(paths.IMAGES_DIR / name).convert("RGB"))
        s = Session(image, clicker, head)
        s.set_prior(load_prior(paths.ARTIFACTS / "priors" / paths.PRIOR_RUN / f"{name}.npz"))
        for o in by_image.get(name, []):
            gt, cls = o.mask, int(o.label)
            r = {"image": name, "class": cls, "area": int(gt.sum())}
            fresh(s)
            i = s.new_object(*deepest(gt))
            r["click_iou"], r["click_cls"] = iou(s.objects[i].mask, gt), s.objects[i].label == cls
            fresh(s)
            new = s.lasso(rough_loop(gt, rng))
            if new:
                ious = [iou(s.objects[j].mask, gt) for j in new]
                b = new[int(np.argmax(ious))]
                union = np.any([s.objects[j].mask for j in new], axis=0)
                r.update(loop_iou=max(ious), loop_union_iou=iou(union, gt), loop_cls=s.objects[b].label == cls,
                         loop_n=len(new))
            else:
                r.update(loop_iou=0.0, loop_union_iou=0.0, loop_cls=False, loop_n=0)
            fresh(s)
            i = s.scribble(axis_stroke(gt))
            r["scribble_iou"], r["scribble_cls"] = iou(s.objects[i].mask, gt), s.objects[i].label == cls
            rows.append(r)
        # groups of broken windows
        wins = [o.mask for o in image_objects(anns[name], 100) if o.label == WINDOW]
        if len(wins) >= 2:
            cents = np.array([np.argwhere(m).mean(0) for m in wins])
            sizes = np.array([np.sqrt(m.sum()) for m in wins])
            for g in rng.choice(len(wins), min(3, len(wins)), replace=False):
                near = np.flatnonzero(np.hypot(*(cents - cents[g]).T) <= 3 * sizes[g])
                if len(near) < 2:
                    continue
                members = [wins[j] for j in near]
                fresh(s)
                new = s.lasso(rough_loop(np.any(members, axis=0), rng, widen=0.05), group=True)
                found = sum(any(iou(s.objects[j].mask, m) >= 0.5 for j in new) for m in members)
                matched = sum(any(iou(s.objects[j].mask, m) >= 0.5 for m in members) for j in new)
                groups.append({"image": name, "windows": len(members), "found": found, "new": len(new),
                               "extra": len(new) - matched})
        print(f"{k + 1}/{len(names)} images, {len(rows)} objects, {len(groups)} groups, {time.time() - t0:.0f}s")

    def mean(key, cls=None):
        v = [r[key] for r in rows if cls is None or r["class"] == cls]
        return round(float(np.mean(v)), 4) if v else None

    summary = {"objects": len(rows), "groups": len(groups)}
    for g in ("click", "loop", "scribble"):
        summary[g] = {"iou": mean(f"{g}_iou"), "class": mean(f"{g}_cls"),
                      "per_class_iou": {CLASS_NAMES[c]: mean(f"{g}_iou", c) for c in range(1, 6)}}
    summary["loop"]["union_iou"] = mean("loop_union_iou")
    summary["loop"]["objects_per_loop"] = mean("loop_n")
    summary["loop"]["empty"] = round(float(np.mean([r["loop_n"] == 0 for r in rows])), 4)
    if groups:
        summary["window_groups"] = {
            "windows_per_group": round(float(np.mean([g["windows"] for g in groups])), 2),
            "found_share": round(sum(g["found"] for g in groups) / sum(g["windows"] for g in groups), 4),
            "extra_per_loop": round(float(np.mean([g["extra"] for g in groups])), 2)}
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "objects": rows, "groups": groups}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
