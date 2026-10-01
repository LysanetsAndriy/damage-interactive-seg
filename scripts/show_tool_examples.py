"""Figures explaining how SAM and the DINOv2 prior are combined.

class_choice.png: for real validation objects, the photo, the prior's map and
the class scores inside the true mask (right and wrong cases).
mask_choice.png: SAM's 3 first-click masks, which one SAM's score picks and
which one the prior-region rule picks.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

from dmgseg import paths
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.data.split import load_split
from dmgseg.eval.simulator import iou, next_click
from dmgseg.prior.kfold import load_prior
from dmgseg.tool.assign import CLICKABLE, choose_first_mask, class_scores, fit_prior, prior_region

COLORS = np.array([[160, 160, 160], [40, 170, 60], [255, 165, 0], [170, 50, 200], [30, 90, 255], [230, 30, 30]]) / 255
OUT = Path("artifacts/figures")


def crop(mask, h, w, pad=0.6, minimum=60):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min(), minimum) * (0.5 + pad))
    return max(0, cy - half), min(h, cy + half), max(0, cx - half), min(w, cx + half)


def outline(ax, mask, box, color="yellow"):
    y0, y1, x0, x1 = box
    b = find_boundaries(mask[y0:y1, x0:x1], mode="inner")
    ys, xs = np.nonzero(b)
    ax.plot(xs, ys, ",", color=color)
    ax.scatter(xs, ys, s=0.6, c=color, linewidths=0)


def load(anns, name):
    a = anns[name]
    img = np.asarray(Image.open(paths.IMAGES_DIR / name).convert("RGB"))
    prior = fit_prior(load_prior(paths.ARTIFACTS / "priors" / paths.PRIOR_RUN / f"{name}.npz"), a.height, a.width)
    return a, img, prior


def pick_class_examples(anns, val):
    """Right and wrong cases, mid-sized objects, deterministic."""
    wanted = [("Broken Window", True), ("Broken Window", False), ("Damage", False), ("Roof", False)]
    found = {}
    for name in val:
        a, _, prior = load(anns, name)
        for o in image_objects(a, 600):
            if o.label not in CLICKABLE:
                continue
            key = (CLASS_NAMES[o.label], max(class_scores(prior, o.mask).items(), key=lambda kv: kv[1])[0] == o.label)
            if key in wanted and key not in found and o.area < 40000:
                found[key] = (name, o)
        if len(found) == len(wanted):
            break
    return [found[k] for k in wanted if k in found]


def class_figure(anns, val):
    examples = pick_class_examples(anns, val)
    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 3.6 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for row, (name, o) in zip(np.atleast_2d(axes), examples):
        a, img, prior = load(anns, name)
        box = crop(o.mask, a.height, a.width)
        y0, y1, x0, x1 = box
        row[0].imshow(img[y0:y1, x0:x1]); outline(row[0], o.mask, box)
        row[0].set_title(f"photo · true class: {CLASS_NAMES[o.label]} ({o.area} px)", fontsize=10)
        row[1].imshow(COLORS[prior.argmax(-1)[y0:y1, x0:x1]]); outline(row[1], o.mask, box, "black")
        row[1].set_title("what DINOv2 (the prior) thinks, per pixel", fontsize=10)
        sc = class_scores(prior, o.mask)
        names = [CLASS_NAMES[c] for c in CLICKABLE]
        vals = [sc[c] for c in CLICKABLE]
        bars = row[2].barh(names, vals, color=[COLORS[c] for c in CLICKABLE])
        best = max(sc, key=sc.get)
        row[2].set_xlim(0, 1); row[2].invert_yaxis()
        row[2].set_title(f"average inside the yellow outline → picks {CLASS_NAMES[best]} "
                         f"{'✓' if best == o.label else '✗'}", fontsize=10)
        bars[CLICKABLE.index(o.label)].set_edgecolor("black"); bars[CLICKABLE.index(o.label)].set_linewidth(2)
        for ax in row[:2]:
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in range(6)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("Which class does a clicked object get?  Yellow = the object (true outline).", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / "class_choice.png", dpi=85)


def mask_figure(anns, val):
    from dmgseg.sam.predictor import SamClicker
    clicker = SamClicker("small", "cpu")
    # one large object where the prior rule helps, one small where it hurts
    cases, want = [], [("Building", "region"), ("Broken Window", "score")]
    for name in val:
        a, img, prior = load(anns, name)
        objs = [o for o in image_objects(a, 600) if (CLASS_NAMES[o.label], None) and o.label in CLICKABLE]
        todo = [w for w in want if w[0] not in [c[2].label and CLASS_NAMES[c[2].label] for c in cases]]
        objs = [o for o in objs if CLASS_NAMES[o.label] in [w[0] for w in todo]]
        if not objs:
            continue
        clicker.set_image(img)
        for o in objs:
            clicker.reset_object()
            x, y, _ = next_click(None, o.mask)
            clicker.click(x, y)
            masks, scores = clicker.candidates
            ious = [iou(m, o.mask) for m in masks]
            ks, kr = int(np.argmax(scores)), choose_first_mask(masks, scores, prior, x, y, "region")
            better = "region" if ious[kr] > ious[ks] + 0.2 else "score" if ious[ks] > ious[kr] + 0.2 else None
            if (CLASS_NAMES[o.label], better) in want and CLASS_NAMES[o.label] not in [CLASS_NAMES[c[2].label] for c in cases]:
                cases.append((name, img, o, prior, x, y, masks, scores, ious, ks, kr))
                break
        if len(cases) == len(want):
            break

    fig, axes = plt.subplots(len(cases), 5, figsize=(19, 4 * len(cases)))
    for row, (name, img, o, prior, x, y, masks, scores, ious, ks, kr) in zip(np.atleast_2d(axes), cases):
        box = crop(np.logical_or.reduce([o.mask, *masks]), *o.mask.shape, pad=0.1)
        y0, y1, x0, x1 = box
        row[0].imshow(img[y0:y1, x0:x1]); outline(row[0], o.mask, box)
        row[0].plot(x - x0, y - y0, "o", ms=10, mec="white", color="#1a9850")
        row[0].set_title(f"click on a {CLASS_NAMES[o.label]} (yellow = true outline)", fontsize=10)
        for i in range(3):
            ax = row[1 + i]
            view = img[y0:y1, x0:x1].astype(float) / 255
            m = masks[i][y0:y1, x0:x1]
            view[m] = 0.5 * view[m] + 0.5 * np.array([0.1, 0.4, 1.0])
            ax.imshow(view); outline(ax, o.mask, box)
            tags = (" ← SAM picks" if i == ks else "") + (" ← prior rule picks" if i == kr else "")
            ax.set_title(f"SAM mask {i + 1}: SAM confidence {scores[i]:.2f}\ntrue IoU {ious[i]:.2f}{tags}", fontsize=10)
        reg = prior_region(prior, x, y)
        row[4].imshow(COLORS[prior.argmax(-1)[y0:y1, x0:x1]])
        rv = np.zeros((y1 - y0, x1 - x0, 4)); rv[reg[y0:y1, x0:x1]] = [1, 1, 1, 0.45]
        row[4].imshow(rv); row[4].plot(x - x0, y - y0, "o", ms=10, mec="white", color="#1a9850")
        row[4].set_title("prior's map; white = the prior's\n'object' around the click", fontsize=10)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Which of SAM's 3 masks to keep after the first click?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "mask_choice.png", dpi=80)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    val = load_split()["val"]
    class_figure(anns, val)
    mask_figure(anns, val)
    print(OUT / "class_choice.png", OUT / "mask_choice.png")


if __name__ == "__main__":
    main()
