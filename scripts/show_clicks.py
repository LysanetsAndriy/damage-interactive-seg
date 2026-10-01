"""Visualize the simulated user + SAM on validation objects.

For each picked object: crop around it, true outline (yellow), SAM mask (blue),
positive clicks (green) and negative clicks (red), after 1, 3, 5 and 10 clicks.

    python scripts/show_clicks.py --image <val image name> --out artifacts/figures/clicks.png
"""
import argparse
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
from dmgseg.sam.predictor import SamClicker

STEPS = (1, 3, 5, 10)


def run_object(clicker, gt, n):
    clicker.reset_object()
    pred, clicks, masks = None, [], []
    for _ in range(n):
        x, y, pos = next_click(pred, gt)
        clicks.append((x, y, pos))
        pred = clicker.click(x, y, pos)
        masks.append(pred)
    return clicks, masks


def crop_box(mask, h, w, pad=0.25):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0) * pad) + 20, int((x1 - x0) * pad) + 20
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="validation image name (default: one with many classes)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures/clicks.png"))
    args = ap.parse_args()

    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    val = load_split()["val"]
    if args.image:
        name = args.image
    else:  # the validation image with the most distinct classes
        name = max(val, key=lambda n: (len({o.label for o in image_objects(anns[n], 400)}), n))
    ann = anns[name]
    image = np.array(Image.open(paths.IMAGES_DIR / name).convert("RGB"))

    # one reasonably large object per class
    by_class = {}
    for o in sorted(image_objects(ann, 400), key=lambda o: -o.area):
        by_class.setdefault(o.label, o)
    objects = [by_class[c] for c in sorted(by_class)]

    clicker = SamClicker("small", "cpu")
    clicker.set_image(image)

    fig, axes = plt.subplots(len(objects), len(STEPS), figsize=(4 * len(STEPS), 3.6 * len(objects)))
    axes = np.atleast_2d(axes)
    for row, obj in zip(axes, objects):
        clicks, masks = run_object(clicker, obj.mask, max(STEPS))
        y0, y1, x0, x1 = crop_box(obj.mask, *obj.mask.shape)
        for ax, k in zip(row, STEPS):
            pred = masks[k - 1]
            view = image[y0:y1, x0:x1].astype(float) / 255
            overlay = view.copy()
            overlay[pred[y0:y1, x0:x1]] = 0.55 * overlay[pred[y0:y1, x0:x1]] + 0.45 * np.array([0.1, 0.4, 1.0])
            overlay[find_boundaries(obj.mask[y0:y1, x0:x1], mode="inner")] = [1, 0.9, 0]
            ax.imshow(overlay)
            for x, y, pos in clicks[:k]:
                ax.plot(x - x0, y - y0, "o", ms=9, mec="white", mew=1.5, color="#1a9850" if pos else "#d73027")
            ax.set_title(f"{CLASS_NAMES[obj.label]} | {k} click{'s' * (k > 1)} | IoU {iou(pred, obj.mask):.2f}", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"{name}: yellow = true outline, blue = SAM mask, green/red = positive/negative clicks", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(args.out)


if __name__ == "__main__":
    main()
