"""Zero-shot tool evaluation: SAM masks + class from the semantic prior.

For every validation object of a clickable class:
- first-click IoU for each rule choosing among SAM's 3 masks (score / purity /
  region / oracle);
- class ranking from the prior inside the mask (true mask = upper bound, mask
  after 1 click, mask after 3 clicks): top-1 (left click right), top-2 (one right
  click), manual (the right class is 3rd or lower -> the user presses a key).

    python -m dmgseg.eval.run_tool_eval --out artifacts/results/tool_zeroshot.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dmgseg import paths
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.data.split import load_split
from dmgseg.eval.run_interactive import MIN_AREA
from dmgseg.eval.simulator import iou, next_click
from dmgseg.prior.kfold import load_prior
from dmgseg.sam.predictor import SamClicker
from dmgseg.tool.assign import CLICKABLE, choose_first_mask, class_ranking, fit_prior

RULES = ("score", "purity", "region", "oracle")
MASKS = ("true mask", "1 click", "3 clicks")


def evaluate(prior_dir, size="small", device="cpu", names=None, max_objects=None, seed=0):
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    names = names or load_split()["val"]
    rng = np.random.default_rng(seed)
    clicker = SamClicker(size, device)
    records, skipped_other = [], 0
    t0 = time.time()
    for name in tqdm(names, desc="images"):
        ann = anns[name]
        image = np.asarray(Image.open(paths.IMAGES_DIR / name).convert("RGB"))
        prior = fit_prior(load_prior(Path(prior_dir) / f"{name}.npz"), *image.shape[:2])
        objs = [o for o in image_objects(ann, MIN_AREA)]
        skipped_other += sum(o.label not in CLICKABLE for o in objs)
        objs = [o for o in objs if o.label in CLICKABLE]
        if max_objects and len(objs) > max_objects:
            objs = [objs[i] for i in rng.choice(len(objs), max_objects, replace=False)]
        clicker.set_image(image)
        clicker.choose = lambda m, s, x, y: choose_first_mask(m, s, prior, x, y, "region")
        for obj in objs:
            gt = obj.mask
            clicker.reset_object()
            x, y, _ = next_click(None, gt)
            pred1 = clicker.click(x, y, True)
            masks, scores = clicker.candidates
            first = {}
            for rule in RULES:
                k = (int(np.argmax([iou(m, gt) for m in masks])) if rule == "oracle"
                     else choose_first_mask(masks, scores, prior, x, y, rule))
                first[rule] = iou(masks[k], gt)
            pred = pred1
            for _ in range(2):  # clicks 2 and 3, continuing from the "region" choice
                cx, cy, pos = next_click(pred, gt)
                pred = clicker.click(cx, cy, pos)
            ranks = {}
            for key, m in zip(MASKS, (gt, pred1, pred)):
                ranks[key] = class_ranking(prior, m).index(obj.label)  # 0 = left click is right
            records.append({"image": name, "label": CLASS_NAMES[obj.label], "area": obj.area,
                            "first_click_iou": {k: round(v, 4) for k, v in first.items()},
                            "iou_3_clicks": round(iou(pred, gt), 4), "class_rank": ranks})
    return records, {"objects": len(records), "skipped_other": skipped_other,
                     "seconds": round(time.time() - t0, 1)}


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        groups[r["label"]].append(r)
    out = {}
    for g, rs in groups.items():
        s = {"n": len(rs)}
        for rule in RULES:
            s[f"iou1/{rule}"] = round(float(np.mean([r["first_click_iou"][rule] for r in rs])), 4)
        for key in MASKS:
            ranks = np.array([r["class_rank"][key] for r in rs])
            s[f"{key}/top1"] = round(float((ranks == 0).mean()), 4)
            s[f"{key}/top2"] = round(float((ranks <= 1).mean()), 4)
            s[f"{key}/manual"] = round(float((ranks >= 2).mean()), 4)
        out[g] = s
    return out


def print_summary(summary):
    order = ["all"] + [c for c in CLASS_NAMES if c in summary]
    print("First-click IoU by rule for choosing among SAM's 3 masks")
    print(f"{'class':15}{'n':>6}" + "".join(f"{r:>9}" for r in RULES))
    for g in order:
        s = summary[g]
        print(f"{g:15}{s['n']:>6}" + "".join(f"{s['iou1/' + r]:9.3f}" for r in RULES))
    print("\nClass from the prior: left click right (top-1) / after one right click (top-2) / manual key")
    print(f"{'class':15}" + "".join(f"{k:>24}" for k in MASKS))
    for g in order:
        s = summary[g]
        print(f"{g:15}" + "".join(f"{s[k + '/top1']:8.1%}{s[k + '/top2']:8.1%}{s[k + '/manual']:8.1%}" for k in MASKS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-dir", type=Path, default=paths.ARTIFACTS / "priors" / paths.PRIOR_RUN)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--max-objects", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    names = load_split()["val"][:args.max_images] if args.max_images else None
    records, meta = evaluate(args.prior_dir, names=names, max_objects=args.max_objects)
    summary = summarize(records)
    print_summary(summary)
    print(meta)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "meta": meta, "objects": records}))


if __name__ == "__main__":
    main()
