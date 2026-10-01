"""Object-level interactive evaluation on the paper's validation images.

    python -m dmgseg.eval.run_interactive --size small --max-images 5 --max-objects 8
"""
import argparse
import json
import random
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
from dmgseg.eval.simulator import simulate_object, summarize

MIN_AREA = 100  # pixels; tinier objects are skipped (reported in the output)


def evaluate(clicker, image_names=None, max_clicks=20, min_area=MIN_AREA,
             max_objects=None, seed=0, image_dir=None):
    """Returns (summary dict, per-object records)."""
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    image_names = image_names or load_split()["val"]
    rng = random.Random(seed)
    traces, records, skipped = defaultdict(list), [], 0
    t0 = time.time()
    for name in tqdm(image_names, desc="images"):
        ann = anns[name]
        objs = image_objects(ann)
        kept = [o for o in objs if o.area >= min_area]
        skipped += len(objs) - len(kept)
        if max_objects and len(kept) > max_objects:
            kept = rng.sample(kept, max_objects)
        clicker.set_image(Image.open(Path(image_dir or paths.IMAGES_DIR) / name).convert("RGB"))
        for obj in kept:
            clicker.oracle_gt = obj.mask
            trace = simulate_object(clicker, obj.mask, max_clicks=max_clicks)
            traces[obj.label].append(trace)
            records.append({"image": name, "shape": obj.index, "label": CLASS_NAMES[obj.label],
                            "area": obj.area, "ious": [round(v, 4) for v in trace.ious]})
    summary = summarize(traces, CLASS_NAMES, max_clicks=max_clicks)
    summary["meta"] = {"images": len(image_names), "objects": len(records),
                       "skipped_small": skipped, "min_area": min_area,
                       "max_clicks": max_clicks, "seconds": round(time.time() - t0, 1)}
    return summary, records


def print_summary(summary, ks=(1, 3, 5, 10)):
    print(f"{'class':15}{'n':>6}" + "".join(f"{'IoU@'+str(k):>9}" for k in ks)
          + f"{'NoC@85':>9}{'NoC@90':>9}{'fail@90':>9}")
    for name, s in summary.items():
        if name == "meta" or s is None:
            continue
        m = s["miou@k"]
        print(f"{name:15}{s['n']:>6}" + "".join(f"{m[k - 1]:9.3f}" for k in ks if k <= len(m))
              + f"{s['noc@85']:9.2f}{s['noc@90']:9.2f}{s['fail@90']:9.1%}")
    print(summary["meta"])


def main():
    from dmgseg.sam.predictor import SamClicker

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--choose", default="score", choices=["score", "oracle"])
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--max-objects", type=int)
    ap.add_argument("--max-clicks", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    names = load_split()["val"][:args.max_images] if args.max_images else None
    clicker = SamClicker(args.size, args.device, choose=args.choose)
    summary, records = evaluate(clicker, names, max_clicks=args.max_clicks, max_objects=args.max_objects)
    print_summary(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"args": vars(args) | {"out": str(args.out)},
                                        "summary": summary, "objects": records}))


if __name__ == "__main__":
    main()
