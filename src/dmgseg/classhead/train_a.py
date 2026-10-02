"""Train and evaluate class head A (MLP on description cards) on the Mac.

Cards come from the Hub (built on molab, section F). Early stopping uses a
development set: the training images of k-fold fold 0, held out from the head's
training. The 44 validation images are used only for the final evaluation.

    python -m dmgseg.classhead.train_a --out artifacts/classhead/head_a.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dmgseg import hub, paths
from dmgseg.classhead.build import load_cards
from dmgseg.classhead.mlp import evaluate_cards, train_head
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.data.split import load_kfold


def subset(cards, images):
    """Cards of the objects whose image is in `images` (object ids renumbered)."""
    keep_obj = np.where(np.isin(cards["obj_image"], list(images)))[0]
    remap = -np.ones(len(cards["obj_image"]), np.int64)
    remap[keep_obj] = np.arange(len(keep_obj))
    rows = np.isin(cards["obj"], keep_obj)
    out = {k: cards[k][rows] for k in ("X", "label", "iou", "obj", "cand", "click")}
    out["obj"] = remap[out["obj"]]
    for k in ("obj_image", "obj_shape", "obj_label", "obj_area"):
        out[k] = cards[k][keep_obj]
    return out


def fetch(name, local_dir=paths.ARTIFACTS):
    """name: "train", "val", or with a suffix such as "train_samft"."""
    path = Path(local_dir) / f"classhead/cards_{name}.npz"
    if not path.exists():
        hub.download_if_exists(f"classhead/cards_{name}.npz", local_dir)
    return load_cards(path)


def print_table(summary):
    rows = ["all"] + [c for c in CLASS_NAMES if c in summary]
    print(f"{'class':15}{'n':>6} | {'first-click IoU: SAM':>21}{'head':>7}{'oracle':>8} | "
          f"{'top-1: prior':>13}{'head':>7} | {'manual: prior':>14}{'head':>7}")
    for g in rows:
        s = summary[g]
        print(f"{g:15}{s['n']:>6} | {s['sam/iou1']:21.3f}{s['head/iou1']:7.3f}{s['oracle/iou1']:8.3f} | "
              f"{s['prior/top1@refined']:13.1%}{s['head/top1@refined']:7.1%} | "
              f"{s['prior/manual@refined']:14.1%}{s['head/manual@refined']:7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.ARTIFACTS / "classhead" / "head_a.pt")
    ap.add_argument("--seeds", type=int, default=3, help="train several seeds, report mean ± std")
    ap.add_argument("--cards", default="", help='card set suffix, e.g. "_samft" (fine-tuned SAM masks)')
    args = ap.parse_args()

    train_all, val = fetch("train" + args.cards), fetch("val" + args.cards)
    dev_images = set(load_kfold()[0]["heldout"])
    train_images = set(train_all["obj_image"]) - dev_images
    train, dev = subset(train_all, train_images), subset(train_all, dev_images)
    print(f"cards: train {len(train['X'])} ({len(train['obj_image'])} objects) | "
          f"dev {len(dev['X'])} ({len(dev['obj_image'])}) | val {len(val['X'])} ({len(val['obj_image'])})")

    results, best = [], None
    for seed in range(args.seeds):
        model = train_head(train, dev, seed=seed, log=lambda *a: None)
        d = evaluate_cards(model, dev)["all"]
        dev_score = d["head/top1@refined"] + d["head/iou1"]   # choose the saved model on dev, never on val
        s = evaluate_cards(model, val)
        results.append(s)
        if best is None or dev_score > best[1]:
            best = (model, dev_score)
        print(f"seed {seed}: val top-1 {s['all']['head/top1@refined']:.3f} | "
              f"first-click IoU {s['all']['head/iou1']:.3f} | manual {s['all']['head/manual@refined']:.3f}")

    print("\nValidation (44 images), the run of the first seed:")
    print_table(results[0])
    keys = ["head/top1@refined", "head/manual@refined", "head/iou1"]
    spread = {k: (float(np.mean([r["all"][k] for r in results])), float(np.std([r["all"][k] for r in results])))
              for k in keys}
    print("\nmean ± std over seeds:", {k: f"{m:.3f} ± {s:.3f}" for k, (m, s) in spread.items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best[0].state_dict(), args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"seeds": results, "spread": spread}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
