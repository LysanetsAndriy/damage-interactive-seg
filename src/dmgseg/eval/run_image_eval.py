"""Image-level evaluation: how good is a whole image after N interactions?

A simulated annotator starts from the automatic pre-label (or from nothing) and
repeatedly fixes the largest wrong region of the label map, choosing the action a
person would take; every action is one interaction:
    nothing labeled there but should be         -> left click (new object)
    an object there with mostly the right area   -> class fix (right click / key)
      but the wrong class
    a higher-priority class is hidden under a     -> left click (new object)
      lower-priority object
    an object covers pixels that should be other  -> Option+click (shrink it)
After each interaction the image's confusion matrix is stored; the global mIoU over
all validation images is reported per interaction count.

    python -m dmgseg.eval.run_image_eval --start prelabel --budget 20
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from dmgseg import paths
from dmgseg.app.engine import PAINT_ORDER, Session
from dmgseg.data.cvat import parse_annotations, semantic_mask
from dmgseg.data.split import load_split
from dmgseg.prior.kfold import load_prior

K = 6


def confusion(pred, gt):
    return np.bincount(gt.ravel().astype(np.int64) * K + pred.ravel(), minlength=K * K).reshape(K, K)


def miou(cm):
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(tp / (cm.sum(0) + cm.sum(1) - tp))


def largest_error(pred, gt, skip=()):
    """Deepest point (x, y) of the largest connected wrong region that does not
    contain an already-tried point, or None."""
    err = (pred != gt).astype(np.uint8)
    if not err.any():
        return None
    n, comp, stats, _ = cv2.connectedComponentsWithStats(err, connectivity=8)
    tried = {int(comp[y, x]) for x, y in skip}
    order = [i for i in (1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])) if int(i) not in tried]
    if not order:
        return None
    i = int(order[0])
    x0, y0, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    region = comp[y0:y0 + h, x0:x0 + w] == i
    dt = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return x + x0, y + y0


def act(session, gt, x, y):
    """Perform the one interaction a person would do at (x, y). Returns its name."""
    pred = session.label_map()
    g, p = int(gt[y, x]), int(pred[y, x])
    i = session.object_at(x, y)
    if i is None or p == 0 and g != 0 and session.objects[i].label != 0:
        session.new_object(x, y)
        return "new object"
    o = session.objects[i]
    if g == 0:
        session.refine(x, y, False, i)
        return "shrink"
    share = np.logical_and(o.mask, gt == g).sum() / max(o.mask.sum(), 1)
    if share > 0.5:                       # right area, wrong class
        if g in o.ranking and (o.ranking.index(g) - o.choice) % len(o.ranking) == 1 and o.manual is None:
            session.next_class(i)
            return "right click"
        session.set_class(i, g)
        return "class key"
    if PAINT_ORDER.index(g) > PAINT_ORDER.index(o.label):
        session.new_object(x, y)          # a higher-priority object hidden under this one
        return "new object"
    session.refine(x, y, False, i)        # the object spills over another class
    return "shrink"


def simulate(session, gt, budget, start, undo_worse=True):
    """undo_worse: like a person, undo an action that made the image worse (fewer
    correct pixels); the undo costs one more interaction and the spot is not
    tried again."""
    if start == "prelabel":
        session.prelabel()
    lm = session.label_map()
    cms, actions, tried = [confusion(lm, gt)], [], []
    correct = int((lm == gt).sum())
    while len(actions) < budget:
        pt = largest_error(lm, gt, tried)
        if pt is None:
            cms.append(cms[-1]); actions.append("none")
            continue
        a = act(session, gt, *pt)
        lm = session.label_map()
        new_correct = int((lm == gt).sum())
        if undo_worse and new_correct < correct:
            cms.append(confusion(lm, gt)); actions.append(a + " (worse)")
            tried.append(pt)
            if len(actions) < budget:
                session.undo()
                lm = session.label_map()
                cms.append(confusion(lm, gt)); actions.append("undo")
            continue
        correct = new_correct
        cms.append(confusion(lm, gt)); actions.append(a)
    return np.stack(cms), actions


def main():
    from dmgseg.app.engine import ClassHead
    from dmgseg.sam.predictor import SamClicker
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=["prelabel", "empty"], default="prelabel")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--sam-weights", type=Path, help="fine-tuned SAM decoder (sam/finetuned_decoder.pt)")
    ap.add_argument("--head", type=Path, default=paths.ARTIFACTS / "classhead" / "head_a.pt")
    ap.add_argument("--no-undo", action="store_true", help="keep actions that make the image worse")
    ap.add_argument("--priors", default=paths.PRIOR_RUN,
                    help="comma-separated prior runs whose probabilities are averaged for the draft "
                         "(e.g. an ensemble); the class head always gets the first one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    names = load_split()["val"][:args.max_images]
    clicker = SamClicker("small", "cpu", decoder_weights=args.sam_weights)
    head = ClassHead(args.head) if args.head and args.head.exists() else None
    total, per_image, action_log, t0 = None, {}, {}, time.time()
    for name in tqdm(names, desc=f"images ({args.start})"):
        image = np.asarray(Image.open(paths.IMAGES_DIR / name).convert("RGB"))
        gt = semantic_mask(anns[name])
        s = Session(image, clicker, head)
        runs = args.priors.split(",")
        priors = [load_prior(paths.ARTIFACTS / "priors" / r / f"{name}.npz") for r in runs]
        if len(priors) > 1:
            from dmgseg.tool.assign import fit_prior
            h, w = priors[0].shape[:2]
            s.set_prior(np.mean([fit_prior(p, h, w) for p in priors], axis=0), head_prior=priors[0])
        else:
            s.set_prior(priors[0])
        cms, actions = simulate(s, gt, args.budget, args.start, undo_worse=not args.no_undo)
        total = cms if total is None else total + cms
        per_image[name] = [round(float(miou(c)), 4) for c in cms]
        action_log[name] = actions
    curve = [round(float(miou(c)), 4) for c in total]
    print("global mIoU after k interactions:", {k: curve[k] for k in (0, 1, 2, 3, 5, 10, 15, 20) if k < len(curve)})
    counts = {}
    for a in action_log.values():
        for x in a:
            counts[x] = counts.get(x, 0) + 1
    print("actions used:", counts, f"| {time.time() - t0:.0f}s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "curve": curve,
                                        "per_image": per_image, "actions": action_log,
                                        "confusion": total.tolist()}, indent=1))


if __name__ == "__main__":
    main()
