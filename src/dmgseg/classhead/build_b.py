"""Build the map-crop dataset for class head B (same simulated clicks as for A).

Per image a shard is written: crop maps (dino, sam, prior) and per-candidate
samples (mask map, scalars, targets, and the rule baselines for evaluation).

    shards = build_b(names, prior_dir, clicker, dino, out_dir, start_modes=("center", "random"))
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from dmgseg import paths
from dmgseg.classhead.build import _random_interior_click
from dmgseg.classhead.features import REFINED
from dmgseg.classhead.maps import crop_box, mask_map, prior_map, sam_map
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.eval.run_interactive import MIN_AREA
from dmgseg.eval.simulator import iou, next_click
from dmgseg.prior.kfold import load_prior
from dmgseg.tool.assign import CLICKABLE, fit_prior

N_SCALARS = 14


def scalars(mask, box, height, width, sam_score, cand_type, click_no):
    """Geometry (8, as in A's cards) + SAM score + candidate one-hot (4) + click/3,
    computed on the crop for speed."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    out = np.zeros(N_SCALARS, np.float32)
    if len(ys):
        area = int(mask.sum())
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        edges = sub ^ cv2.erode(sub.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[:8] = [np.log(area / (height * width)), bw / width, bh / height, np.log(bw / bh),
                   area / (bw * bh), (xs.mean() + x0) / width, (ys.mean() + y0) / height,
                   max(edges.sum(), 1) ** 2 / (4 * np.pi * area)]
    out[8] = sam_score
    out[9 + cand_type] = 1
    out[13] = click_no / 3
    return out


def _starts(mask, mode, rng):
    if mode == "center":
        return next_click(None, mask)[:2]
    return _random_interior_click(mask, rng, 0.3 if mode == "random" else 0.0)


def build_b(names, prior_dir, clicker, dino, out_dir, start_modes=("center",), refine_clicks=2,
            seed=0, progress=None, image_dir=None):
    """Writes one shard per image to out_dir (skips existing shards). Refinement
    clicks continue from the first start mode only."""
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i_img, name in enumerate(tqdm(names, desc="map crops")):
        if progress and i_img:
            progress(i_img, len(names))
        shard = out_dir / f"{name}.npz"
        if shard.exists():
            continue
        ann = anns[name]
        image = np.asarray(Image.open(Path(image_dir or paths.IMAGES_DIR) / name).convert("RGB"))
        h, w = image.shape[:2]
        prior = fit_prior(load_prior(Path(prior_dir) / f"{name}.npz"), h, w)
        clicker.set_image(image)
        embed = clicker.image_embedding
        crops, sams, priors, boxes = [], [], [], []
        S = {k: [] for k in ("crop", "mask", "scalars", "label", "iou", "obj_shape", "cand", "click",
                             "sam_score", "prior_mean")}

        for o in image_objects(ann, MIN_AREA):
            if o.label not in CLICKABLE:
                continue
            for s_i, mode in enumerate(start_modes):
                x, y = _starts(o.mask, mode, rng)
                clicker.reset_object()
                pred = clicker.click(x, y, True)
                masks, scores = clicker.candidates
                box = crop_box(list(masks), (x, y), h, w)
                c = len(boxes)
                boxes.append(box)
                crops.append(image[box[1]:box[3], box[0]:box[2]])
                sams.append(sam_map(embed, box, h, w).cpu().numpy().astype(np.float16))
                priors.append(prior_map(prior, box).astype(np.float16))

                def add(m, score, ctype, cno):
                    S["crop"].append(c); S["mask"].append(mask_map(m, box).astype(np.float16))
                    S["scalars"].append(scalars(m, box, h, w, score, ctype, cno))
                    S["label"].append(o.label); S["iou"].append(iou(m, o.mask)); S["obj_shape"].append(o.index)
                    S["cand"].append(ctype); S["click"].append(cno); S["sam_score"].append(score)
                    pm = prior[m].mean(0) if m.any() else np.full(6, 1 / 6)
                    S["prior_mean"].append(np.array([pm[k] for k in CLICKABLE], np.float32))

                for k in range(len(masks)):
                    add(masks[k], float(scores[k]), k, 1)
                if s_i == 0:
                    for r in range(refine_clicks):
                        cx, cy, pos = next_click(pred, o.mask)
                        pred = clicker.click(cx, cy, pos)
                        add(pred, clicker.last_score, REFINED, 2 + r)

        if not boxes:
            np.savez(shard, empty=np.array(True))
            continue
        dino_maps = dino.reduced(crops).cpu().numpy().astype(np.float16)
        np.savez(shard, dino=dino_maps, sam=np.stack(sams), prior=np.stack(priors),
                 boxes=np.array(boxes), **{k: np.stack(v) if k in ("mask", "scalars", "prior_mean")
                                           else np.array(v) for k, v in S.items()})
    return out_dir


def fit_dino_pca(dino, names, n_images=20, image_dir=None):
    """PCA for the DINOv2 tokens, fitted on crops around annotated objects of a few
    training images (unsupervised: only boxes are used, no labels)."""
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    crops = []
    for name in names[:n_images]:
        image = np.asarray(Image.open(Path(image_dir or paths.IMAGES_DIR) / name).convert("RGB"))
        h, w = image.shape[:2]
        for o in image_objects(anns[name], MIN_AREA)[:40]:
            ys, xs = np.nonzero(o.mask)
            x0, y0, x1, y1 = crop_box([o.mask], (int(xs.mean()), int(ys.mean())), h, w)
            crops.append(image[y0:y1, x0:x1])
    dino.fit_pca(crops)
    return len(crops)


def load_shards(shard_dir):
    """Concatenate shards -> dict of arrays; 'crop' indices and object ids made global."""
    out, n_crops, n_obj = {}, 0, 0
    for i, f in enumerate(sorted(Path(shard_dir).glob("*.npz"))):
        d = np.load(f)
        if "empty" in d.files:
            continue
        objs = d["obj_shape"]
        _, local_obj = np.unique(objs, return_inverse=True)
        parts = {"dino": d["dino"], "sam": d["sam"], "prior": d["prior"],
                 "crop": d["crop"] + n_crops, "obj": local_obj + n_obj,
                 "image": np.array([f.stem] * len(objs))}
        for k in ("mask", "scalars", "label", "iou", "cand", "click", "sam_score", "prior_mean", "obj_shape"):
            parts[k] = d[k]
        for k, v in parts.items():
            out.setdefault(k, []).append(v)
        n_crops += len(d["dino"])
        n_obj += local_obj.max() + 1
    return {k: np.concatenate(v) for k, v in out.items()}
