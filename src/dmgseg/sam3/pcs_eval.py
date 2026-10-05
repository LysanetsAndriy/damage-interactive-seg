"""Is SAM 3 (Promptable Concept Segmentation) useful for the damage tool?

E1  text prompts -> semantic map. Every class is prompted with a few phrases; SAM 3's
    per-prompt semantic map (sigmoid) is compared with the class mask. The best
    phrase per class is chosen on a development set (training images of k-fold
    fold 0), never on validation. On validation: per-class IoU, a 6-class map
    (classes painted in the dataset's priority order) and its fusion with the
    DINOv2 prior (B2b).
E2  exemplar prompts ("find all similar"): one ground-truth object box (Broken
    Window, Damage) as a positive exemplar, optionally with text; the share of the
    other objects of that class in the image found (IoU >= 0.5) and the number of
    extra detections.

    sam3_job(workdir, device, status)  # molab
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dmgseg import paths
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations, semantic_mask
from dmgseg.data.objects import image_objects

FIXED_KINDS = ("polygon", "box", "mask")
PHRASES = {
    1: ["building", "building facade", "apartment building", "wall of a building"],
    2: ["roof", "rooftop", "roof of a building"],
    3: ["damaged wall", "destroyed part of a building", "burnt facade", "collapsed wall", "damage"],
    4: ["broken window", "window", "shattered window", "empty window opening"],
    5: ["damaged roof", "collapsed roof", "hole in a roof", "burnt roof"],
}
PRIORITY = [1, 2, 3, 4, 5]          # paint order of the dataset (later = on top)
K = 6


def load(device):
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    return model, proc


@torch.no_grad()
def concept_maps(model, proc, image, phrases, device):
    """{phrase: probability map (H, W) float16 at the image size} + instance results."""
    inputs = proc(images=image, return_tensors="pt").to(device)
    vis = model.get_vision_features(pixel_values=inputs.pixel_values)
    h, w = image.size[1], image.size[0]
    maps, inst = {}, {}
    for p in phrases:
        t = proc(text=p, return_tensors="pt").to(device)
        out = model(vision_embeds=vis, **t)
        sem = torch.sigmoid(out.semantic_seg.float())                      # (1, 1, h', w')
        maps[p] = F.interpolate(sem, size=(h, w), mode="bilinear", align_corners=False)[0, 0].half().cpu()
        r = proc.post_process_instance_segmentation(out, threshold=0.4, mask_threshold=0.5,
                                                    target_sizes=inputs.get("original_sizes").tolist())[0]
        inst[p] = {"n": len(r["masks"]), "scores": r["scores"].float().cpu().numpy().round(3).tolist()}
    return maps, inst


def confusion(pred, gt):
    return np.bincount(gt.ravel().astype(np.int64) * K + pred.ravel(), minlength=K * K).reshape(K, K)


def miou(cm):
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    return float(np.nanmean(iou)), iou


def six_class(maps_by_class, h, w, thr=0.5):
    """Paint each class where its map > thr, in priority order; the rest is Other."""
    out = np.zeros((h, w), np.uint8)
    for c in PRIORITY:
        out[maps_by_class[c] > thr] = c
    return out


def probs6(maps_by_class, h, w):
    """SAM 3 maps as a 6-class distribution (Other = 1 - max), normalized."""
    p = np.zeros((h, w, K), np.float32)
    for c, m in maps_by_class.items():
        p[..., c] = m
    p[..., 0] = 1 - p[..., 1:].max(-1)
    return p / np.clip(p.sum(-1, keepdims=True), 1e-6, None)


def e1(model, proc, dev_anns, val_anns, device, prior_dir, status):
    all_phrases = [p for ps in PHRASES.values() for p in ps]
    # --- choose the best phrase per class on the development images (binary IoU)
    inter = {p: 0 for p in all_phrases}
    union = {p: 0 for p in all_phrases}
    cls_of = {p: c for c, ps in PHRASES.items() for p in ps}
    for k, ann in enumerate(dev_anns):
        image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
        gt = semantic_mask(ann, kinds=FIXED_KINDS)
        maps, _ = concept_maps(model, proc, image, all_phrases, device)
        for p, m in maps.items():
            pm, g = m.numpy() > 0.5, gt == cls_of[p]
            inter[p] += int((pm & g).sum())
            union[p] += int((pm | g).sum())
        status(state="E1 dev", stage=f"{k + 1}/{len(dev_anns)}")
    dev_iou = {p: inter[p] / max(union[p], 1) for p in all_phrases}
    best = {c: max(ps, key=lambda p: dev_iou[p]) for c, ps in PHRASES.items()}

    # --- validation
    from dmgseg.prior.kfold import load_prior
    from dmgseg.tool.assign import fit_prior
    v_inter = {p: 0 for p in all_phrases}
    v_union = {p: 0 for p in all_phrases}
    cms = {name: np.zeros((K, K), np.int64) for name in
           ("SAM3 text only", "DINOv2 prior (B2b)", "fusion 0.3 SAM3", "fusion 0.5 SAM3")}
    counts = {}
    for k, ann in enumerate(val_anns):
        image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
        gt = semantic_mask(ann, kinds=FIXED_KINDS)
        h, w = gt.shape
        maps, inst = concept_maps(model, proc, image, all_phrases, device)
        for p, m in maps.items():
            pm, g = m.numpy() > 0.5, gt == cls_of[p]
            v_inter[p] += int((pm & g).sum())
            v_union[p] += int((pm | g).sum())
        counts[ann.name] = {p: inst[p]["n"] for p in best.values()}
        by_class = {c: maps[best[c]].float().numpy() for c in PHRASES}
        cms["SAM3 text only"] += confusion(six_class(by_class, h, w), gt)
        prior = fit_prior(load_prior(Path(prior_dir) / f"{ann.name}.npz"), h, w)
        cms["DINOv2 prior (B2b)"] += confusion(prior.argmax(-1).astype(np.uint8), gt)
        s3 = probs6(by_class, h, w)
        for a in (0.3, 0.5):
            cms[f"fusion {a} SAM3"] += confusion(((1 - a) * prior + a * s3).argmax(-1).astype(np.uint8), gt)
        status(state="E1 val", stage=f"{k + 1}/{len(val_anns)}")
    val_iou = {p: v_inter[p] / max(v_union[p], 1) for p in all_phrases}
    six = {}
    for name, cm in cms.items():
        m, iou = miou(cm)
        six[name] = {"global/miou": m, "iou": {CLASS_NAMES[c]: float(iou[c]) for c in range(K)}}
    return {"best_phrase": {CLASS_NAMES[c]: p for c, p in best.items()},
            "dev_iou": dev_iou, "val_iou": val_iou, "six_class": six, "instances": counts}


def e2(model, proc, val_anns, device, status, classes=(4, 3)):
    """One GT object as exemplar box -> share of the other objects of its class found."""
    rng = np.random.default_rng(0)
    res = {}
    for c in classes:
        for mode in ("text", "exemplar", "exemplar + text"):
            res[f"{CLASS_NAMES[c]} | {mode}"] = {"found": 0, "others": 0, "extra": 0, "images": 0}
    phrase = {4: "window", 3: "damaged wall"}
    for k, ann in enumerate(val_anns):
        objs = [o for o in image_objects(ann, 100)]
        image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
        inputs = proc(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vis = model.get_vision_features(pixel_values=inputs.pixel_values)
        for c in classes:
            same = [o.mask for o in objs if o.label == c]
            if len(same) < 2:
                continue
            ex = int(rng.integers(len(same)))
            ys, xs = np.nonzero(same[ex])
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            others = [m for i, m in enumerate(same) if i != ex]
            for mode in ("text", "exemplar", "exemplar + text"):
                kw = {}
                if mode != "exemplar":
                    kw["text"] = phrase[c]
                if mode != "text":
                    kw["input_boxes"], kw["input_boxes_labels"] = [[box]], [[1]]
                t = proc(images=image, return_tensors="pt", **kw).to(device)
                t.pop("pixel_values", None)
                with torch.no_grad():
                    out = model(vision_embeds=vis, **t)
                r = proc.post_process_instance_segmentation(out, threshold=0.4, mask_threshold=0.5,
                                                            target_sizes=inputs.get("original_sizes").tolist())[0]
                pred = [m.cpu().numpy().astype(bool) for m in r["masks"]]
                pred = [m for m in pred if (m & same[ex]).sum() < 0.5 * m.sum()]   # drop the exemplar itself
                hit = [any((m & g).sum() / max((m | g).sum(), 1) >= 0.5 for m in pred) for g in others]
                matched = sum(any((m & g).sum() / max((m | g).sum(), 1) >= 0.5 for g in others) for m in pred)
                d = res[f"{CLASS_NAMES[c]} | {mode}"]
                d["found"] += int(sum(hit))
                d["others"] += len(others)
                d["extra"] += len(pred) - matched
                d["images"] += 1
        status(state="E2", stage=f"{k + 1}/{len(val_anns)}")
    for d in res.values():
        d["recall"] = d["found"] / max(d["others"], 1)
        d["extra_per_image"] = d["extra"] / max(d["images"], 1)
    return res


def table(r):
    rows = ["**E1: best phrase per class (chosen on dev) and its validation IoU**", "",
            "| class | phrase | val IoU | all phrases (val IoU) |", "|---|---|---|---|"]
    for c, ps in PHRASES.items():
        b = r["e1"]["best_phrase"][CLASS_NAMES[c]]
        rows.append(f"| {CLASS_NAMES[c]} | {b} | {r['e1']['val_iou'][b]:.3f} | "
                    + ", ".join(f"{p} {r['e1']['val_iou'][p]:.2f}" for p in ps) + " |")
    rows += ["", "| 6-class map | global mIoU | " + " | ".join(CLASS_NAMES) + " |", "|---|---|" + "---|" * K]
    for name, v in r["e1"]["six_class"].items():
        rows.append(f"| {name} | {v['global/miou']:.4f} | " + " | ".join(f"{v['iou'][c]:.3f}" for c in CLASS_NAMES) + " |")
    rows += ["", "**E2: one exemplar -> the other objects of the class**", "",
             "| class / prompt | images | other objects | found (IoU>=0.5) | extra per image |", "|---|---|---|---|---|"]
    for k, d in r["e2"].items():
        rows.append(f"| {k} | {d['images']} | {d['others']} | {d['recall']:.1%} | {d['extra_per_image']:.1f} |")
    return "\n".join(rows)


def sam3_job(workdir, device, status=None, max_images=None):
    from dmgseg import hub
    from dmgseg.data.split import load_kfold, load_split
    status = status or (lambda **_: None)
    workdir = Path(workdir)
    out = workdir / "sam3"
    out.mkdir(parents=True, exist_ok=True)
    try:
        anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
        dev = [anns[n] for n in load_kfold()[0]["heldout"]][:max_images]
        val = [anns[n] for n in load_split()["val"]][:max_images]
        prior_dir = workdir / "priors" / paths.PRIOR_RUN
        missing = [a.name for a in val if not (prior_dir / f"{a.name}.npz").exists()]
        if missing:
            from huggingface_hub import snapshot_download
            snapshot_download(paths.HF_REPO, repo_type="dataset", local_dir=str(workdir),
                              allow_patterns=[f"priors/{paths.PRIOR_RUN}/{n}.npz" for n in missing])
        status(state="loading SAM 3")
        t0 = time.time()
        model, proc = load(device)
        r = {"e1": e1(model, proc, dev, val, device, prior_dir, status)}
        r["e2"] = e2(model, proc, val, device, status)
        r["seconds"] = round(time.time() - t0)
        (out / "pcs_results.json").write_text(json.dumps(r, indent=1))
        if not max_images:
            hub.safe_upload(out / "pcs_results.json", "sam3/pcs_results.json", retries=6)
        status(state="done", table=table(r))
        return r
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise


def e2_filtered(workdir, device, status=None, thresholds=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """E2 for Broken Window with an exemplar, then the detections filtered by the
    DINOv2 prior: keep a detection if the prior's mean Broken Window probability
    inside it is >= t. -> recall of the other GT windows and extras per image per t."""
    from dmgseg.data.split import load_split
    from dmgseg.prior.kfold import load_prior
    from dmgseg.tool.assign import fit_prior
    status = status or (lambda **_: None)
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    val = [anns[n] for n in load_split()["val"]]
    prior_dir = Path(workdir) / "priors" / paths.PRIOR_RUN
    model, proc = load(device)
    rng = np.random.default_rng(0)
    rows = []                                   # per detection: image, prior p(BW), best IoU with a GT window
    n_others, n_images = 0, 0
    for k, ann in enumerate(val):
        same = [o.mask for o in image_objects(ann, 100) if o.label == 4]
        if len(same) < 2:
            continue
        image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
        ex = int(rng.integers(len(same)))
        ys, xs = np.nonzero(same[ex])
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        others = [m for i, m in enumerate(same) if i != ex]
        prior = fit_prior(load_prior(prior_dir / f"{ann.name}.npz"), ann.height, ann.width)[..., 4]
        t = proc(images=image, input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**t)
        r = proc.post_process_instance_segmentation(out, threshold=0.4, mask_threshold=0.5,
                                                    target_sizes=t.get("original_sizes").tolist())[0]
        for m in r["masks"]:
            m = m.cpu().numpy().astype(bool)
            if not m.any() or (m & same[ex]).sum() >= 0.5 * m.sum():
                continue
            ious = [(m & g).sum() / max((m | g).sum(), 1) for g in others]
            best = int(np.argmax(ious))
            rows.append((n_images, float(prior[m].mean()), float(ious[best]), best))
        n_others += len(others)
        n_images += 1
        status(state="E2 filtered", stage=f"{k + 1}/{len(val)}")
    out = {}
    for th in thresholds:
        kept = [r for r in rows if r[1] >= th]
        found = {(r[0], r[3]) for r in kept if r[2] >= 0.5}
        extra = sum(1 for r in kept if r[2] < 0.5)
        out[th] = {"recall": len(found) / max(n_others, 1), "extra_per_image": extra / max(n_images, 1)}
    return {"images": n_images, "others": n_others, "by_threshold": out}
