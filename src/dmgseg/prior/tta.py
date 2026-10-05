"""Test-time augmentation (flip, multi-scale) and ensembles of prior models.

No retraining: the same sliding-window inference is run on several views of the
image (mirrored, rescaled) and/or with several trained models, and the class
probabilities are averaged at the original resolution.

experiment(): evaluates combinations on the validation images in one pass (every
model x view is predicted once and reused by all combinations that contain it).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dmgseg import paths
from dmgseg.data.cvat import CLASS_NAMES, semantic_mask
from dmgseg.eval.metrics import SegMetrics, summary
from dmgseg.prior.infer import predict_probs
from dmgseg.prior.patches import upscale_if_small

FIXED_KINDS = ("polygon", "box", "mask")


def view_probs(model, embed_model, image, view, ev, device):
    """Probabilities [H, W, C] of `image` (PIL, already >= patch size) for one view:
    view = (scale, flip). Returned at the image's own resolution."""
    scale, flip = view
    w, h = image.size
    img = image.transpose(Image.FLIP_LEFT_RIGHT) if flip else image
    if scale != 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    probs, used = predict_probs(model, embed_model, img, device=device, batch_size=8, **ev)
    p = torch.from_numpy(probs).permute(2, 0, 1)[None]
    if p.shape[-2:] != (h, w):            # rescaled view, or upscaled because too small
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
    p = p[0].permute(1, 2, 0).numpy()
    return p[:, ::-1] if flip else p


def experiment(models, combos, images, embed_model, device, status=None):
    """models: {name: (model, eval dict)}; combos: {combo name: [(model name, view)]}.
    -> {combo: summary of global/paper metrics on fixed labels}."""
    metrics = {c: SegMetrics(len(CLASS_NAMES)) for c in combos}
    needed = sorted({(m, v) for parts in combos.values() for m, v in parts})
    t0 = time.time()
    for k, ann in enumerate(images):
        image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
        # the notebook upscales small images to the patch size; use the largest one
        biggest = max(ev["patch_size"] for _, ev in models.values())
        image, _ = upscale_if_small(image, None, biggest)
        true = semantic_mask(ann, kinds=FIXED_KINDS)
        if true.shape != (image.size[1], image.size[0]):
            true = np.array(Image.fromarray(true).resize(image.size, Image.NEAREST))
        cache = {}
        for m, v in needed:
            model, ev = models[m]
            with torch.no_grad():
                cache[(m, v)] = view_probs(model, embed_model, image, v, ev, device)
        for c, parts in combos.items():
            pred = np.mean([cache[p] for p in parts], axis=0).argmax(-1).astype(np.uint8)
            metrics[c].update(pred, true)
        if status:
            status(state="evaluating", stage=f"image {k + 1}/{len(images)}", seconds=round(time.time() - t0))
    return {c: summary(m.compute(), CLASS_NAMES) for c, m in metrics.items()}


def results_table(results):
    head = "| combination | global mIoU | global mF1 | paper mIoU | " + " | ".join(CLASS_NAMES) + " |"
    rows = [head, "|" + "---|" * (4 + len(CLASS_NAMES))]
    for name, v in results.items():
        rows.append(f"| {name} | {v['global/miou']:.4f} | {v['global/mf1']:.4f} | {v['paper/miou']:.4f} | "
                    + " | ".join(f"{v['global/iou'][c]:.3f}" for c in CLASS_NAMES) + " |")
    return "\n".join(rows)


def tta_job(workdir, device, status=None):
    """The molab job: DINOv2 priors (paper, B2b, P1-P3) with flip / multi-scale TTA
    and ensembles, on the 44 validation images; result in runs/tta_queue/."""
    from dmgseg import hub
    from dmgseg.config import load_config
    from dmgseg.prior.embedding import load_embedding_model
    from dmgseg.prior.train import split_annotations
    from dmgseg.prior.unet_dinov2 import load_prior_model

    workdir = Path(workdir)
    repo = paths.PROJECT_ROOT
    specs = {"paper": (paths.PAPER_WEIGHTS, None, {"patch_size": 518, "stride": 300, "model_input": 518})}
    for name, cfg_name in (("B2b", "b2b"), ("P1", "p1"), ("P2", "p2"), ("P3", "p3")):
        cfg = load_config(repo / "configs" / f"prior_dinov2_6c_{cfg_name}.yaml")
        w = hub.download_if_exists(f"runs/{cfg.run_name}/final.pt", workdir)
        ev = dict(cfg.get("eval", {}))
        ev.setdefault("model_input", cfg.data.model_input)
        specs[name] = (w, cfg.model.get("img_size"), ev)
    models = {n: (load_prior_model(w, device, img_size=s), ev) for n, (w, s, ev) in specs.items()}
    base, flip = (1.0, False), (1.0, True)
    ms = [(s, f) for s in (0.75, 1.0, 1.25) for f in (False, True)]
    combos = {
        "B2b": [("B2b", base)],
        "B2b + flip": [("B2b", base), ("B2b", flip)],
        "B2b + flip + scales 0.75/1.25": [("B2b", v) for v in ms],
        "B2b + P2 + P3": [(m, base) for m in ("B2b", "P2", "P3")],
        "B2b + P2 + P3 + flip": [(m, v) for m in ("B2b", "P2", "P3") for v in (base, flip)],
        "all 5 DINOv2": [(m, base) for m in models],
        "all 5 DINOv2 + flip": [(m, v) for m in models for v in (base, flip)],
    }
    _, val = split_annotations()
    embed_model = load_embedding_model(device)
    results = experiment(models, combos, val, embed_model, device, status)
    out = workdir / "runs" / "tta_queue" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    hub.upload(out, "runs/tta_queue/results.json")
    if status:
        status(state="done", table=results_table(results))
    return results


def cache_val_priors(workdir, device, runs=("p2", "p3"), status=None):
    """Full-image priors of the validation images for other runs (as kfold.py does
    for B2b), saved to priors/<run_name>/ locally and on the Hub, so ensembles can
    be evaluated in the tool's image-level simulation on the Mac."""
    from dmgseg import hub
    from dmgseg.config import load_config
    from dmgseg.prior.embedding import load_embedding_model
    from dmgseg.prior.kfold import save_prior
    from dmgseg.prior.train import split_annotations
    from dmgseg.prior.unet_dinov2 import load_prior_model
    workdir = Path(workdir)
    _, val = split_annotations()
    embed_model = load_embedding_model(device)
    for r in runs:
        cfg = load_config(paths.PROJECT_ROOT / "configs" / f"prior_dinov2_6c_{r}.yaml")
        ev = dict(cfg.get("eval", {}))
        ev.setdefault("model_input", cfg.data.model_input)
        w = hub.download_if_exists(f"runs/{cfg.run_name}/final.pt", workdir)
        model = load_prior_model(w, device, img_size=cfg.model.get("img_size"))
        out = workdir / "priors" / cfg.run_name
        out.mkdir(parents=True, exist_ok=True)
        for k, ann in enumerate(val):
            image = Image.open(paths.IMAGES_DIR / ann.name).convert("RGB")
            with torch.no_grad():
                probs, _ = predict_probs(model, embed_model, image, device=device, batch_size=8, **ev)
            save_prior(out / f"{ann.name}.npz", probs)
            if status:
                status(state="caching", stage=f"{cfg.run_name} {k + 1}/{len(val)}")
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), path_in_repo=f"priors/{cfg.run_name}",
                              repo_id=paths.HF_REPO, repo_type="dataset")
    if status:
        status(state="done")
