"""Class head B on molab (notebook section H): data, training, ablations, results.

1. DINOv2 PCA on crops of 20 training images.
2. Map crops: train (246 images, center + random first clicks, out-of-fold
   priors), val (44, center, B2b priors), val_random (44, random first clicks).
3. Variants full / no_dino / no_sam, 3 seeds each; the saved seed is chosen on the
   dev set (fold-0 training images); validation is evaluation only.
4. classhead/head_b.pt (best full model + PCA) and classhead/head_b_results.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from dmgseg import hub, paths
from dmgseg.classhead.build_b import build_b, fit_dino_pca, load_shards
from dmgseg.classhead.fusion import GpuData, evaluate, subset, train_fusion
from dmgseg.classhead.maps import DinoCrops
from dmgseg.data.split import load_kfold, load_split

VARIANTS = {"full": (), "no_dino": ("dino",), "no_sam": ("sam",)}


def h_job(workdir, device, status=None, seeds=3, max_images=None, epochs=40):
    status = status or (lambda **_: None)
    workdir = Path(workdir)
    root = workdir / "classhead_b"
    try:
        split = load_split()
        train_names, val_names = split["train"][:max_images], split["val"][:max_images]
        status(state="downloading priors")
        snapshot_download(paths.HF_REPO, repo_type="dataset", local_dir=str(workdir),
                          allow_patterns=[f"priors/oof_{paths.PRIOR_RUN}/*.npz"]
                          + [f"priors/{paths.PRIOR_RUN}/{n}.npz" for n in split["val"]])
        from dmgseg.sam.predictor import SamClicker
        clicker = SamClicker("small", device)
        dino = DinoCrops(device)
        pca_path = root / "dino_pca.pt"
        if pca_path.exists():
            p = torch.load(pca_path)
            dino.pca_mean, dino.pca_basis = p["mean"].to(device), p["basis"].to(device)
        else:
            status(state="fitting PCA")
            fit_dino_pca(dino, train_names)
            root.mkdir(parents=True, exist_ok=True)
            torch.save({"mean": dino.pca_mean.cpu(), "basis": dino.pca_basis.cpu()}, pca_path)

        sets = {
            "train": (train_names, workdir / "priors" / f"oof_{paths.PRIOR_RUN}", ("center", "random")),
            "val": (val_names, workdir / "priors" / paths.PRIOR_RUN, ("center",)),
            "val_random": (val_names, workdir / "priors" / paths.PRIOR_RUN, ("random",)),
        }
        for name, (names, prior_dir, modes) in sets.items():
            status(state="building", stage=f"{name}: 0/{len(names)} images")
            build_b(names, prior_dir, clicker, dino, root / name, start_modes=modes, seed=7,
                    progress=lambda d, t, n=name: status(stage=f"{n}: {d}/{t} images") if d % 10 == 0 else None)
        del clicker, dino
        torch.cuda.empty_cache()

        status(state="training", stage="loading data")
        data = {n: load_shards(root / n) for n in sets}
        dev_images = set(load_kfold()[0]["heldout"])
        tr = subset(data["train"], set(data["train"]["image"]) - dev_images)
        dv = subset(data["train"], set(data["train"]["image"]) & dev_images)
        results, best_full = {}, None
        for variant, drop in VARIANTS.items():
            g_tr, g_dv = GpuData(tr, device, drop), GpuData(dv, device, drop)
            g_val, g_vr = GpuData(data["val"], device, drop), GpuData(data["val_random"], device, drop)
            runs = []
            for seed in range(seeds):
                status(stage=f"{variant}, seed {seed}")
                model, dev_score = train_fusion(g_tr, g_dv, seed=seed, epochs=epochs, log=lambda *a: None)
                runs.append({"seed": seed, "dev_score": dev_score,
                             "val": evaluate(model, g_val), "val_random": evaluate(model, g_vr)})
                if variant == "full" and (best_full is None or dev_score > best_full[1]):
                    best_full = ({k: v.cpu() for k, v in model.state_dict().items()}, dev_score)
            results[variant] = runs
            del g_tr, g_dv, g_val, g_vr
            torch.cuda.empty_cache()

        out = root / "head_b_results.json"
        out.write_text(json.dumps(results, indent=1))
        torch.save({"model": best_full[0], "pca": torch.load(pca_path)}, root / "head_b.pt")
        if max_images is None:
            hub.upload(out, "classhead/head_b_results.json", message="Class head B results (variants x seeds)")
            hub.upload(root / "head_b.pt", "classhead/head_b.pt", message="Class head B (full, dev-selected seed) + DINOv2 PCA")
        status(state="done", table=results_table(results))
        return results
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise


def results_table(results):
    keys = [("head/top1@refined", "right class"), ("head/manual@refined", "manual"), ("head/iou1", "1st-click IoU")]
    lines = ["| variant | val: " + " | ".join(k[1] for k in keys) + " | random clicks: " + " | ".join(k[1] for k in keys) + " |",
             "|---|" + "---|" * (2 * len(keys))]
    for variant, runs in results.items():
        cells = []
        for part in ("val", "val_random"):
            for k, _ in keys:
                v = np.array([r[part]["all"][k] for r in runs])
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    base = results[next(iter(results))][0]["val"]["all"]
    lines.append(f"\nRules on the same masks: right class {base['prior/top1@refined']:.3f}, "
                 f"manual {base['prior/manual@refined']:.3f}, SAM first-click IoU {base['sam/iou1']:.3f} "
                 f"(oracle {base['oracle/iou1']:.3f})")
    return "\n".join(lines)
