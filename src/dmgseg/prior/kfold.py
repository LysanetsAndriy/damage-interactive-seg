"""Out-of-fold priors for the class head (PLAN §2.2, option A).

Each training image gets class probabilities from a model that did not see it:
5 folds over the 246 paper-training images; fold k trains on the other 4/5 and
predicts its held-out images. The paper model provides priors for the 44
validation images (it never saw them) and, for reference, all training images.

Priors are stored per image as uint8 probabilities (x255), shape (6, H, W), at the
resolution used for inference, under priors/<set>/<image>.npz on Hugging Face.
"""
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

from dmgseg import hub, paths
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.split import load_kfold, load_split
from dmgseg.prior.infer import predict_probs
from dmgseg.prior.train import train
from dmgseg.prior.unet_dinov2 import load_prior_model


def save_prior(path, probs):
    q = np.clip(np.rint(probs * 255), 0, 255).astype(np.uint8).transpose(2, 0, 1)
    np.savez_compressed(path, probs=q)


def load_prior(path):
    """-> float32 (H, W, C) probabilities."""
    return np.load(path)["probs"].transpose(1, 2, 0).astype(np.float32) / 255.0


def cache_priors(model, embed_model, names, out_dir, device, patch_size=518, stride=300,
                 image_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for name in tqdm(names, desc=f"priors -> {out_dir.name}"):
        target = out_dir / f"{name}.npz"
        if target.exists():
            continue
        image = Image.open(Path(image_dir or paths.IMAGES_DIR) / name).convert("RGB")
        probs, _ = predict_probs(model, embed_model, image, patch_size, stride, device=device)
        save_prior(target, probs)
    return out_dir


def _hub_has(path_in_repo):
    return HfApi().file_exists(paths.HF_REPO, path_in_repo, repo_type="dataset")


def _upload_dir(local_dir, path_in_repo, message):
    HfApi().upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo,
                          repo_id=paths.HF_REPO, repo_type="dataset", commit_message=message)


def model_priors(weights, set_name, embed_model, workdir, device, eval_cfg):
    """Priors of one model for all 290 images, stored as priors/<set_name>/ on the Hub.
    For the tool's prior (B2b) the validation images are out-of-sample."""
    if _hub_has(f"priors/{set_name}/_done.json"):
        print(f"{set_name} priors: already on the Hub")
        return
    split = load_split()
    model = load_prior_model(weights, device)
    out = cache_priors(model, embed_model, split["val"] + split["train"],
                       Path(workdir) / "priors" / set_name, device, **eval_cfg)
    (out / "_done.json").write_text(json.dumps({"weights": str(weights), "images": len(list(out.glob("*.npz")))}))
    _upload_dir(out, f"priors/{set_name}", f"Priors from {set_name} (all images)")
    del model
    torch.cuda.empty_cache()


def run_kfold(cfg, workdir, embed_model, embed_fn, device, folds=None, set_name="oof"):
    """Train each fold (resumable) and cache its held-out priors under
    priors/<set_name>/. Finished folds (_fold<k>_done.json on the Hub) are skipped."""
    folds_def = load_kfold()
    folds = range(len(folds_def)) if folds is None else folds
    out = Path(workdir) / "priors" / set_name
    base = cfg.run_name
    for k in folds:
        marker = f"_fold{k}_done.json"
        if _hub_has(f"priors/{set_name}/{marker}"):
            print(f"fold {k}: already done")
            continue
        fold_cfg = type(cfg)({**cfg, "run_name": f"{base}_fold{k}"})
        run_dir = train(fold_cfg, workdir, embed_fn, device=device, split=folds_def[k])
        model = load_prior_model(run_dir / "final.pt", device)
        cache_priors(model, embed_model, folds_def[k]["heldout"], out, device, **cfg.eval)
        (out / marker).write_text(json.dumps({"fold": k, "heldout": folds_def[k]["heldout"]}))
        _upload_dir(out, f"priors/{set_name}", f"Out-of-fold priors ({set_name}), fold {k}")
        del model
        torch.cuda.empty_cache()
        print(f"fold {k}: done")


def run_prior_job(cfg, workdir, embed_model, embed_fn, device, prior_weights, status=None):
    """Section D: tool-prior priors for all images, then the 5 folds.
    status(**fields) is called on each stage (see experiments.status_writer)."""
    status = status or (lambda **_: None)
    try:
        status(state="priors", stage="tool prior on all 290 images")
        model_priors(prior_weights, paths.PRIOR_RUN, embed_model, workdir, device, dict(cfg.eval))
        for k in range(len(load_kfold())):
            status(state="training", stage=f"fold {k}", run=f"{cfg.run_name}_fold{k}")
            run_kfold(cfg, workdir, embed_model, embed_fn, device, folds=[k], set_name=f"oof_{paths.PRIOR_RUN}")
        status(state="done", stage="all folds")
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise


def check_priors_complete(prior_dir, names):
    missing = [n for n in names if not (Path(prior_dir) / f"{n}.npz").exists()]
    return missing
