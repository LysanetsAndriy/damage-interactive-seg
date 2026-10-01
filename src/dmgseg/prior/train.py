"""Training of the DINOv2-Emb U-Net, ported from the notebook's train/train_loop/eval_loop.

Additions over the notebook (training itself is unchanged):
- checkpoint every epoch and resume from it (molab sessions end after 12 h);
- checkpoints and metrics pushed to the Hugging Face repo under runs/<run_name>/;
- metrics in both conventions (see dmgseg.eval.metrics).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmgseg import hub, paths
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations
from dmgseg.data.split import load_split
from dmgseg.eval.metrics import SegMetrics, summary
from dmgseg.prior.dataset import PatchedDataset, train_transform
from dmgseg.prior.losses import CombinedLoss
from dmgseg.prior.patches import build_patch_dataset
from dmgseg.prior.unet_dinov2 import UNetDinoV2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_annotations():
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    split = load_split()
    return [anns[n] for n in split["train"]], [anns[n] for n in split["val"]]


def prepare_patches(cfg, embed_fn, workdir):
    """Build train/val patch folders once; reuse them if already complete."""
    train_anns, val_anns = split_annotations()
    limit = cfg.data.get("limit_images")  # smoke tests only
    if limit:
        train_anns, val_anns = train_anns[:limit], val_anns[:limit]
    dirs = {}
    for name, anns in (("train", train_anns), ("val", val_anns)):
        tag = f"{cfg.data.patch_size}_{cfg.data.stride}" + (f"_first{limit}" if limit else "")
        d = Path(workdir) / f"patches_{tag}" / name
        done = d / "_done.json"
        if not done.exists():
            n = build_patch_dataset(anns, d, cfg.data.patch_size, cfg.data.stride, embed_fn,
                                    tuple(cfg.data.label_kinds))
            done.write_text(json.dumps({"patches": n}))
        dirs[name] = d
    return dirs


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, device, amp_dtype, optimizer=None, scaler=None):
    """One pass over loader. Trains if optimizer is given. Returns (mean loss, SegMetrics)."""
    training = optimizer is not None
    model.train(training)
    metrics = SegMetrics(model.num_classes)
    total_loss, n = 0.0, 0
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for imgs, masks, emb, pos in tqdm(loader, desc="train" if training else "val", leave=False):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            emb, pos = emb.to(device), pos.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(imgs, emb, pos)
                loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            metrics.update(logits.argmax(1), masks)
    return total_loss / max(n, 1), metrics


def train(cfg, workdir, embed_fn, device=None, push=True, max_batches=None, split=None):
    """Train with cfg (see configs/prior_dinov2_6c_paper.yaml).

    workdir: local folder for patches and checkpoints.
    embed_fn: PIL image -> 1536-d global embedding (used only to build patches).
    max_batches: limit batches per epoch (smoke tests only).
    split: optional {"train": names, "heldout": names}, both subsets of the paper's
        training images (k-fold). Default: paper train / paper validation.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workdir = Path(workdir)
    run_dir = workdir / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    hub_prefix = f"runs/{cfg.run_name}"

    set_seed(cfg.seed)
    dirs = prepare_patches(cfg, embed_fn, workdir)
    size = cfg.data.model_input
    if split is None:
        train_ds = PatchedDataset(dirs["train"], size=size, transform=train_transform(size))
        val_ds = PatchedDataset(dirs["val"], size=size)
    else:
        train_ds = PatchedDataset(dirs["train"], size=size, transform=train_transform(size),
                                  sources=split["train"])
        val_ds = PatchedDataset(dirs["train"], size=size, sources=split["heldout"])
    if max_batches:
        bs = cfg.train.batch_size
        train_ds.files = train_ds.files[:max_batches * bs]
        val_ds.files = val_ds.files[:max_batches * bs]
    loader_kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                     pin_memory=device.type == "cuda", persistent_workers=cfg.train.num_workers > 0)
    if cfg.train.num_workers > 0 and sys.platform == "linux":
        # marimo/molab default to "spawn", whose workers re-import the notebook and
        # re-run its cells; "fork" (PyTorch's usual Linux default) does not.
        loader_kw["multiprocessing_context"] = "fork"
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"patches: train={len(train_ds)} val={len(val_ds)} | device={device}")

    model = UNetDinoV2(cfg.model.num_classes, cfg.model.backbone,
                       pretrained=cfg.model.pretrained_backbone).to(device)
    weights = torch.tensor(cfg.train.class_weights, dtype=torch.float32, device=device)
    criterion = CombinedLoss(**cfg.train.loss, class_weights=weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", **cfg.train.scheduler)
    amp_dtype = getattr(torch, cfg.train.amp_dtype)
    scaler = _grad_scaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    # Resume from a local or Hub checkpoint.
    start_epoch, best_val, history = 0, float("inf"), []
    last = run_dir / "last.pt"  # same relative path as on the Hub: runs/<run_name>/last.pt
    if not last.exists() and push:
        hub.download_if_exists(f"{hub_prefix}/last.pt", workdir)
    if last.exists():
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_val, history = state["epoch"] + 1, state["best_val"], state["history"]
        print(f"resumed from {last} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion, device, amp_dtype, optimizer, scaler)
        val_loss, val_m = run_epoch(model, val_loader, criterion, device, amp_dtype)
        scheduler.step(val_loss)

        record = {
            "epoch": epoch, "seconds": round(time.time() - t0, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "val_loss": val_loss,
            "train": summary(train_m.compute(), CLASS_NAMES),
            "val": summary(val_m.compute(), CLASS_NAMES),
        }
        history.append(record)
        print(f"epoch {epoch + 1}/{cfg.train.epochs} | {record['seconds']}s | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val mIoU global {record['val']['global/miou']:.4f} / paper {record['val']['paper/miou']:.4f} | "
              f"val mF1 {record['val']['global/mf1']:.4f}")

        improved = val_loss <= best_val  # notebook criterion: lowest validation loss
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "history": history}, run_dir / "last.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))

        if push:
            # Every upload stays in the repo history, so the 3.8 GB resume checkpoint
            # is pushed only every few epochs; best.pt (1.3 GB) whenever it improves.
            hub.upload(run_dir / "history.json", f"{hub_prefix}/history.json")
            if improved:
                hub.upload(run_dir / "best.pt", f"{hub_prefix}/best.pt")
            final = epoch + 1 == cfg.train.epochs
            if not final and (epoch + 1) % cfg.train.get("push_last_every", 5) == 0:
                hub.upload(run_dir / "last.pt", f"{hub_prefix}/last.pt")

    # Weights after the last epoch (best.pt is chosen by validation loss, which can
    # favour an early epoch).
    torch.save(model.state_dict(), run_dir / "final.pt")
    if push:
        hub.upload(run_dir / "final.pt", f"{hub_prefix}/final.pt")

    return run_dir
