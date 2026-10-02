"""Fine-tune SAM 2.1's prompt encoder + mask decoder on the damage dataset.

The image encoder stays frozen: its features for every image are computed once and
kept on the GPU. Training follows the interactive protocol (RITM / SAM):
    click 1 (center or random point, sometimes with a jittered box), 3 candidate
    masks -> loss on the best one; then 1-3 correction clicks at the largest error,
    the previous low-res mask fed back as a prompt -> loss on every step.
Losses (as in SAM): 20 x focal + dice on the 256x256 low-res logits, plus MSE between
the predicted and the real IoU. All work happens in SAM's 256x256 square frame.
"""
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from dmgseg import paths
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.eval.run_interactive import MIN_AREA
from dmgseg.eval.simulator import next_click
from dmgseg.sam.predictor import SAM2_MODELS

LOW = 256        # SAM's low-res mask grid
FRAME = 1024     # SAM's input frame (points/boxes are given in this frame)


def trainable_state(model):
    """Only the parts we train (small: ~16 MB)."""
    return {k: v for k, v in model.state_dict().items()
            if k.startswith(("sam_mask_decoder.", "sam_prompt_encoder."))}


def load_finetuned(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    return model


class SamData:
    """Frozen SAM image features (on the GPU) + 256x256 object masks."""

    def __init__(self, predictor, names, device, image_dir=None, min_area=MIN_AREA):
        anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
        self.embed, self.hr0, self.hr1, masks, img_of = [], [], [], [], []
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        for i, name in enumerate(tqdm(names, desc="SAM features")):
            image = np.asarray(Image.open((image_dir or paths.IMAGES_DIR) / name).convert("RGB"))
            with torch.no_grad():
                predictor.set_image(image)
            f = predictor._features
            self.embed.append(f["image_embed"][0].to(device, dt))
            self.hr0.append(f["high_res_feats"][0][0].to(device, dt))
            self.hr1.append(f["high_res_feats"][1][0].to(device, dt))
            for o in image_objects(anns[name], min_area):           # all classes, incl. Other
                m = cv2.resize(o.mask.astype(np.float32), (LOW, LOW), interpolation=cv2.INTER_AREA) > 0.5
                if m.sum() >= 4:
                    masks.append(m)
                    img_of.append(i)
        self.masks = torch.from_numpy(np.stack(masks)).to(device)   # (N, 256, 256) bool
        self.img_of = torch.tensor(img_of, device=device)
        self.device = device

    def __len__(self):
        return len(self.masks)

    def features(self, idx):
        im = self.img_of[idx].tolist()
        return (torch.stack([self.embed[i] for i in im]),
                [torch.stack([self.hr0[i] for i in im]), torch.stack([self.hr1[i] for i in im])])


def _deepest(mask):
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return x, y, dt


def first_click(mask, rng, random_click):
    x, y, dt = _deepest(mask)
    if random_click:
        ys, xs = np.nonzero(dt >= max(0.3 * dt.max(), 1))
        i = rng.integers(len(ys))
        x, y = xs[i], ys[i]
    return float(x), float(y)


def jittered_box(mask, rng, jitter=0.1):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    d = rng.uniform(-jitter, jitter, 4) * np.array([w, h, w, h])
    return np.clip(np.array([x0, y0, x1, y1]) + d, 0, LOW)


def decode(model, embed, hr, coords, labels, mask_input, multimask):
    """coords (B, P, 2) in the 1024 frame, labels (B, P) with -1 = padding,
    box corners as labels 2/3. -> low-res logits (B, C, 256, 256), IoU pred (B, C)."""
    sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=mask_input)
    logits, iou_pred, _, _ = model.sam_mask_decoder(
        image_embeddings=embed, image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=multimask, repeat_image=False, high_res_features=hr)
    return logits, iou_pred


def mask_losses(logits, gt):
    """Per-sample focal (x20) + dice, logits/gt (B, C, H, W) -> (B, C)."""
    gt = gt.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pt = p * gt + (1 - p) * (1 - gt)
    alpha = 0.25 * gt + 0.75 * (1 - gt)
    focal = (alpha * (1 - pt) ** 2 * ce).mean((-2, -1))
    inter = (p * gt).sum((-2, -1))
    dice = 1 - (2 * inter + 1) / (p.sum((-2, -1)) + gt.sum((-2, -1)) + 1)
    return 20 * focal + dice


def batch_iou(pred, gt):
    inter = (pred & gt).sum((-2, -1)).float()
    union = (pred | gt).sum((-2, -1)).float()
    return inter / union.clamp_min(1)


def interactive_step(model, data, idx, rng, max_corrections=3, box_prob=0.3, random_prob=0.5, train=True):
    """One simulated interaction on objects idx. Returns (loss, IoU after each click)."""
    gt = data.masks[idx]
    embed, hr = data.features(idx)
    B = len(idx)
    gts = gt.cpu().numpy()
    pts, lbls = [], []
    for b in range(B):
        x, y = first_click(gts[b], rng, train and rng.random() < random_prob)
        p, l = [[x * 4 + 2, y * 4 + 2]], [1]
        if train and rng.random() < box_prob:
            bx = jittered_box(gts[b], rng) * 4
            p, l = [[bx[0], bx[1]], [bx[2], bx[3]]] + p, [2, 3] + l
        pts.append(p)
        lbls.append(l)

    def tensors(pts, lbls):
        n = max(len(p) for p in pts)
        c = torch.zeros(B, n, 2, device=gt.device)
        lab = torch.full((B, n), -1, dtype=torch.int, device=gt.device)
        for b, (p, l) in enumerate(zip(pts, lbls)):
            c[b, :len(p)] = torch.tensor(p, device=gt.device)
            lab[b, :len(l)] = torch.tensor(l, device=gt.device)
        return c, lab

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gt.is_cuda):
        coords, labels = tensors(pts, lbls)
        logits, iou_pred = decode(model, embed, hr, coords, labels, None, True)
    logits, iou_pred = logits.float(), iou_pred.float()
    gt3 = gt[:, None].expand_as(logits)
    l_mask = mask_losses(logits, gt3)                        # (B, 3)
    real_iou = batch_iou(logits > 0, gt3)
    best = l_mask.argmin(1)
    loss = l_mask.gather(1, best[:, None]).mean() + F.mse_loss(iou_pred, real_iou)
    # continue from the mask SAM itself would pick (realistic), as the app does
    pick = iou_pred.argmax(1)
    cur = logits[torch.arange(B), pick][:, None]
    ious = [batch_iou(cur[:, 0] > 0, gt).mean().item()]

    n_corr = rng.integers(1, max_corrections + 1) if train else max_corrections
    for _ in range(n_corr):
        pred = (cur[:, 0] > 0).cpu().numpy()
        for b in range(B):
            x, y, pos = next_click(pred[b], gts[b])
            pts[b].append([x * 4 + 2, y * 4 + 2])
            lbls[b].append(int(pos))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gt.is_cuda):
            coords, labels = tensors(pts, lbls)
            logits, iou_pred = decode(model, embed, hr, coords, labels, cur.detach().clamp(-32, 32), False)
        cur = logits.float()
        l_step = mask_losses(cur, gt[:, None])[:, 0]
        loss = loss + l_step.mean() + F.mse_loss(iou_pred.float()[:, 0], batch_iou(cur[:, 0] > 0, gt))
        ious.append(batch_iou(cur[:, 0] > 0, gt).mean().item())
    return loss / (1 + n_corr), ious


@torch.no_grad()
def dev_score(model, data, batch=64, seed=0):
    """Mean IoU after 1 and 3 clicks (center first click, no boxes)."""
    rng = np.random.default_rng(seed)
    model.eval()
    totals, n = np.zeros(3), 0
    for i in range(0, len(data), batch):
        idx = torch.arange(i, min(i + batch, len(data)), device=data.device)
        _, ious = interactive_step(model, data, idx, rng, max_corrections=2, train=False)
        totals += np.array(ious) * len(idx)
        n += len(idx)
    return totals / n


def finetune(predictor, train_data, dev_data, epochs=12, lr=3e-5, weight_decay=0.01, batch=32,
             patience=3, seed=0, log=print):
    model = predictor.model
    for p in model.parameters():
        p.requires_grad = False
    params = [p for n, p in model.named_parameters()
              if n.startswith(("sam_mask_decoder.", "sam_prompt_encoder."))]
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(train_data) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    base = dev_score(model, dev_data)
    log(f"zero-shot dev IoU after 1/2/3 clicks: {np.round(base, 4).tolist()}")
    best, best_state, bad, history = base[0] + base[-1], {k: v.clone() for k, v in trainable_state(model).items()}, 0, []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train_data), device=train_data.device)
        for i in range(0, len(train_data), batch):
            loss, _ = interactive_step(model, train_data, perm[i:i + batch], rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        d = dev_score(model, dev_data)
        history.append({"epoch": epoch, "dev_iou": d.tolist(), "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch + 1}/{epochs}: dev IoU after 1/2/3 clicks {np.round(d, 4).tolist()} ({time.time() - t0:.0f}s)")
        score = d[0] + d[-1]
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in trainable_state(model).items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state, strict=False)
    model.eval()
    return {"zero_shot_dev": base.tolist(), "history": history}


def new_predictor(size="small", device="cpu"):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor.from_pretrained(SAM2_MODELS[size], device=device)
