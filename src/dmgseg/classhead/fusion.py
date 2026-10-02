"""Class head B: a small fusion network on feature-map crops.

Input per candidate: the crop's maps (DINOv2 PCA 256, SAM 256, prior 6), the
candidate mask (1), all on a 16 x 16 grid, plus 14 scalars.
    1x1 convs: DINOv2 -> 96, SAM -> 64, prior+mask -> 32  => 192 channels
    3 residual 3x3 conv blocks (GroupNorm, GELU)
    pooling: mean inside the mask, mean over the crop, max inside the mask
    + scalars -> MLP -> 5 class logits + 1 quality logit
Ablations drop a source ("no_dino", "no_sam") by zeroing its input.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmgseg.classhead.mlp import TO_HEAD, evaluate_predictions
from dmgseg.tool.assign import CLICKABLE


def _proj(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.GroupNorm(8, c_out), nn.GELU())


class Block(nn.Module):
    def __init__(self, c, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                 nn.Dropout2d(dropout), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class FusionHead(nn.Module):
    def __init__(self, c_dino=256, c_sam=256, c_prior=6, n_scalars=14, width=192, blocks=3,
                 dropout=0.2, n_classes=len(CLICKABLE)):
        super().__init__()
        self.dino = _proj(c_dino, 96)
        self.sam = _proj(c_sam, 64)
        self.local = _proj(c_prior + 1, 32)
        self.blocks = nn.Sequential(*[Block(width, dropout) for _ in range(blocks)])
        self.scal = nn.Sequential(nn.Linear(n_scalars, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * width + 32, 256), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(256, n_classes)
        self.quality = nn.Linear(256, 1)
        self.register_buffer("s_mean", torch.zeros(n_scalars))
        self.register_buffer("s_std", torch.ones(n_scalars))

    def forward(self, dino, sam, prior, mask, scalars):
        x = torch.cat([self.dino(dino), self.sam(sam), self.local(torch.cat([prior, mask], 1))], 1)
        x = self.blocks(x)
        w = mask / mask.sum((2, 3), keepdim=True).clamp_min(1e-3)
        inside = (x * w).sum((2, 3))
        crop = x.mean((2, 3))
        peak = (x + (mask - 1) * 1e4).amax((2, 3)).clamp_min(-1e3)
        h = self.head(torch.cat([inside, crop, peak, self.scal((scalars - self.s_mean) / self.s_std)], 1))
        return self.cls(h), self.quality(h).squeeze(-1)


class GpuData:
    """The whole dataset as tensors on one device; samples refer to crops by index."""

    def __init__(self, d, device, drop=()):
        f16 = torch.float16
        self.dino = torch.from_numpy(d["dino"]).to(device, f16)
        self.sam = torch.from_numpy(d["sam"]).to(device, f16)
        if "dino" in drop:
            self.dino.zero_()
        if "sam" in drop:
            self.sam.zero_()
        self.prior = torch.from_numpy(d["prior"]).to(device, f16)
        self.crop = torch.from_numpy(d["crop"]).to(device)
        self.mask = torch.from_numpy(d["mask"]).to(device, f16)[:, None]
        self.scalars = torch.from_numpy(d["scalars"]).to(device)
        self.y = torch.tensor([TO_HEAD[int(c)] for c in d["label"]], device=device)
        self.iou = torch.from_numpy(d["iou"].astype(np.float32)).to(device)
        self.meta = {"obj": d["obj"], "click": d["click"], "cand": d["cand"], "label": d["label"],
                     "iou": d["iou"], "sam_score": d["sam_score"], "prior_mean": d["prior_mean"]}

    def __len__(self):
        return len(self.y)

    def batch(self, idx, flip=False):
        c = self.crop[idx]
        parts = [self.dino[c].float(), self.sam[c].float(), self.prior[c].float(), self.mask[idx].float()]
        if flip:  # horizontal flip of all maps (approximate for DINOv2/SAM features)
            parts = [p.flip(-1) for p in parts]
        s = self.scalars[idx].clone()
        if flip:
            s[:, 5] = 1 - s[:, 5]   # center x
        return (*parts, s)


def subset(d, images):
    """Rows of a load_shards dict whose image is in `images` (crops/objects renumbered)."""
    rows = np.isin(d["image"], list(images))
    crops = np.unique(d["crop"][rows])
    remap = -np.ones(len(d["dino"]), np.int64)
    remap[crops] = np.arange(len(crops))
    out = {k: d[k][rows] for k in d if k not in ("dino", "sam", "prior")}
    out["crop"] = remap[out["crop"]]
    _, out["obj"] = np.unique(out["obj"], return_inverse=True)
    for k in ("dino", "sam", "prior"):
        out[k] = d[k][crops]
    return out


@torch.no_grad()
def predict(model, data, batch=2048):
    model.eval()
    probs, qual = [], []
    for i in range(0, len(data), batch):
        idx = torch.arange(i, min(i + batch, len(data)), device=data.y.device)
        logits, q = model(*data.batch(idx))
        probs.append(torch.softmax(logits, -1).float().cpu())
        qual.append(torch.sigmoid(q).float().cpu())
    return torch.cat(probs).numpy(), torch.cat(qual).numpy()


def evaluate(model, data):
    probs, qual = predict(model, data)
    return evaluate_predictions(data.meta, probs, qual)


def train_fusion(train, dev, epochs=40, lr=1e-3, weight_decay=0.05, batch=512, patience=8, seed=0,
                 log=print):
    """train/dev: GpuData. Early stopping on dev (class top-1 on refined masks +
    first-click IoU of the quality-chosen mask)."""
    torch.manual_seed(seed)
    device = train.y.device
    model = FusionHead(c_dino=train.dino.shape[1], c_sam=train.sam.shape[1]).to(device)
    model.s_mean.copy_(train.scalars.mean(0))
    model.s_std.copy_(train.scalars.std(0).clamp_min(1e-4))
    counts = torch.bincount(train.y, minlength=len(CLICKABLE)).float()
    class_w = (counts.sum() / counts.clamp_min(1)) ** 0.5
    class_w = class_w / class_w.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda"
    best, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train), device=device)
        for i in range(0, len(train), batch):
            idx = perm[i:i + batch]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, q = model(*train.batch(idx, flip=bool(torch.rand(1) < 0.5)))
                loss = F.cross_entropy(logits.float(), train.y[idx], weight=class_w) + \
                    F.binary_cross_entropy_with_logits(q.float(), train.iou[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        m = evaluate(model, dev)["all"]
        score = m["head/top1@refined"] + m["head/iou1"]
        log(f"epoch {epoch + 1:3d} | dev top-1 {m['head/top1@refined']:.3f} | dev first-click IoU {m['head/iou1']:.3f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best
