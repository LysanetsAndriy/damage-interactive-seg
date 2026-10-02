"""Class head, version A: an MLP on description cards.

Per candidate mask: 288-d card -> (5 class logits over the clickable classes,
1 quality logit = predicted IoU of the mask with the object the user means).
Among the 3 first-click candidates the one with the highest quality is kept.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmgseg.classhead.features import group_slices
from dmgseg.tool.assign import CLICKABLE

SAM_SCORE_COL = group_slices()["sam_info"].start
PRIOR_MEAN_COLS = [group_slices()["prior_in"].start + c for c in CLICKABLE]

TO_HEAD = {c: i for i, c in enumerate(CLICKABLE)}   # dataset class id -> head index


class CardHead(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.2, n_classes=len(CLICKABLE)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.cls = nn.Linear(hidden // 2, n_classes)
        self.quality = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.body((x - self.mean) / self.std)
        return self.cls(h), self.quality(h).squeeze(-1)


def head_labels(labels):
    return np.array([TO_HEAD[int(c)] for c in labels], np.int64)


def train_head(train, dev, epochs=80, lr=1e-3, weight_decay=1e-4, batch=512, patience=12,
               hidden=256, dropout=0.2, seed=0, log=print):
    """train/dev: card dicts (see build.py). Early stopping on dev score =
    class top-1 on refined masks + mean IoU of the quality-chosen first mask."""
    torch.manual_seed(seed)
    X = torch.from_numpy(train["X"])
    y = torch.from_numpy(head_labels(train["label"]))
    q = torch.from_numpy(train["iou"])
    model = CardHead(X.shape[1], hidden, dropout)
    model.mean.copy_(X.mean(0))
    model.std.copy_(X.std(0).clamp_min(1e-4))
    counts = torch.bincount(y, minlength=len(CLICKABLE)).float()
    class_w = (counts.sum() / counts.clamp_min(1)) ** 0.5
    class_w = class_w / class_w.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            logits, qual = model(X[idx])
            loss = F.cross_entropy(logits, y[idx], weight=class_w) + \
                F.binary_cross_entropy_with_logits(qual, q[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m = evaluate_cards(model, dev)
        score = m["all"]["head/top1@refined"] + m["all"]["head/iou1"]
        log(f"epoch {epoch + 1:3d} | dev top-1 {m['all']['head/top1@refined']:.3f} | "
            f"dev first-click IoU {m['all']['head/iou1']:.3f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model, X):
    logits, qual = model.eval()(torch.from_numpy(X))
    return torch.softmax(logits, -1).numpy(), torch.sigmoid(qual).numpy()


def prior_ranking_probs(X):
    """Baseline: the mean prior inside the mask, clickable classes only."""
    return X[:, PRIOR_MEAN_COLS]


def _rank_of(probs, label_idx):
    order = np.argsort(-probs, axis=1)
    return np.array([int(np.where(order[i] == label_idx[i])[0][0]) for i in range(len(order))])


def evaluate_cards(model, cards, class_names=None):
    """Head A on cards (or only the rules if model is None); see evaluate_predictions."""
    probs, qual = predict(model, cards["X"]) if model is not None else (None, None)
    meta = {k: cards[k] for k in ("obj", "click", "cand", "label", "iou")}
    meta["sam_score"] = cards["X"][:, SAM_SCORE_COL]
    meta["prior_mean"] = prior_ranking_probs(cards["X"])
    return evaluate_predictions(meta, probs, qual)


def evaluate_predictions(meta, probs=None, qual=None):
    """Per object: first-click mask IoU (SAM score / predicted quality / oracle) and
    class rank (prior average vs the model's class probabilities) on the chosen
    first mask and on the last refined mask.

    meta: arrays per candidate -- obj, click, cand, label, iou, sam_score,
    prior_mean (N, 5 clickable classes). probs (N, 5) / qual (N,) from any model."""
    from dmgseg.data.cvat import CLASS_NAMES
    rows = []
    for oid in np.unique(meta["obj"]):
        idx = np.where(meta["obj"] == oid)[0]
        first = idx[meta["click"][idx] == 1][:3]           # the standard first click
        refined = idx[meta["cand"][idx] == 3]
        last = refined[np.argmax(meta["click"][refined])] if len(refined) else first[0]
        label_idx = TO_HEAD[int(meta["label"][idx[0]])]
        sam_pick = first[np.argmax(meta["sam_score"][first])]
        r = {"label": CLASS_NAMES[int(meta["label"][idx[0]])],
             "sam/iou1": float(meta["iou"][sam_pick]),
             "oracle/iou1": float(meta["iou"][first].max()),
             "prior/rank@refined": int(_rank_of(meta["prior_mean"][[last]], [label_idx])[0]),
             "prior/rank@first": int(_rank_of(meta["prior_mean"][[sam_pick]], [label_idx])[0])}
        if probs is not None:
            head_pick = first[np.argmax(qual[first])]
            r["head/iou1"] = float(meta["iou"][head_pick])
            r["head/rank@refined"] = int(_rank_of(probs[[last]], [label_idx])[0])
            r["head/rank@first"] = int(_rank_of(probs[[head_pick]], [label_idx])[0])
        rows.append(r)
    return summarize_rows(rows)


def summarize_rows(rows):
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    out = {}
    for g, rs in groups.items():
        s = {"n": len(rs)}
        for key in rs[0]:
            if key == "label":
                continue
            v = np.array([r[key] for r in rs])
            if key.endswith("iou1"):
                s[key] = float(v.mean())
            else:
                who, where = key.split("/rank@")
                s[f"{who}/top1@{where}"] = float((v == 0).mean())
                s[f"{who}/top2@{where}"] = float((v <= 1).mean())
                s[f"{who}/manual@{where}"] = float((v >= 2).mean())
        out[g] = s
    return out
