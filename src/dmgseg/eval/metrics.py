"""Segmentation metrics.

Two families are reported side by side:

- global: every metric (IoU, Dice, Precision, Recall, F1) from ONE confusion
  matrix summed over the whole dataset. This is the standard definition and the
  one used from now on.
- paper: what the paper's notebook reported -- IoU and Dice as nanmean over
  per-image (or per-patch) values, F1/Precision/Recall from the global matrix.
"""
import warnings

import numpy as np
import torch


class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
        self.per_item_iou = []
        self.per_item_dice = []

    def update(self, pred, true):
        """pred, true: integer label maps (numpy or torch), shape (H, W) or (B, H, W)."""
        if torch.is_tensor(pred):
            pred, true = pred.detach(), true.detach()
            if pred.ndim == 2:
                pred, true = pred[None], true[None]
            k = self.num_classes
            for p, t in zip(pred, true):
                cm = torch.bincount(t.reshape(-1).long() * k + p.reshape(-1).long(),
                                    minlength=k * k).reshape(k, k).cpu().numpy()
                self._add(cm)
            return
        pred, true = np.asarray(pred), np.asarray(true)
        if pred.ndim == 2:
            pred, true = pred[None], true[None]
        k = self.num_classes
        for p, t in zip(pred, true):
            cm = np.bincount(t.ravel().astype(np.int64) * k + p.ravel(), minlength=k * k).reshape(k, k)
            self._add(cm)

    def _add(self, cm):
        self.confusion += cm
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            self.per_item_iou.append(np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan))
            self.per_item_dice.append(np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan))

    def compute(self):
        cm = self.confusion.astype(float)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            iou = tp / (tp + fp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)
        with warnings.catch_warnings():  # a class absent from every item -> NaN, which is fine
            warnings.simplefilter("ignore", RuntimeWarning)
            paper_iou = np.nanmean(np.stack(self.per_item_iou), axis=0) if self.per_item_iou else iou * np.nan
            paper_dice = np.nanmean(np.stack(self.per_item_dice), axis=0) if self.per_item_dice else iou * np.nan
        return {
            "pixel_accuracy": float(tp.sum() / cm.sum()),
            "global": {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1},
            "paper": {"iou": paper_iou, "dice": paper_dice, "f1": f1},
            "confusion": self.confusion.copy(),
        }


def summary(result, class_names):
    """Compact JSON-friendly dict: per-class and mean values for both families."""
    out = {"pixel_accuracy": result["pixel_accuracy"]}
    for family in ("global", "paper"):
        for name, values in result[family].items():
            out[f"{family}/{name}"] = {c: float(v) for c, v in zip(class_names, values)}
            out[f"{family}/m{name}"] = float(np.nanmean(values))
    out["confusion"] = result["confusion"].tolist()
    return out
