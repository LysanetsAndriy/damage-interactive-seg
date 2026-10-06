"""The app's networks on ONNX Runtime (no PyTorch): same inputs and outputs as the
PyTorch versions, so they can replace them one by one.

OnnxSam.predict(points, labels, mask_input, multimask) mirrors
SAM2ImagePredictor.predict: the padding point, "no previous mask", the choice among
the 4 output masks (multimask: tokens 1-3; single: token 0, or the best of 1-3 when
token 0 is unstable) and the bilinear upsampling to the image size.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dmgseg import paths

ONNX_DIR = paths.ARTIFACTS / "onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def session(path, threads=None):
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opt.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def normalize(rgb_u8):
    return ((rgb_u8.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]


def resize_float_bilinear(rgb_u8, size):
    """Exactly SAM 2's own input: torchvision's antialiased bilinear resize of the
    float image == PIL bilinear on float channels (max difference 0.0000)."""
    from PIL import Image
    f = [np.asarray(Image.fromarray(rgb_u8[..., i].astype(np.float32) / 255, mode="F").resize(size, Image.BILINEAR))
         for i in range(3)]
    return ((np.stack(f, -1) - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def resize_pil(rgb_u8, size):
    """As the PyTorch prior path (PIL bilinear on the uint8 image)."""
    from PIL import Image
    return np.asarray(Image.fromarray(rgb_u8).resize(size, Image.BILINEAR))


class OnnxSam:
    def __init__(self, folder=ONNX_DIR):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())["sam"]
        self.enc = session(folder / "sam_encoder.onnx")
        self.dec = session(folder / "sam_decoder.onnx")
        self.features = self.orig_hw = None

    def set_image(self, rgb):
        h, w = rgb.shape[:2]
        embed, hr0, hr1 = self.enc.run(None, {"image": resize_float_bilinear(rgb, (1024, 1024))})
        self.features, self.orig_hw = (embed, hr0, hr1), (h, w)

    def predict(self, points, labels, mask_input=None, multimask=True):
        """points (N,2) image pixels, labels (N,) -> masks (C,H,W) bool, scores (C,),
        low-res logits (C,256,256)."""
        h, w = self.orig_hw
        p = np.asarray(points, np.float32).reshape(-1, 2) * np.array([1024 / w, 1024 / h], np.float32)
        p = np.concatenate([p, np.zeros((1, 2), np.float32)])[None]                  # + padding point
        lab = np.concatenate([np.asarray(labels, np.float32), [-1.0]])[None].astype(np.float32)
        has = np.array([0.0 if mask_input is None else 1.0], np.float32)
        mi = np.zeros((1, 1, 256, 256), np.float32) if mask_input is None else \
            np.asarray(mask_input, np.float32).reshape(1, 1, 256, 256)
        e, h0, h1 = self.features
        masks, iou = self.dec.run(None, {"image_embed": e, "high_res_0": h0, "high_res_1": h1,
                                         "point_coords": p, "point_labels": lab, "mask_input": mi, "has_mask": has})
        masks, iou = masks[0], iou[0]
        if multimask:
            masks, iou = masks[1:], iou[1:]
        elif self.meta["stability"]:
            d = self.meta["stability_delta"]
            single = masks[0]
            area_i, area_u = (single > d).sum(), (single > -d).sum()
            stable = (area_i / area_u if area_u > 0 else 1.0) >= self.meta["stability_thresh"]
            k = 0 if stable else 1 + int(np.argmax(iou[1:]))
            masks, iou = masks[k:k + 1], iou[k:k + 1]
        else:
            masks, iou = masks[:1], iou[:1]
        full = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in masks])
        return full > self.meta["mask_threshold"], iou, np.clip(masks, -32, 32)


class OnnxPrior:
    """Sliding-window prior (as prior/infer.predict_probs) on ONNX Runtime."""

    def __init__(self, folder=ONNX_DIR):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        self.net = session(folder / "prior.onnx")
        self.embed = session(folder / "embed.onnx")

    def embedding(self, rgb):
        s = self.meta["embed"]["size"]
        return self.embed.run(None, {"image": normalize(resize_pil(rgb, (s, s)))})[0]

    def predict_probs(self, rgb, batch=2, progress=None):
        from dmgseg.prior.patches import patch_grid
        ps = self.meta["prior"]["patch_size"]
        h, w = rgb.shape[:2]
        if h < ps or w < ps:                          # as upscale_if_small
            s = max(ps / w, ps / h)
            rgb = resize_pil(rgb, (int(round(w * s)), int(round(h * s))))
            h, w = rgb.shape[:2]
        emb = self.embedding(rgb)
        acc = np.zeros((6, h, w), np.float32)
        cnt = np.zeros((1, h, w), np.float32)
        grid = patch_grid(w, h, ps, self.meta["prior"]["stride"])
        for i in range(0, len(grid), batch):
            corners = grid[i:i + batch]
            x = np.concatenate([normalize(rgb[y:y + ps, x0:x0 + ps]) for x0, y in corners])
            pos = np.array([[x0 / w, y / h] for x0, y in corners], np.float32)
            logits = self.net.run(None, {"x": x, "emb": np.repeat(emb, len(corners), 0), "pos": pos})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs = e / e.sum(1, keepdims=True)
            for (x0, y), p in zip(corners, probs):
                acc[:, y:y + ps, x0:x0 + ps] += p
                cnt[:, y:y + ps, x0:x0 + ps] += 1
            if progress:
                progress(min(i + batch, len(grid)), len(grid))
        return (acc / cnt).transpose(1, 2, 0)


class OnnxHead:
    def __init__(self, folder=ONNX_DIR):
        self.net = session(Path(folder) / "head_a.onnx")

    def score(self, cards):
        logits, q = self.net.run(None, {"cards": np.stack(cards).astype(np.float32)})
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), 1 / (1 + np.exp(-q))
