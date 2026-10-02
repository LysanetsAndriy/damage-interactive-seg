"""Feature-map crops for class head B (fusion of frozen DINOv2 + SAM + prior).

One crop per click: a square around SAM's candidate masks (and the click) with
25 % context on each side -- computed from SAM's output only, so the app can do
exactly the same. On a GRID x GRID map of that crop:

    dino   DINOv2 ViT-L/14 (original, frozen) tokens of the crop at 224 px,
           PCA-reduced 1024 -> DINO_DIM
    sam    SAM 2 image embedding (256 ch) roi-aligned to the crop
    prior  the 6 prior probabilities, area-resized to the crop grid
    mask   per candidate: the share of each grid cell covered by the mask
"""
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

GRID = 16
DINO_INPUT = 224                 # 16 x 16 tokens of 14 px
DINO_DIM = 256
DINO_MODEL = "vit_large_patch14_reg4_dinov2.lvd142m"
CONTEXT = 0.25
MIN_SIDE = 48
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def crop_box(masks, click, height, width, context=CONTEXT, min_side=MIN_SIDE):
    """Square (x0, y0, x1, y1) around the union of masks and the click, clipped to the image."""
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    cx, cy = click
    xs = np.append(xs, cx) if len(xs) else np.array([cx])
    ys = np.append(ys, cy) if len(ys) else np.array([cy])
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0, min_side) * (1 + 2 * context)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0 = int(round(mx - side / 2)), int(round(my - side / 2))
    bx1, by1 = int(round(mx + side / 2)), int(round(my + side / 2))
    # shift inside the image where possible, then clip
    if bx0 < 0:
        bx1, bx0 = bx1 - bx0, 0
    if by0 < 0:
        by1, by0 = by1 - by0, 0
    if bx1 > width:
        bx0, bx1 = max(0, bx0 - (bx1 - width)), width
    if by1 > height:
        by0, by1 = max(0, by0 - (by1 - height)), height
    return bx0, by0, bx1, by1


def area_resize(arr, size=GRID):
    """(h, w[, c]) float -> (c, size, size), area interpolation (OpenCV's INTER_AREA
    handles at most 4 channels at a time, so channels are resized one by one)."""
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return np.stack([cv2.resize(arr[..., c], (size, size), interpolation=cv2.INTER_AREA)
                     for c in range(arr.shape[-1])])


def mask_map(mask, box):
    x0, y0, x1, y1 = box
    return area_resize(mask[y0:y1, x0:x1])[0]


def prior_map(prior, box):
    x0, y0, x1, y1 = box
    return area_resize(prior[y0:y1, x0:x1])


def sam_map(sam_embed, box, height, width):
    """roi-align SAM's (256, 64, 64) embedding (covering the whole image) to the box."""
    gh, gw = sam_embed.shape[-2:]
    x0, y0, x1, y1 = box
    roi = torch.tensor([[0, x0 * gw / width, y0 * gh / height, x1 * gw / width, y1 * gh / height]],
                       dtype=sam_embed.dtype, device=sam_embed.device)
    return roi_align(sam_embed[None].float(), roi.float(), output_size=GRID, aligned=True)[0]


class DinoCrops:
    """Frozen original DINOv2: crops -> (n, 1024, GRID, GRID) tokens, optionally PCA-reduced."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0,
                                       img_size=DINO_INPUT).to(device).eval()
        self.n_prefix = self.model.num_prefix_tokens   # cls + register tokens
        self.pca_mean = self.pca_basis = None

    @torch.no_grad()
    def tokens(self, crops, batch=64):
        """crops: list of uint8 (h, w, 3) arrays -> float tensor (n, 1024, GRID, GRID) on device."""
        out = []
        for i in range(0, len(crops), batch):
            x = torch.stack([torch.from_numpy(cv2.resize(c, (DINO_INPUT, DINO_INPUT), interpolation=cv2.INTER_AREA))
                             for c in crops[i:i + batch]]).permute(0, 3, 1, 2).float() / 255
            x = ((x - MEAN) / STD).to(self.device)
            kind = torch.device(self.device).type
            with torch.autocast(kind, enabled=kind == "cuda"):
                t = self.model.forward_features(x)[:, self.n_prefix:]
            out.append(t.float().transpose(1, 2).reshape(len(x), -1, GRID, GRID))
        return torch.cat(out)

    def fit_pca(self, crops, dim=DINO_DIM):
        t = self.tokens(crops)                                    # (n, 1024, G, G)
        flat = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])
        self.pca_mean = flat.mean(0)
        _, _, v = torch.pca_lowrank(flat - self.pca_mean, q=dim, center=False)
        self.pca_basis = v[:, :dim]                               # (1024, dim)

    def reduced(self, crops):
        t = self.tokens(crops)
        n, c, g, _ = t.shape
        flat = t.permute(0, 2, 3, 1).reshape(-1, c)
        red = (flat - self.pca_mean) @ self.pca_basis
        return red.reshape(n, g, g, -1).permute(0, 3, 1, 2)
