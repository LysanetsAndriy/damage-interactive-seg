"""Patch dataset and augmentations, ported from the notebook."""
import json
from pathlib import Path

import numpy as np
import torch
from albumentations import (Compose, ElasticTransform, GaussNoise, HorizontalFlip,
                            HueSaturationValue, RandomBrightnessContrast, RandomResizedCrop)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dmgseg.prior.embedding import IMAGENET_MEAN, IMAGENET_STD


def train_transform(size):
    """Augmentations from the paper (Section 'Dataset Preparation and Splitting')."""
    return Compose([
        RandomResizedCrop(size=(size, size), scale=(0.6, 0.8), ratio=(1.0, 1.0), p=1.0),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        GaussNoise(std_range=(0.1, 0.2), p=0.4),
        ElasticTransform(alpha=20, sigma=60, p=0.4),
    ])


def patch_sources(patch_dir):
    """{patch file name: source image name}, cached in _sources.json."""
    cache = Path(patch_dir) / "_sources.json"
    if cache.exists():
        return json.loads(cache.read_text())
    index = {f.name: str(np.load(f)["source"]) for f in sorted(Path(patch_dir).glob("*.npz"))}
    cache.write_text(json.dumps(index))
    return index


class PatchedDataset(Dataset):
    """Reads the .npz patches written by patches.build_patch_dataset.

    Patches are resized to `size` (bilinear image, nearest mask) before augmentation,
    as in the notebook.
    """

    def __init__(self, patch_dir, size=518, transform=None, sources=None, copy_paste=None):
        """copy_paste: a CopyPaste applied to the resized patch before `transform`."""
        self.files = sorted(Path(patch_dir).glob("*.npz"))
        if sources is not None:
            index = patch_sources(patch_dir)
            keep = set(sources)
            self.files = [f for f in self.files if index[f.name] in keep]
        self.size = size
        self.transform = transform
        self.copy_paste = copy_paste
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        image = np.array(Image.fromarray(patch["image"]).resize((self.size, self.size), Image.BILINEAR))
        mask = np.array(Image.fromarray(patch["mask"].astype(np.uint8)).resize((self.size, self.size), Image.NEAREST))

        if self.copy_paste is not None:
            image, mask = self.copy_paste(image, mask)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        return (
            self.normalize(image),
            torch.as_tensor(mask, dtype=torch.long),
            torch.as_tensor(patch["embedding"], dtype=torch.float32),
            torch.as_tensor(patch["position"], dtype=torch.float32),
        )


# ----------------------------------------------------------------- copy-paste
# Context-aware copy-paste for the rare classes: objects are cut from other
# training images and pasted onto pixels of the class they belong on.
PASTE_HOSTS = {4: (1, 3), 3: (1,), 5: (2,)}        # Broken Window / Damage on Building, Damaged roof on Roof
PASTE_WEIGHTS = {4: 0.4, 5: 0.3, 3: 0.3}


def build_object_bank(annotations, label_kinds, patch_size, min_px=64, max_frac=0.2, image_dir=None):
    """{class: [(rgb crop uint8, mask crop bool)]} of connected regions of the
    rare classes in the training images (at the scale the patches are cut at)."""
    import cv2
    from dmgseg import paths
    from dmgseg.data.cvat import semantic_mask
    from dmgseg.prior.patches import upscale_if_small
    bank = {c: [] for c in PASTE_HOSTS}
    for ann in annotations:
        image = Image.open(Path(image_dir or paths.IMAGES_DIR) / ann.name).convert("RGB")
        mask = semantic_mask(ann, kinds=label_kinds)
        image, mask = upscale_if_small(image, mask, patch_size)
        rgb = np.asarray(image)
        for c in PASTE_HOSTS:
            n, comp, stats, _ = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a < min_px or a > max_frac * mask.size or w > patch_size or h > patch_size:
                    continue
                bank[c].append((rgb[y:y + h, x:x + w].copy(), comp[y:y + h, x:x + w] == i))
    return bank


class CopyPaste:
    """Paste 1..max_objects bank objects into a (resized) training patch with
    probability p; each lands centred on a random pixel of a host class, scaled
    by `scale` (patch resize factor) x a random jitter, with feathered edges."""

    def __init__(self, bank, scale, p=0.5, max_objects=3, jitter=(0.7, 1.3), seed=None):
        self.bank = {c: v for c, v in bank.items() if v}
        self.scale, self.p, self.max_objects, self.jitter = scale, p, max_objects, jitter
        self.seed, self.rng, self._pid = seed, np.random.default_rng(seed), None

    def __call__(self, image, mask):
        import os

        import cv2
        if self._pid != os.getpid():         # each DataLoader worker: its own stream
            self._pid = os.getpid()
            self.rng = np.random.default_rng([self.seed or 0, torch.initial_seed() % 2**32])
        if not self.bank or self.rng.random() >= self.p:
            return image, mask
        image, mask = image.copy(), mask.copy()
        classes = [c for c in PASTE_WEIGHTS if c in self.bank]
        w = np.array([PASTE_WEIGHTS[c] for c in classes])
        H, W = mask.shape
        for _ in range(self.rng.integers(1, self.max_objects + 1)):
            c = classes[self.rng.choice(len(classes), p=w / w.sum())]
            hosts = np.flatnonzero(np.isin(mask, PASTE_HOSTS[c]))
            if len(hosts) == 0:
                continue
            rgb, m = self.bank[c][self.rng.integers(len(self.bank[c]))]
            f = self.scale * self.rng.uniform(*self.jitter)
            oh, ow = max(2, int(round(m.shape[0] * f))), max(2, int(round(m.shape[1] * f)))
            if oh >= H or ow >= W:
                continue
            rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if self.rng.random() < 0.5:
                rgb, m = rgb[:, ::-1], m[:, ::-1]
            cy, cx = divmod(int(hosts[self.rng.integers(len(hosts))]), W)
            y0, x0 = min(max(cy - oh // 2, 0), H - oh), min(max(cx - ow // 2, 0), W - ow)
            alpha = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)[..., None]   # soft edges
            region = image[y0:y0 + oh, x0:x0 + ow].astype(np.float32)
            image[y0:y0 + oh, x0:x0 + ow] = (alpha * rgb + (1 - alpha) * region).astype(np.uint8)
            mask[y0:y0 + oh, x0:x0 + ow][m] = c
        return image, mask
