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

    def __init__(self, patch_dir, size=518, transform=None, sources=None):
        self.files = sorted(Path(patch_dir).glob("*.npz"))
        if sources is not None:
            index = patch_sources(patch_dir)
            keep = set(sources)
            self.files = [f for f in self.files if index[f.name] in keep]
        self.size = size
        self.transform = transform
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

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        return (
            self.normalize(image),
            torch.as_tensor(mask, dtype=torch.long),
            torch.as_tensor(patch["embedding"], dtype=torch.float32),
            torch.as_tensor(patch["position"], dtype=torch.float32),
        )
