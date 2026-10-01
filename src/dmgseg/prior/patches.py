"""Patch extraction, ported from the notebook's save_patched_dataset_with_embedding.

The same grid is used for training patches and for full-image inference.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dmgseg import paths
from dmgseg.data.cvat import semantic_mask


def upscale_if_small(image, mask, patch_size):
    """Resize so both sides are >= patch_size (bilinear image, nearest mask)."""
    w, h = image.size
    if w >= patch_size and h >= patch_size:
        return image, mask
    scale = max(patch_size / w, patch_size / h)
    size = (int(round(w * scale)), int(round(h * scale)))
    image = image.resize(size, Image.BILINEAR)
    if mask is not None:
        mask = np.array(Image.fromarray(mask).resize(size, Image.NEAREST))
    return image, mask


def _axis_positions(length, patch, stride):
    if length == patch:
        return [0]
    n = max(2, int(round((length - patch) / stride)) + 1)
    step = (length - patch) / (n - 1)
    return [min(int(round(i * step)), length - patch) for i in range(n)]


def patch_grid(width, height, patch_size, stride):
    """Top-left corners (x, y) of the dynamic-stride grid, row by row."""
    xs = _axis_positions(width, patch_size, stride)
    ys = _axis_positions(height, patch_size, stride)
    return [(x, y) for y in ys for x in xs]


def build_patch_dataset(images, out_dir, patch_size, stride, embed_fn, label_kinds,
                        image_dir=None):
    """Write one .npz per patch: image (uint8 HxWx3), mask (uint8 HxW),
    embedding (float32 [1536]) and position (float32 [x/W, y/H]).

    embed_fn: PIL image -> numpy embedding (see embedding.image_embedding).
    Returns the number of patches written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    for ann in tqdm(images, desc=f"patches -> {out_dir.name}"):
        image = Image.open(Path(image_dir or paths.IMAGES_DIR) / ann.name).convert("RGB")
        mask = semantic_mask(ann, kinds=label_kinds)
        image, mask = upscale_if_small(image, mask, patch_size)
        width, height = image.size
        embedding = np.asarray(embed_fn(image), dtype=np.float32)
        image_np = np.asarray(image)

        for x, y in patch_grid(width, height, patch_size, stride):
            np.savez_compressed(
                out_dir / f"patch_{index}.npz",
                image=image_np[y:y + patch_size, x:x + patch_size],
                mask=mask[y:y + patch_size, x:x + patch_size],
                embedding=embedding,
                position=np.array([x / width, y / height], dtype=np.float32),
                source=ann.name,
            )
            index += 1
    return index
