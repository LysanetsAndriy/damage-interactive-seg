"""Full-image sliding-window inference, ported from the notebook's predict_full_image.

Returns class PROBABILITIES (averaged over overlapping patches), not just the
argmax, because the interactive tool needs them.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dmgseg import paths
from dmgseg.data.cvat import semantic_mask
from dmgseg.eval.metrics import SegMetrics
from dmgseg.prior.embedding import IMAGENET_MEAN, IMAGENET_STD, image_embedding
from dmgseg.prior.patches import patch_grid, upscale_if_small

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def predict_probs(model, embed_model, image, patch_size=518, stride=300, model_input=518,
                  device="cpu", batch_size=8, progress=None):
    """Returns (probs [H, W, C] float32, image actually used).

    The image is upscaled only if it is smaller than patch_size (as in the notebook).
    progress(done, total): optional callback after each batch of patches.
    If patch_size != model_input, patches are resized to model_input and the
    predictions resized back.
    """
    image, _ = upscale_if_small(image, None, patch_size)
    width, height = image.size
    embedding = torch.as_tensor(image_embedding(embed_model, image, device), device=device)
    num_classes = model.conv_last.out_channels

    probs_sum = torch.zeros(num_classes, height, width, device=device)
    counts = torch.zeros(1, height, width, device=device)
    grid = patch_grid(width, height, patch_size, stride)

    for start in range(0, len(grid), batch_size):
        corners = grid[start:start + batch_size]
        batch = []
        for x, y in corners:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            if patch_size != model_input:
                patch = patch.resize((model_input, model_input), Image.BILINEAR)
            batch.append(_to_tensor(patch))
        x_in = torch.stack(batch).to(device)
        emb = embedding.unsqueeze(0).expand(len(corners), -1)
        pos = torch.tensor([[x / width, y / height] for x, y in corners], dtype=torch.float32, device=device)

        probs = torch.softmax(model(x_in, emb, pos).float(), dim=1)
        if patch_size != model_input:
            probs = F.interpolate(probs, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
        for (x, y), p in zip(corners, probs):
            probs_sum[:, y:y + patch_size, x:x + patch_size] += p
            counts[:, y:y + patch_size, x:x + patch_size] += 1
        if progress:
            progress(min(start + batch_size, len(grid)), len(grid))

    probs = (probs_sum / counts).permute(1, 2, 0).cpu().numpy()
    return probs, image


def evaluate_full_images(model, embed_model, images, label_kinds, patch_size=518, stride=300,
                         model_input=518, device="cpu", image_dir=None, on_image=None):
    """Notebook-style evaluation on whole images. Returns a SegMetrics.

    on_image(ann, probs, pred, true) is called per image, e.g. to cache priors.
    """
    metrics = SegMetrics(model.conv_last.out_channels)
    model.eval()
    for ann in tqdm(images, desc="evaluating"):
        image = Image.open(Path(image_dir or paths.IMAGES_DIR) / ann.name).convert("RGB")
        probs, used = predict_probs(model, embed_model, image, patch_size, stride, model_input, device)
        pred = probs.argmax(-1).astype(np.uint8)
        true = semantic_mask(ann, kinds=label_kinds)
        if true.shape != pred.shape:
            true = np.array(Image.fromarray(true).resize(used.size, Image.NEAREST))
        metrics.update(pred, true)
        if on_image:
            on_image(ann, probs, pred, true)
    return metrics
