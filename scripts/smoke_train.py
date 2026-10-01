"""CPU smoke test of the full training pipeline (not a real training run).

Random backbone, fake global embedding, 2 images, 1 batch per epoch, 2 epochs
(the second resumes from the first). Then full-image inference with the real
paper weights on one validation image.
"""
import shutil
import tempfile

import numpy as np
import torch

from dmgseg import paths
from dmgseg.config import load_config
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.split import load_split
from dmgseg.prior.infer import evaluate_full_images
from dmgseg.prior.train import train
from dmgseg.prior.unet_dinov2 import load_prior_model


class FakeEmbedder(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1536)


def fake_embed(image):
    return np.zeros(1536, dtype=np.float32)


workdir = tempfile.mkdtemp(prefix="dmgseg_smoke_")
overrides = {"data.limit_images": 2, "model.pretrained_backbone": False,
             "train.batch_size": 1, "train.num_workers": 0, "run_name": "smoke"}
try:
    cfg = load_config("configs/prior_dinov2_6c_paper.yaml", **overrides, **{"train.epochs": 1})
    train(cfg, workdir, fake_embed, device="cpu", push=False, max_batches=1)
    cfg = load_config("configs/prior_dinov2_6c_paper.yaml", **overrides, **{"train.epochs": 2})
    run_dir = train(cfg, workdir, fake_embed, device="cpu", push=False, max_batches=1)
    print("files:", sorted(p.name for p in run_dir.iterdir()))

    model = load_prior_model(paths.PRIOR_WEIGHTS)
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    val = [anns[load_split()["val"][2]]]
    m = evaluate_full_images(model, FakeEmbedder(), val, ("polygon", "box", "mask"),
                             patch_size=518, stride=300)
    r = m.compute()
    print(f"paper weights, 1 val image, FAKE embedding: pixel acc {r['pixel_accuracy']:.3f}")
finally:
    shutil.rmtree(workdir)
