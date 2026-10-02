"""Does inference at the training scale help the prior?

Training patches were 640 px crops resized to 518 (objects at 81% size); the
notebook's evaluation used 518 px crops at native scale. Compare both on the 44
validation images (fixed labels, B2b weights). CPU: ~1 h.

    python scripts/scale_test.py --out artifacts/results/scale_test.json
"""
import argparse
import json
from pathlib import Path

from dmgseg import hub
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.eval.metrics import summary
from dmgseg.prior.compare import FIXED_KINDS
from dmgseg.prior.embedding import load_embedding_model
from dmgseg.prior.infer import evaluate_full_images
from dmgseg.prior.train import split_annotations
from dmgseg.prior.unet_dinov2 import load_prior_model

SETTINGS = {
    "518 crops, native scale (notebook)": dict(patch_size=518, stride=300, model_input=518),
    "640 crops -> 518 (training scale)": dict(patch_size=640, stride=370, model_input=518),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/results/scale_test.json"))
    args = ap.parse_args()
    model = load_prior_model(hub.prior_weights(), "cpu")
    embed = load_embedding_model("cpu")
    _, val = split_annotations()
    results = {}
    for name, kw in SETTINGS.items():
        m = evaluate_full_images(model, embed, val, FIXED_KINDS, device="cpu", **kw)
        results[name] = summary(m.compute(), CLASS_NAMES)
        r = results[name]
        print(f"{name}: global mIoU {r['global/miou']:.4f} | mF1 {r['global/mf1']:.4f} | "
              + " ".join(f"{c} {r['global/iou'][c]:.3f}" for c in CLASS_NAMES), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
