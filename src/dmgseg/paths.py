"""Filesystem locations shared by local runs, molab and Colab.

Locally everything sits in the project root. In the cloud, set DMGSEG_DATA to the
folder returned by `huggingface_hub.snapshot_download` and DMGSEG_ARTIFACTS to a
writable directory.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("DMGSEG_DATA", PROJECT_ROOT))
ARTIFACTS = Path(os.environ.get("DMGSEG_ARTIFACTS", PROJECT_ROOT / "artifacts"))


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# Local layout: Dataset_CVAT/...; the Hugging Face repo keeps the same name
CVAT_DIR = _first_existing(DATA_ROOT / "Dataset_CVAT", DATA_ROOT / "dataset")
ANNOTATIONS_XML = CVAT_DIR / "annotations.xml"
IMAGES_DIR = CVAT_DIR / "images" / "default"

# The paper's published 6-class DINOv2-Emb model (local file name / Hugging Face name).
# Kept as the reference in all comparisons.
PAPER_WEIGHTS = _first_existing(
    DATA_ROOT / "best_model_dinov2_6_classes_6e.pth",
    DATA_ROOT / "weights" / "paper_dinov2_emb_6class.pth",
)
PRIOR_WEIGHTS = PAPER_WEIGHTS  # old name, still used by already-created notebook cells

# The prior used by the tool: B2b (batch 16, cosine, layer-wise LR decay), final
# epoch; chosen 2026-10-01 (global mIoU 0.610 vs 0.607 for the paper model, with
# no validation-set-based checkpoint selection). Fetched with hub.prior_weights().
PRIOR_RUN = "dinov2_emb_6c_b2b"

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "paper_split.json"

# Private Hugging Face dataset repo with the images, annotations and all weights.
HF_REPO = os.environ.get("DMGSEG_HF_REPO", "cTaJloHe-Ka4yP/damage-seg-data")
