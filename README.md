# Interactive damage segmentation

Click / box / lasso / scribble segmentation of war-damaged buildings, with the class
chosen automatically from a semantic prior. Continues the paper
*Embedding-Enhanced U-Net with a Patch-Based Approach and a Novel Dataset for
Ground-Level Building Damage Segmentation* (Applied AI, 2026).

See [PLAN.md](PLAN.md) for the design, experiments and timeline.

## Setup (Mac, Intel)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[mac,app,dev]"
.venv/bin/hf auth login
```

SAM 2 (needs torch >= 2.5 officially; it works with the Mac's torch 2.2 when installed
from source without its version pin):

```bash
git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party/sam2
uv pip install --python .venv/bin/python setuptools wheel hydra-core iopath
cd third_party/sam2 && SAM2_BUILD_CUDA=0 uv pip install --python ../../.venv/bin/python --no-deps --no-build-isolation -e .
```

Data and weights live in the private Hugging Face repo set in `src/dmgseg/paths.py`
(`HF_REPO`). Locally they are read from `Dataset_CVAT/` in the project root.

## Layout

- `src/dmgseg/data` — CVAT parser (polygon, box, brush mask; polylines ignored), paper split
- `src/dmgseg/prior` — DINOv2-Emb U-Net (the paper model): patches, dataset, losses,
  training with resume + Hugging Face checkpoints, full-image inference returning probabilities
- `src/dmgseg/eval/metrics.py` — global (standard) and paper-style metrics side by side
- `src/dmgseg/sam/predictor.py` — SAM 2.1 click wrapper (multimask first click, logits fed back)
- `src/dmgseg/eval/simulator.py`, `run_interactive.py` — simulated user, NoC@85/90, IoU@k
- `configs/` — one YAML per run
- `scripts/` — one-off utilities (Hugging Face upload, CPU smoke test of training)
- `tests/` — `pytest`: parser, split, Table 2/3 reproduction, metrics, loss
- `nb/` — thin marimo notebooks for molab
