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

Data and weights live in the private Hugging Face repo set in `src/dmgseg/paths.py`
(`HF_REPO`). Locally they are read from `Dataset_CVAT/` in the project root.

## Layout

- `src/dmgseg/data` — CVAT parser (polygon, box, brush mask; polylines ignored), paper split
- `src/dmgseg/prior` — DINOv2-Emb U-Net (the paper model)
- `scripts/` — one-off utilities (Hugging Face upload)
- `nb/` — thin marimo notebooks for molab
