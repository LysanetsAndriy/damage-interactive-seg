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

## The app (Damage Annotator)

```bash
.venv/bin/python -m dmgseg.app                 # File > Open Folder / Open Image
.venv/bin/python -m dmgseg.app path/to/folder  # open a folder of images directly
```

Folder mode: progress is saved automatically in `<folder>/_damage_annotator/`;
the DINOv2 prior is cached there and pre-computed for the next images in the
background; Page Up/Down switch images; File > Export CVAT XML exports the folder
(masks: exact; polygons: outlines) for CVAT or the training parser.

View (same on all systems; Cmd/Option on macOS = Ctrl/Alt on Windows and Linux):
wheel or two-finger swipe scrolls (Shift+wheel: sideways), **Ctrl/Cmd+wheel** or a
pinch zooms at the cursor, Ctrl/Cmd +/-/0 zoom by keyboard, F fits, Space+drag or the
middle button pans. Previous/next image: Page Up/Down or Ctrl/Cmd+Left/Right.
Shrink: Alt/Option+click or Shift+right click (for Linux desktops that use Alt+drag).
Help > Shortcuts lists everything with the right key names for the system.

Drawing (left-drag on the image; the stroke is yellow / cyan / green / red while drawing):
- **loop** around a thing: that object. SAM proposes shapes (loop box, points inside,
  "not this" points outside) and the one that matches the loop best wins.
- **Cmd + loop**: every object the prior finds inside (a group of windows).
- **line** over one thing: one object through points along the line.
- **Shift / Option + loop**: add / cut exactly that area (pixel-precise edits).
- **Shift / Option + line**: grow / shrink the active object with SAM along the line.
- On pre-label objects (and after exact edits) Shift / Option clicks are local: only
  the piece under the cursor is added / removed. Option+click acts on the object under
  the cursor.
- Objects list: multi-select, right click = Delete / Set class, Backspace deletes the
  selection, "Select tiny" selects objects < 50 px.

A Windows-98-style desktop tool. Left click: new object (SAM mask, class and mask
chosen by class head A); right click / Ctrl+click: next class; Shift+click: grow;
Option+click: shrink; 1-5: set class; Delete; Ctrl+Z. File > Export Mask writes a
PNG of class ids and a JSON with object polygons. The DINOv2 prior runs in the
background (~40 s per image on the Intel Mac CPU); objects clicked before it is
ready get their class when it arrives.

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
- `src/dmgseg/classhead` — class head A (MLP on cards) and B (fusion on feature maps)
- `src/dmgseg/app` — the desktop app: `engine.py` (logic, GUI-free) and `gui.py` (Qt)
- `nb/` — thin marimo notebooks for molab
