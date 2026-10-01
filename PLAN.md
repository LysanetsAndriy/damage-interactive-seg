# Interactive Damage Segmentation — Project Plan

**Lab task (CV, KNU):** build a model that returns a segmentation mask for the object the user clicks, and an app to test it on any image the user picks.
**Extension:** masks come with one of the 6 classes chosen automatically, so the tool works as a labeling tool that grows the paper's dataset.
**Deadline:** 22 November 2026 · **Training:** molab (RTX Pro 6000, 96 GB) · **Inference:** the MacBook

---

## 0. Findings from the existing materials (read first)

Checked on 2026-09-29 against `Dataset_CVAT/`, the notebook and the Mac.

### 0.1 Brush masks were never used in training (polylines correctly were not)

`annotations.xml` has four kinds of shapes. The notebook's `parse_annotations()` reads only `polygon` and `box`:

| Shape | Count | Read by the notebook? |
|---|---|---|
| polygon | 10,862 | yes |
| box | 732 | yes (turned into a polygon) |
| **mask** (CVAT brush, RLE) | **264**, of which **190 are Damage** | **no, dropped** |
| **polyline** | **187** (171 Roof, 15 Damaged roof) | **no, dropped** |

**Checked on 2026-09-30 (`src/dmgseg/data/cvat.py`):**

- **The brush masks are real labels and are now used.** All 264 decode cleanly: the RLE length equals the bbox area, and every mask touches all four bbox edges. Adding them changes **0.26% of pixels** in 77 images (>1% in 23 images, at most 6.2%). By class: Damage +2.4%, Damaged roof +1.4%, Broken Window +0.6%, Building −1.2%, Roof −1.6%.
- **The polylines are not region labels.** 176 of the 187 are open. A visual check showed they trace thin **roof edges** (flat roofs seen from the ground). Closing them would paint sky or wall as Roof. **They are ignored**, as in the paper. An earlier estimate of "+90% Roof" came from wrongly closing them and is retracted.
- **Paper mode** (`kinds=PAPER_KINDS`) reproduces Table 2 exactly (66.10 / 19.42 / 1.24 / 8.66 / 2.66 / 1.92).

**Conclusion:** the dataset fix is small and mostly adds Damage. The new 6-class model is trained on the full labels (brush masks included, polylines ignored). The weak Roof results are **not** explained by dropped annotations.

### 0.2 Why the paper's IoU and F1 don't match

Both evaluation functions compute:

- **IoU** as `nanmean` of IoU per image or per patch;
- **F1 / Precision / Recall** from a **global** confusion matrix summed over the whole dataset.

These are two different ways of averaging, which is why a mean F1 of 0.746 sits next to a mean IoU of 0.471. **From now on, every metric comes from one global confusion matrix** (`src/dmgseg/eval/metrics.py`). We also recompute the paper model's numbers this way so all comparisons are fair.

### 0.3 The Mac is an Intel machine, which limits PyTorch

- The laptop has an Intel i7-9750H, 16 GB RAM, and an AMD Radeon Pro 5300M (4 GB).
- **PyTorch stopped publishing Intel-macOS builds after version 2.2.x.** Current SAM 2 code needs torch ≥ 2.5.
- Apple's GPU backend (MPS) on AMD GPUs is slow and unreliable, and 4 GB is too little memory for ViT-L.

**Decision:** the app runs its models through **ONNX Runtime on the CPU**. The models are exported to ONNX on molab. The Mac never needs a new PyTorch; torch 2.2.2 is kept only for debugging.

**Speed on the Mac:**

| What runs | When | Rough time |
|---|---|---|
| SAM image encoder | Once per image | ~1–4 s |
| SAM decoder + class head | Per click | < 100 ms |
| DINOv2-Emb prior (304M ViT-L, several 518 px patches) | Once per image | ~20–60 s on the CPU |

The app shows a progress bar during the prior and caches results. For the 290 dataset images, priors are precomputed on molab.

### 0.4 The train/validation split can be reproduced exactly

The paper's split comes from `train_test_split(annotations, test_size=0.15, random_state=1212)` over the images in XML order, which gives 246 / 44. We reproduce it and save it as `data/splits/paper_split.json`. **The 44 validation images are never used for training anything in this project.**

### 0.5 Weights

The first file provided (`best_model2_dinov2_640to518_simple_unet.pth`) turned out to be the **3-class** DINOv2-Emb model. The 6-class checkpoint `best_model_dinov2_6_classes_6e.pth` was then recovered from Google Drive (2026-10-01). It loads into `UNetDinoV2(num_classes=6)` with `strict=True` (315.5M parameters). It is on Hugging Face as `weights/paper_dinov2_emb_6class.pth`, and the 3-class file was removed (still in the repo history). **To check:** that this is the published model, by recomputing the paper's validation metrics with the paper's evaluation code (expected mean F1 ≈ 0.746). CPU speed on the Mac: **3.8 s per 518 px patch**.

---

## 1. System overview

Two frozen models, plus two trained parts that connect them:

```
                     ┌───────────────────────────────┐
  image ────────────►│ DINOv2-Emb U-Net   (frozen)   │──► P: 6×H×W class probabilities ─┐
    │                │ = the paper model             │     ("what is where")            │
    │                └───────────────────────────────┘                                  │
    │                ┌───────────────────────────────┐                                  ▼
    └───────────────►│ SAM 2.1 image encoder (frozen)│──► image features ──┐   ┌──────────────────┐
                     └───────────────────────────────┘                     │   │ CLASS HEAD       │
                                                                           ▼   │ (trained by us)  │──► class
  user prompts ─────────────────────────────────────────► SAM 2.1 MASK DECODER │ mask + P pooled  │
  (clicks / box / lasso / scribble)                        (fine-tuned by us)──┴──────────────────┘
                                                               │
                                                               ▼  object mask
                                        editor: paint the objects in class-priority order
                                                               ▼
                                       semantic map → export (CVAT XML, PNG)
```

### How the user interacts

Class choice is automatic by default.

| Input | Meaning | What the system does |
|---|---|---|
| Left click | "Object here" | SAM mask from the click; class = class head output; painted |
| Shift + left / Shift + right | Grow / shrink the current object | Adds a positive or negative point to the same SAM prompt |
| Right click on an object | "Wrong class" | Bans the current class; next most likely class from the class head |
| Box drag / lasso (`L`) / scribble (`S`) | Rough area or shape of an object | Sent to SAM as a box or mask prompt |
| Scroll | Mask granularity | Switches between SAM's 3 multimask outputs |
| `1`–`6` | Manual class (fallback) | Logged; this counts toward the "manual picks" metric |
| `Ctrl+Z`, `Enter` | Undo / accept image | — |

**Layering rule:** classes are painted in the same priority order as the dataset: Other < Building < Roof < Damage < Broken Window < Damaged roof. A window clicked on a facade therefore ends up on top of the building, just as in CVAT.

---

## 2. Components

### 2.1 Data layer — `src/dmgseg/data/`

- `cvat.py`:
  - parses **all four** shape types into a list of **objects** `{label, full-resolution mask, z_order}` per image;
  - also builds the **semantic map** (priority compositing, same as the paper).
- `split.py`: reproduces the paper split and writes `paper_split.json`.
- Each object's full mask is the **SAM training target**. The semantic map is the **evaluation target**.
- There are ~12k objects in total. Very small objects (under ~64 px, like tiny window boxes) are kept for evaluation, but sampled less often in training.

### 2.2 Semantic prior — `src/dmgseg/prior/`

- `unet_dinov2.py`: the `UNetDinoV2` class from the notebook, unchanged. First check that `best_model2_dinov2_640to518_simple_unet.pth` loads with `strict=True`; the name says "simple_unet", so the architecture may differ slightly.
- `infer.py`: the notebook's sliding-window `predict_full_image`, changed to return **probabilities** (softmax averaged over overlapping patches) instead of the argmax. It uses the ConvNeXt-L CLIP global embedding, as in the paper.
- **Out-of-fold priors (important for the class head):** on its own 246 training images, the paper model is overconfident and almost always right. A class head trained on those priors would learn to "always trust DINOv2" and then fail on new images. Two options:
  - **A (preferred):** 5-fold cross-training of DINOv2-Emb on the 246 training images, run on molab, with batch 16–32 instead of 4. Each image then gets a prior from a model that never saw it. This is about 5 runs of about 1–2 hours each.
  - **B (fallback):** corrupt the training-image priors with temperature smoothing, spatial blur and random class-probability noise, calibrated so their error rate matches the one on the 44 validation images.

### 2.3 SAM — `src/dmgseg/sam/`

- **Model:** SAM 2.1, Hiera-S (Hiera-B+ as an ablation). Its image encoder is frozen and runs with SAM's own preprocessing: 1024 px input and SAM's normalization. This avoids the problems of your earlier SAM-in-U-Net attempt.
- **Prompt support:** points, box, mask. The mask prompt is SAM's own 256×256 low-resolution input. Lasso and scribble are converted into prompts:
  - **lasso:** a coarse mask plus its bounding box;
  - **scribble:** up to N points sampled along the stroke plus a coarse mask of the stroke.
- **Fine-tuning** (`train/finetune_sam.py`, on molab):
  - Trainable: the mask decoder, the prompt encoder, and optionally LoRA on the last encoder blocks (ablation). Everything else is frozen.
  - Training follows the RITM/SAM protocol: per object, run **k = 1…8 iterative correction rounds**. The next click goes at the peak of the distance transform of the largest error region (positive click if it's a missed area, negative if it's extra). The previous low-res logits are fed back in as the mask prompt.
  - Each sample randomly starts from **click / box (±10% jitter) / lasso (a dilated, wobbly contour) / scribble (the mask skeleton plus noise)**.
  - Loss: focal + dice on the mask, plus MSE on the IoU prediction (as in SAM).
  - Image embeddings for the 246 training images (with flip augmentation) are cached once, so training runs only the decoder. It's fast: roughly minutes per epoch.

### 2.4 Class head — `src/dmgseg/classhead/` (a new model, part of the lab's "create a model" requirement)

**Inputs for each predicted object mask M:**

1. Mean and max of the DINOv2 probabilities P inside M (12 values).
2. SAM's mask token / object embedding from the decoder (256).
3. Mean-pooled SAM image features inside M (256).
4. Geometry: area, bounding box, aspect ratio, center (x, y), solidity (~8 values).
5. The class distribution of the surrounding region (a ring around M, 6 values). This is context such as "inside a facade" or "on the roof line".

**Model and output:** an MLP (512 → 256 → 6) that outputs 6-way softmax probabilities. A right click takes the next class in this ranking.

**Training data and loss:** SAM masks predicted for the ground-truth objects (so the head sees realistic, imperfect masks) combined with out-of-fold priors. The class is taken from the ground-truth object. Loss: cross-entropy with class weights (as in the paper) or focal loss.

**Baseline to compare against (training-free):** the class is the argmax of mean(P inside M). This is exactly the "without our model" line in the report.

### 2.5 Interaction simulator and evaluation — `src/dmgseg/eval/`

- `metrics.py`: global confusion matrix giving IoU, F1, precision and recall per class, plus mIoU and mF1 (fixes finding 0.2).
- `simulator.py`: a simulated user, so everything can be evaluated with no human:
  - **Object level** (the lab task): for each ground-truth object, 1–20 clicks → **NoC@85, NoC@90, mIoU@k curves**. Also a **class accuracy** and a **manual-pick rate** (the share of objects where the simulated user had to press 1–6 because the right-click cycle didn't reach the right class within 2 tries).
  - **Image level** (annotation tool / paper): start from the prior's argmax. At each step, find the largest wrong region against the ground-truth semantic map and choose left click, right click or lasso by the same rule a person would use. This gives **semantic mIoU vs. number of interactions**, and the **number of interactions needed to reach 90% mIoU per image**.
- All evaluation is on the **44 paper validation images**.

### 2.6 App — `app/`

- **Backend:** FastAPI + ONNX Runtime (CPU). Components:
  - `sam_encoder.onnx`
  - `sam_decoder.onnx`
  - `class_head.onnx`
  - `prior_dinov2.onnx`
  - `convnext_embed.onnx`

  Per image it caches the SAM features and P in memory and on disk (`.npz`).
- **Frontend:** one HTML page with a canvas, no framework. It has:
  - the image with a class overlay and an opacity slider;
  - hover preview of the object under the cursor (the decoder is fast enough);
  - lasso and scribble drawing, undo stack, class legend with counts;
  - an "Open image…" button, which is the lab's "test on a user-chosen image".
- **Export:** CVAT 1.1 XML (polygons from mask contours, so the paper pipeline can re-import them), PNG index masks, and `session.jsonl` with every interaction and its timing, used for the annotation-cost experiment.

---

## 3. molab workflow

- **Code:** in a GitHub repo, installed in the notebook with `pip install -e`. The notebooks stay thin.
- **Data and weights:** in a **private Hugging Face dataset repo**, downloaded with `huggingface_hub`. molab storage is limited and can change.
  - `dataset/`: images, annotations.xml, paper_split.json
  - `weights/`: best_model2_dinov2_*.pth
  - `cache/`: priors, SAM embeddings (created on molab)
  - `checkpoints/`: fine-tuned SAM, class head, exported ONNX files
- **Rules:**
  - no tokens in notebook cells (molab notebooks are public by link); use molab secrets or paste the token at runtime;
  - checkpoint every epoch and push to HF every N epochs;
  - every script can resume from the latest checkpoint (sessions last up to 12 h, and are killed after 90 min idle);
  - first run a 5-minute test to check the idle rule during long training cells.
- **Fallback:** the same scripts run on Colab (T4) with smaller batches. The config sets batch size and precision (bf16 on Blackwell, fp16 on T4).

**Notebooks (marimo):**

| Notebook | Purpose |
|---|---|
| `nb/00_setup_check.py` | GPU, versions, HF access, idle test |
| `nb/01_prior_cache.py` | Paper-model priors for all 290 images + recomputed paper metrics |
| `nb/02_prior_kfold.py` | 5-fold out-of-fold priors (option A) |
| `nb/03_sam_cache.py` | SAM image embeddings |
| `nb/04_sam_finetune.py` | Decoder fine-tuning |
| `nb/05_classhead.py` | Class head training |
| `nb/06_eval.py` | All experiments → CSV + figures |
| `nb/07_export_onnx.py` | ONNX export + numeric check (ONNX vs. torch, max abs diff) |

---

## 4. Repository layout

```
Комп зір/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/               # yaml per experiment
├── data/                  # Dataset_CVAT/ (gitignored), splits/
├── src/dmgseg/
│   ├── data/              # cvat.py, split.py, objects.py
│   ├── prior/             # unet_dinov2.py, infer.py, kfold.py
│   ├── sam/               # model wrapper, prompt encoding (click/box/lasso/scribble)
│   ├── classhead/         # features.py, model.py
│   ├── train/             # finetune_sam.py, train_classhead.py
│   ├── eval/              # metrics.py, simulator.py, run_eval.py
│   └── export/            # onnx.py
├── nb/                    # thin marimo notebooks for molab
├── app/                   # server.py, static/index.html, static/app.js
├── tests/                 # parser, metrics, simulator, onnx parity
├── artifacts/             # gitignored: caches, checkpoints, onnx, results
└── report/                # lab report + figures
```

---

## 5. Experiments (for the report and the paper)

| # | Question | Compared | Main metric |
|---|---|---|---|
| E1 | Does adapting SAM help on damage? | SAM zero-shot vs. fine-tuned decoder (vs. + LoRA) | NoC@85/90, mIoU@1/3/5 clicks, per class |
| E2 | Does the class head beat simple pooling? | argmax(mean P in M) vs. class head; OOF vs. corrupted priors | Top-1 / top-2 class accuracy, manual-pick rate |
| E3 | Which prompt works best for which class? | click vs. box vs. lasso vs. scribble | Interactions to 85% IoU, per class |
| E4 | Full labeling cost | prior only (0 interactions) → semantic mIoU vs. number of interactions | Interactions to 90% mIoU per image |
| E5 (paper) | SAM regions vs. Felzenszwalb superpixels as post-processing | Majority vote inside SAM automatic masks vs. inside Felzenszwalb superpixels (paper setup) | Global per-class IoU/F1, especially Broken Window |
| E6 (paper, opt.) | Effect of the dropped annotations | DINOv2-Emb retrained on the full labels (finding 0.1) | Per-class IoU on the 44 validation images |
| E7 (opt.) | Real annotation time | Tool vs. CVAT polygons on ~10 new images, timed by you | Minutes per image, agreement between the two |

---

## 6. Timeline

Today is Tue 29 Sep; the deadline is Sat 22 Nov (~7.5 weeks). **End of week 3 = a working lab** (zero-shot system + app prototype). Everything after that improves quality, and nothing after that is required to pass.

| Week | Dates | Deliverables | Done when |
|---|---|---|---|
| **1** | 29 Sep – 5 Oct | Repo + env (Python 3.12 venv; torch 2.2.2 CPU + onnxruntime on the Mac). `cvat.py` for all 4 shape types + visual check of polylines and masks. `paper_split.json`. `metrics.py` + tests. Upload to HF. `00_setup_check` on molab | Object and semantic masks look right for 10 random images; split = 246/44 |
| **2** | 6 – 12 Oct | The paper model loads (strict). `01_prior_cache`: priors for all 290 + **paper metrics recomputed consistently**. Start `02_prior_kfold` (runs in the background on molab) | Recomputed val F1 is close to the paper's 0.746 (same model); cached P on HF |
| **3** | 13 – 19 Oct | `03_sam_cache`. `simulator.py` (clicks first). **Zero-shot system:** SAM + argmax(mean P), right-click cycling. First E1/E2 baseline numbers. **App v0** (click + right click, torch or ONNX) | ✅ **Lab minimum works end to end** |
| **4** | 20 – 26 Oct | Box, lasso and scribble simulation + prompt encoding. `04_sam_finetune` with iterative training. E1, E3 | Fine-tuned model beats zero-shot on NoC@85 on the validation set |
| **5** | 27 Oct – 2 Nov | Class head (features, training on OOF priors). E2. Image-level simulator, E4 | Class head beats the pooling baseline on top-1 accuracy / manual-pick rate |
| **6** | 3 – 9 Nov | `07_export_onnx` + parity tests. **App v1:** all prompt types, hover preview, undo, export to CVAT XML/PNG, session log | Full annotation of one new image on the Mac without errors; CVAT import of the export works |
| **7** | 10 – 16 Nov | E5 (SAM vs. Felzenszwalb). E6 if time allows. E7 on ~10 images. Figures. Optional: "fix one, fix similar ones" prototype suggestions | All result tables and figures produced from scripts |
| **8** | 17 – 22 Nov | Lab report, README with a demo GIF, cleanup. Buffer for slips | Submitted |

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| molab limits change or GPUs disappear | Same scripts on Colab T4 (smaller batch, fp16). Decoder-only fine-tuning fits in 16 GB |
| SAM 2 → ONNX export breaks (the Hiera encoder is known to be awkward to export) | (a) Community SAM2 ONNX exporters; (b) run the SAM 2 **encoder** on molab only for dataset images and use SAM v1 ViT-B (official ONNX decoder, works with torch 2.2) in the app; (c) the app runs SAM v1 end to end |
| DINOv2 prior too slow on the Mac CPU | Larger stride (fewer patches), fp16/int8 quantized ONNX, or run the prior in a background thread so the user can start clicking right away (class = "?" until P arrives) |
| k-fold priors cost too much time | Option B (corrupted priors), fully described in 2.2 |
| Weights don't match the notebook architecture ("simple_unet") | Inspect the state_dict keys and rebuild the matching class; the notebook history on GitHub can help |
| Polylines and brush masks turn out to be something other than expected | Keep them out (paper behavior) but document it; E6 becomes "future work" |
| Deadline pressure | Cut in this order: E7 → prototype suggestions → E6 → LoRA ablation → scribble. The core (fine-tuned SAM + class head + app + E1/E2/E4) stays |

---

## 8. Mapping to the lab requirement

| Requirement | Where it's met |
|---|---|
| "Create a model that finds a segmentation mask for the clicked object" | The fine-tuned SAM decoder (the mask) + our class head (the class), both trained on our data (§2.3, §2.4). Evaluated with the standard interactive metrics NoC@85/90 and mIoU@k (§2.5) |
| "Create an app to test the model on a user-chosen image" | `app/`: open any image, click, get the mask and class, correct it, export (§2.6) |
| Extensions beyond the requirement | Automatic class choice, box/lasso/scribble prompts, right-click correction, a prior from the paper model, and a labeling export compatible with the paper pipeline |

## 9. Open decisions (defaults chosen, change if you want)

- SAM 2.1 **Hiera-S** as the main model, Hiera-B+ only as an ablation, for Mac CPU speed.
- A Hugging Face private repo for data (instead of Google Drive).
- **A new GitHub repo for this project** (not the paper's repo).
- The report language: English by default. Tell me if the lab needs Ukrainian.
