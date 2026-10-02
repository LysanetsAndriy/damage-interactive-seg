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

### 0.6 Results so far (2026-10-01, molab)

Full-image evaluation on the 44 validation images (518 px patches, stride 300), fixed labels:

| Model | Global mIoU | Global mF1 | Paper-style mIoU |
|---|---|---|---|
| Paper model (`paper_dinov2_emb_6class.pth`) | **0.607** | **0.744** | 0.468 |
| Retrained on fixed labels, final epoch (`runs/.../final.pt`) | 0.605 | 0.742 | 0.474 |
| Retrained, lowest val loss (`runs/.../best.pt`, epoch 2) | 0.569 | 0.710 | 0.451 |

- The published model is verified: paper-style mF1 0.7462 / mIoU 0.4711 reproduced exactly.
- **E6 answered: the brush-mask fix makes no measurable difference** (0.605 vs 0.607 is run-to-run noise).
- **The paper's training pipeline reproduces** the published quality from scratch, in about 1 hour on molab.
- Choosing the checkpoint by lowest validation loss picks a weak early epoch. Future runs (k-fold) use the final epoch.
- **The paper model stays the prior for the tool.**

### 0.6b Training settings for the 96 GB GPU (E, 2026-10-01): B2b becomes the prior

Full-image global mIoU on the validation set (fixed labels): paper model 0.607 · B1 (paper recipe, retrained) 0.605 · **B2a** (batch 16, LR 4e-5, cosine + warm-up, 15 epochs) 0.599 · **B2b** (B2a + layer-wise LR decay 0.8) **0.610**.

- No setting is clearly better (differences ≈ run-to-run noise). The T4-era batch size was **not** the bottleneck; the data size is.
- Layer-wise LR decay is the only change that helps consistently (Damage, Broken Window, Damaged roof).
- **Decision:** the tool's prior is **B2b** (`runs/dinov2_emb_6c_b2b/final.pt`, `paths.PRIOR_RUN`). Its recipe never uses the validation set (fixed schedule, final epoch), unlike the paper model's lowest-val-loss selection. The paper model (`paths.PAPER_WEIGHTS`) stays the published reference.
- The k-fold (section D) uses the B2b recipe → `priors/dinov2_emb_6c_b2b/` (all images) and `priors/oof_dinov2_emb_6c_b2b/` (out-of-fold, training images).

### 0.6c Class head A beats the rules (2026-10-02)

Validation, 1,314 clickable objects (44 images), masks after 3 clicks; head A = MLP on 288-d cards, trained on 6,899 objects (out-of-fold priors), early stopping on 2,214 dev objects (fold-0 images); 3 seeds:

| | Rules (prior average + SAM score) | Head A |
|---|---|---|
| Left click gives the right class | 56.8 % | **80.4 % ± 0.8** |
| Manual key needed (right class 3rd or lower) | 21.5 % | **2.8 % ± 0.2** |
| First-click mask IoU (oracle 0.658) | 0.565 | **0.591 ± 0.003** |

Per class: Broken Window 55.5 → 85.0 % (manual 24.6 → 0.1 %), Damage 58.9 → 71.5 %, Building 76.7 → 85.0 %, Roof 27.5 → 35.0 %; Damaged roof got worse (67.6 → 62.2 %). The mask choice gains little (Building stays at 0.28 vs the 0.61 oracle): the motivation for B (feature maps).

Robustness to where the user clicks first (saved head A, the same 1,314 objects):

| First click | 1st-click IoU: SAM / head | Right class: rules / head | Manual: rules / head |
|---|---|---|---|
| center (protocol) | 0.565 / 0.588 | 56.8 % / 81.4 % | 21.5 % / 3.1 % |
| random (≥ 30 % depth) | 0.554 / 0.581 | 56.2 % / 81.3 % | 21.8 % / 3.0 % |
| anywhere inside | 0.531 / 0.569 | 56.4 % / 82.0 % | 22.0 % / 3.0 % |

The class choice is insensitive to the click position; mask quality drops slightly for careless clicks, and the head stays ahead of SAM's own choice.

**Class head B v1 (fusion on 16×16 feature-map crops) is worse than A** (molab, section H; 3 seeds each, validation):

| | Right class | Manual | 1st-click IoU |
|---|---|---|---|
| Rules | 56.8 % | 21.5 % | 0.565 |
| **Head A** | **80.4 %** | **2.8 %** | **0.591** |
| B full (DINOv2 + SAM + prior) | 68.7 ± 2.6 % | 5.8 % | 0.514 |
| B without DINOv2 | 71.1 ± 0.7 % | 5.6 % | 0.540 |
| B without SAM | 71.6 ± 0.3 % | 5.3 % | 0.525 |

Diagnosis: (1) one crop per click is sized by the largest of SAM's 3 candidates, so small candidates cover a few cells of the 16×16 grid and the exact per-mask statistics that A uses are lost; (2) unstable, majority-biased training (e.g. no_dino seed 0: Broken Window 99 %, Damage 4 %); (3) more inputs → worse (overfitting with ~50k samples). Not used in the app.

Inference-scale test: B2b at 640→518 crops 0.6096 vs 518 native 0.6100, no difference. RandomResizedCrop(0.6–0.8) already zooms training patches by 1.12–1.29×, so the effective training scale was ~100 %.

### 0.6d Crop size / scale experiments (G, 2026-10-02)

Full-image global IoU on the validation set (fixed labels), B2b recipe, each model at its own scale:

| Model (crop → input) | mIoU | Building | Roof | Damage | Broken Window | Damaged roof |
|---|---|---|---|---|---|---|
| paper model (518→518 eval) | 0.607 | 0.670 | 0.522 | 0.549 | 0.434 | 0.530 |
| B2b (640→518 train; 518 eval) | 0.610 | 0.681 | 0.509 | 0.559 | 0.439 | 0.537 |
| P1 518→518 (more detail, less context) | 0.605 | 0.684 | 0.474 | **0.569** | 0.419 | **0.547** |
| P2 640→644 (more detail, same context) | 0.612 | **0.694** | 0.492 | 0.553 | **0.454** | 0.537 |
| P3 800→518 (more context) | **0.619** | **0.694** | **0.564** | 0.542 | 0.438 | 0.533 |

- More **context** (P3) helps the large classes, especially **Roof +5.5** and Building, and gives the best mean (+0.9 over B2b). More **detail** (P2) gives the best Broken Window (+1.5). Less context (P1) hurts Roof (−3.5).
- All differences are ≤ 1 point overall, below the 1.5-point bar set for replacing the prior, so **B2b stays the prior**. Worth reporting as a per-class context/detail trade-off; a P2+P3 ensemble is a cheap follow-up (inference only).

### 0.6e SAM 2.1-S decoder fine-tuning (I, 2026-10-02)

Prompt encoder + mask decoder trained (image encoder frozen) with simulated clicks and boxes on 196 training images, early stopping on 50 dev images (fold 0); ~1 min/epoch on molab. Validation (44 images, 1,477 objects incl. Other, center first click, 20 clicks):

| SAM | IoU@1 | IoU@3 | IoU@5 | NoC@85 | NoC@90 | never reaches 90 % |
|---|---|---|---|---|---|---|
| zero-shot | 0.534 | 0.723 | 0.784 | 8.56 | 12.83 | 43.0 % |
| **fine-tuned** | **0.574** | **0.781** | **0.837** | **6.29** | **10.19** | **30.5 %** |

27 % fewer clicks to reach 85 % IoU. Per class NoC@85: Other 11.3→7.4, Broken Window 7.2→5.0, Damage 10.4→8.2, Roof 11.9→10.5, Damaged roof 14.1→12.7, Building 5.2→5.0. Weights: `sam/finetuned_decoder.pt` (Hub).

### 0.6f Auto pre-label and image-level evaluation (2026-10-02)

**Pre-label:** prior blobs snapped by SAM with box + point prompts; "grow-only" (SAM may only add unlabeled pixels to a blob). 10 val images: prior map 0.6705 → pre-label 0.6856 mIoU. Without grow-only, SAM spills cost Roof −12 points.

**Image-level evaluation** (44 val images, global mIoU after k interactions; a simulated annotator fixes the largest wrong region with the action a person would take, undoing actions that make the image worse; the undo counts as an interaction):

| Start, SAM | k=0 | 2 | 3 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| empty, zero-shot (v1 user, no undo) | 0.107 | 0.180 | 0.201 | 0.234 | 0.294 | 0.328 | 0.349 |
| empty, fine-tuned | 0.107 | 0.175 | 0.227 | 0.244 | 0.285 | 0.317 | 0.329 |
| **pre-label, zero-shot** | **0.618** | 0.634 | 0.637 | 0.656 | 0.682 | 0.697 | 0.681 |
| **pre-label, fine-tuned** | 0.613 | 0.649 | 0.653 | 0.662 | 0.642 | 0.692 | 0.697 |

- **The automatic draft is worth far more than 20 clicks:** it starts at 0.61–0.62, while 20 clicks from an empty image reach only 0.33–0.35.
- Fine-tuned SAM helps object-level clicking a lot (NoC@85 −27 %) but whole images only slightly (± noise): the image mIoU is dominated by class decisions and large regions. ~25 % of the simulated actions were undone as harmful.

### 0.6g Head A retrained on the fine-tuned SAM's masks (step 3, 2026-10-03)

Cards rebuilt with the fine-tuned SAM decoder (molab J2, `classhead/cards_*_samft.npz`); head A retrained (3 seeds, dev-selected). Validation, 1,314 objects:

| | SAM's own first mask | Head A (zero-shot SAM cards) | **Head A (fine-tuned SAM cards)** |
|---|---|---|---|
| Right class on the left click | — | 80.4 % ± 0.8 | **80.9 % ± 0.1** |
| Manual key needed | — | 2.8 % | **2.1 %** |
| First-click mask IoU (oracle) | 0.565 → 0.596 with fine-tuned SAM | 0.591 (0.658) | **0.621** (0.691) |

Per class (right class on the left click, old → new head): **Roof 35.0 → 57.5 %**, **Damaged roof 62.2 → 70.3 %**, Damage 71.5 → 73.4 %, Broken Window 85.0 → 85.1 %, Building 85.0 → 81.7 %. More stable across seeds (± 0.1 %). The app uses fine-tuned SAM + this head (`classhead/head_a_samft.pt`).

### 0.7 SAM 2.1 runs on the Mac with torch 2.2 (2026-10-01)

Installed from source without its torch pin (README). Hiera-S on the i7 CPU: **image encoder 1.85 s, decoder ~67 ms per click**, so the app can use PyTorch directly; ONNX becomes optional. On a first sample, SAM's own score often picks the wrong one of the 3 first-click masks (Building 1-click IoU 0.01 → 0.63 with the best mask). **Choosing the mask with the DINOv2 prior** is a candidate extra contribution (E2b).

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
