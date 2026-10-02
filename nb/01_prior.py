import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 01 · DINOv2-Emb prior: verify the paper model, retrain on fixed labels

    Needs: a GPU (notebook specs button) and `HF_TOKEN` in the Secrets panel.
    All code lives in the GitHub repo; this notebook only calls it.
    """)
    return (mo,)


@app.cell
def _():
    # Get or update the code and make it importable.
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPO_URL = "https://github.com/LysanetsAndriy/damage-interactive-seg.git"
    REPO = Path.home() / "damage-interactive-seg"
    if (REPO / ".git").exists():
        subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"])
    else:
        subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)

    _deps = ["timm", "albumentations", "huggingface_hub", "scikit-image",
             "scikit-learn", "opencv-python-headless", "pyyaml"]
    try:
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, *_deps], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_deps], check=True)

    sys.path.insert(0, str(REPO / "src"))
    _commit = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--oneline"],
                             capture_output=True, text=True).stdout.strip()
    print("code at:", _commit)
    print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
    return Path, REPO


@app.cell
def _(Path):
    import torch

    from dmgseg import hub

    DATA = hub.download_data(local_dir=str(Path.home() / "dmgseg_data"))
    WORKDIR = Path("/tmp/dmgseg")  # patches + checkpoints; checkpoints also go to HF
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    from dmgseg import paths
    print("data:", DATA)
    print("weights:", paths.PAPER_WEIGHTS.name, paths.PAPER_WEIGHTS.exists())
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
    print("torch", torch.__version__)
    return DEVICE, WORKDIR, hub, paths, torch


@app.cell
def _(DEVICE):
    from dmgseg.prior.embedding import image_embedding, load_embedding_model

    embed_model = load_embedding_model(DEVICE)

    def embed_fn(image):
        return image_embedding(embed_model, image, DEVICE)
    return embed_fn, embed_model


@app.cell
def _(mo):
    mo.md("""
    ## A. Is this the published model?

    Full-image evaluation on the 44 validation images with the notebook's settings
    (518 px patches, stride 300). If the weights are the published ones, the
    **paper-style** numbers on **paper labels** should be close to the paper:
    mean F1 ≈ 0.746, mean IoU ≈ 0.471.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Run evaluation of the paper model (~5 min)")
    eval_button
    return (eval_button,)


@app.cell
def _(DEVICE, WORKDIR, embed_model, eval_button, hub, mo, paths):
    mo.stop(not eval_button.value)

    import json

    from dmgseg.data.cvat import CLASS_NAMES, PAPER_KINDS
    from dmgseg.eval.metrics import summary
    from dmgseg.prior.infer import evaluate_full_images
    from dmgseg.prior.train import split_annotations
    from dmgseg.prior.unet_dinov2 import load_prior_model

    paper_model = load_prior_model(paths.PAPER_WEIGHTS, DEVICE)
    _, val_anns = split_annotations()
    paper_eval = {}
    for _labels, _kinds in (("paper_labels", PAPER_KINDS), ("fixed_labels", ("polygon", "box", "mask"))):
        _m = evaluate_full_images(paper_model, embed_model, val_anns, _kinds,
                                  patch_size=518, stride=300, device=DEVICE)
        paper_eval[_labels] = summary(_m.compute(), CLASS_NAMES)

    _out = WORKDIR / "paper_model_eval.json"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(json.dumps(paper_eval, indent=1))
    hub.upload(_out, "runs/paper_model_eval/metrics.json")

    mo.md("\n".join(
        f"- **{k}**: paper-style mF1 {v['paper/mf1']:.4f}, paper-style mIoU {v['paper/miou']:.4f} · "
        f"global mIoU {v['global/miou']:.4f}, global mF1 {v['global/mf1']:.4f}"
        for k, v in paper_eval.items()))
    return


@app.cell
def _(mo):
    mo.md("""
    ## B. Retrain on fixed labels (paper settings)

    `configs/prior_dinov2_6c_paper.yaml`. Resumes automatically from
    `runs/<run_name>/last.pt` on Hugging Face if a previous session was cut off.
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start / resume training (hours)")
    train_button
    return (train_button,)


@app.cell
def _(DEVICE, REPO, WORKDIR, embed_fn, mo, train_button):
    mo.stop(not train_button.value)

    from dmgseg.config import load_config
    from dmgseg.prior.train import train

    cfg = load_config(REPO / "configs" / "prior_dinov2_6c_paper.yaml")
    run_dir = train(cfg, WORKDIR, embed_fn, device=DEVICE)
    print("finished:", run_dir)
    return


@app.cell
def _(mo):
    mo.md("""
    ## C. Compare checkpoints

    Full-image evaluation (as in A) of the paper model, the new `best.pt`
    (lowest validation loss) and the new `final.pt` (after the last epoch).
    """)
    return


@app.cell
def _(mo):
    compare_button = mo.ui.run_button(label="Compare checkpoints (~3 min)")
    compare_button
    return (compare_button,)


@app.cell
def _(DEVICE, WORKDIR, compare_button, embed_model, hub, mo, paths):
    mo.stop(not compare_button.value)

    from dmgseg.prior.compare import compare_checkpoints, export_final_weights, results_table

    _run = WORKDIR / "runs" / "dinov2_emb_6c_paper_fixedlabels"
    if not (_run / "final.pt").exists():
        export_final_weights(_run)
    comparison = compare_checkpoints(
        {"paper model": paths.PAPER_WEIGHTS, "new best.pt": _run / "best.pt", "new final.pt": _run / "final.pt"},
        embed_model, DEVICE, _run / "comparison.json")
    hub.upload(_run / "comparison.json", f"runs/{_run.name}/comparison.json")
    mo.md("**Fixed labels**\n\n" + results_table(comparison, "fixed_labels")
          + "\n\n**Paper labels**\n\n" + results_table(comparison, "paper_labels"))
    return (comparison,)


@app.cell
def _(mo):
    mo.md("""
    ## D. Priors for the class head (~3.5 h)

    The tool's prior is **B2b** (`runs/dinov2_emb_6c_b2b/final.pt`, chosen after E).

    1. B2b priors for all 290 images -> `priors/dinov2_emb_6c_b2b/` (~5 min).
    2. Five fold models with B2b's recipe on the 246 training images; each predicts
       its held-out fold -> `priors/oof_dinov2_emb_6c_b2b/`. Every training image
       gets a prior from a model that never saw it.

    Runs in a background thread (survives cell interruptions). Finished parts are
    skipped, so pressing the button again resumes.
    """)
    return


@app.cell
def _(mo):
    kfold_button = mo.ui.run_button(label="B2b priors + 5 folds (~3.5 h)")
    kfold_button
    return (kfold_button,)


@app.cell
def _(DEVICE, REPO, WORKDIR, embed_fn, embed_model, hub, kfold_button, mo):
    mo.stop(not kfold_button.value)

    from dmgseg.config import load_config as _load_config
    from dmgseg.prior.experiments import start_in_background as _start, status_writer
    from dmgseg.prior.kfold import run_prior_job

    kfold_cfg = _load_config(REPO / "configs" / "prior_dinov2_6c_kfold.yaml")
    d_message = _start(
        "d_queue", kfold_cfg, WORKDIR, embed_model, embed_fn, DEVICE,
        hub.prior_weights(WORKDIR), target=run_prior_job,
        status=status_writer(WORKDIR, "d_queue"),
    )
    mo.md(f"**{d_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    d_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="1m", label="Auto-refresh")
    d_refresh
    return (d_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, d_refresh, mo, paths):
    d_refresh

    import json as _djson

    from dmgseg.prior.experiments import is_running as _d_running

    _st_path = WORKDIR / "runs" / "d_queue" / "status.json"
    _st = _djson.loads(_st_path.read_text()) if _st_path.exists() else {}
    _n_prior = len(list((WORKDIR / "priors" / paths.PRIOR_RUN).glob("*.npz")))
    _n_oof = len(list((WORKDIR / "priors" / f"oof_{paths.PRIOR_RUN}").glob("*.npz")))
    _rows = []
    for _k in range(5):
        _h = WORKDIR / "runs" / f"dinov2_emb_6c_b2b_kfold_fold{_k}" / "history.json"
        _hist = _djson.loads(_h.read_text()) if _h.exists() else []
        if _hist:
            _r = _hist[-1]
            _rows.append(f"| {_k} | {len(_hist)}/15 | {_r['val']['global/miou']:.4f} | {_r['val']['global/mf1']:.4f} |")
    mo.md(
        f"**Job:** {'running' if _d_running('d_queue') else 'not running'} · **state:** {_st.get('state', '-')} · "
        f"**stage:** {_st.get('stage', '-')} · **updated:** {_st.get('updated', '-')}\n\n"
        f"**Priors:** B2b {_n_prior}/290 · out-of-fold {_n_oof}/246\n\n"
        + "| fold | epochs | held-out mIoU | held-out mF1 |\n|---|---|---|---|\n" + "\n".join(_rows)
        + (f"\n\n**Error**\n```\n{_st['error'][-1500:]}\n```" if _st.get('error') else "")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## E. B2a / B2b: training settings for a 96 GB GPU (~1.5 h)

    - **B2a**: batch 16, LR 4e-5, cosine schedule with 1 warm-up epoch, 15 epochs.
    - **B2b**: B2a + layer-wise LR decay 0.8 for the DINOv2 encoder.

    Both use the final epoch, then everything is compared on full validation
    images against the paper model and B1 (paper settings, retrained).
    Runs in a background thread: the cell returns at once and the training
    survives interruptions of the cell. Progress: `runs/b2_queue/status.json`
    and `runs/<run>/history.json` on Hugging Face.
    """)
    return


@app.cell
def _(mo):
    b2_button = mo.ui.run_button(label="Run B2a then B2b, then compare (~1.5 h)")
    b2_button
    return (b2_button,)


@app.cell
def _(DEVICE, REPO, WORKDIR, b2_button, embed_fn, embed_model, hub, mo, paths):
    mo.stop(not b2_button.value)

    from dmgseg.prior.experiments import start_in_background

    hub.download_if_exists("runs/dinov2_emb_6c_paper_fixedlabels/final.pt", WORKDIR)
    b2_message = start_in_background(
        "b2_queue",
        [REPO / "configs" / "prior_dinov2_6c_b2a.yaml", REPO / "configs" / "prior_dinov2_6c_b2b.yaml"],
        WORKDIR, embed_model, embed_fn, DEVICE,
        baselines={
            "paper model": paths.PAPER_WEIGHTS,
            "B1 paper settings": WORKDIR / "runs" / "dinov2_emb_6c_paper_fixedlabels" / "final.pt",
        },
    )
    mo.md(f"**{b2_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    b2_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="30s", label="Auto-refresh")
    b2_refresh
    return (b2_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, b2_refresh, mo, torch):
    b2_refresh

    import json as _json

    from dmgseg.prior.experiments import is_running as _is_running

    _rows = []
    for _run in ("dinov2_emb_6c_b2a", "dinov2_emb_6c_b2b"):
        _h = WORKDIR / "runs" / _run / "history.json"
        for _r in (_json.loads(_h.read_text()) if _h.exists() else []):
            _rows.append(f"| {_run[-3:].upper()} | {_r['epoch'] + 1} | {_r['seconds']:.0f}s | {_r['train_loss']:.4f} | {_r['val_loss']:.4f} | {_r['val']['global/miou']:.4f} | {_r['val']['global/mf1']:.4f} | {_r['lr']:.1e} |")
    _st_path = WORKDIR / "runs" / "b2_queue" / "status.json"
    _st = _json.loads(_st_path.read_text()) if _st_path.exists() else {}
    _gpu = f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated / {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak" if torch.cuda.is_available() else "no GPU"
    mo.md(
        f"**Queue:** {'running' if _is_running('b2_queue') else 'not running'} · **state:** {_st.get('state', '-')} {_st.get('run', '')} · **updated:** {_st.get('updated', '-')} · **GPU:** {_gpu}\n\n"
        + "| run | epoch | time | train loss | val loss | val mIoU | val mF1 | LR |\n|---|---|---|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + (f"\n\n**Result**\n\n{_st['table']}" if _st.get('table') else "")
        + (f"\n\n**Error**\n```\n{_st['error'][-1500:]}\n```" if _st.get('error') else "")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## F. Class head: description cards (~20-30 min)

    The simulated user clicks every object; SAM gives 3 candidate masks per first
    click (+ refined masks after clicks 2-3). Each mask becomes a 288-number card
    (prior inside/around the mask, SAM appearance, geometry, SAM score).

    - **train**: 246 training images, out-of-fold priors, extra random first clicks
    - **val**: 44 validation images, B2b priors

    Result: `classhead/cards_{train,val}.npz` on Hugging Face (the head trains on the Mac).
    """)
    return


@app.cell
def _(mo):
    f_button = mo.ui.run_button(label="Build class-head cards (~30 min)")
    f_button
    return (f_button,)


@app.cell
def _(DEVICE, WORKDIR, f_button, mo):
    mo.stop(not f_button.value)

    from dmgseg.sam.install import ensure_sam2
    from dmgseg.classhead.run_build import cards_job
    from dmgseg.prior.experiments import start_in_background as _f_start, status_writer as _f_status

    print("SAM 2:", ensure_sam2())
    f_message = _f_start("f_queue", WORKDIR, DEVICE, target=cards_job, status=_f_status(WORKDIR, "f_queue"))
    mo.md(f"**{f_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    f_refresh = mo.ui.refresh(options=["30s", "1m"], default_interval="30s", label="Auto-refresh")
    f_refresh
    return (f_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, f_refresh, mo):
    f_refresh

    import json as _fjson

    _p = WORKDIR / "runs" / "f_queue" / "status.json"
    _fst = _fjson.loads(_p.read_text()) if _p.exists() else {}
    mo.md(f"**state:** {_fst.get('state', '-')} · **stage:** {_fst.get('stage', '-')} · "
          f"**updated:** {_fst.get('updated', '-')}"
          + (f"\n\n```\n{_fst['error'][-1500:]}\n```" if _fst.get("error") else ""))
    return


@app.cell
def _(mo):
    mo.md("""
    ## G. Patch size / scale experiments (~3 h, waits for F)

    B2b recipe, only the crops and the DINOv2 input change:

    - **P1** 518 px crops, native (objects 100 %, less context)
    - **P2** 640 px crops at 644 px input (objects 100 %, same context as B2b)
    - **P3** 800 px crops shrunk to 518 (objects 65 %, more context)

    Each is evaluated at its own scale, against the paper model and B2b (at native
    518 px crops and at its 640->518 training scale). Starts automatically when the
    card job (F) is finished. Progress in the table below.
    """)
    return


@app.cell
def _(mo):
    g_button = mo.ui.run_button(label="Queue P1, P2, P3 (starts after F)")
    g_button
    return (g_button,)


@app.cell
def _(DEVICE, REPO, WORKDIR, embed_fn, embed_model, g_button, hub, mo, paths):
    mo.stop(not g_button.value)

    from dmgseg.config import load_config as _g_load
    from dmgseg.prior.compare import checkpoint_spec as _g_spec
    from dmgseg.prior.experiments import start_in_background as _g_start

    _b2b_cfg = _g_load(REPO / "configs" / "prior_dinov2_6c_b2b.yaml")
    _b2b = hub.prior_weights(WORKDIR)
    g_message = _g_start(
        "g_queue",
        [REPO / "configs" / f"prior_dinov2_6c_{_k}.yaml" for _k in ("p1", "p2", "p3")],
        WORKDIR, embed_model, embed_fn, DEVICE,
        baselines={
            "paper model": paths.PAPER_WEIGHTS,
            "B2b": _b2b,
            "B2b at training scale": {**_g_spec(_b2b_cfg, _b2b),
                                      "eval": {"patch_size": 640, "stride": 370, "model_input": 518}},
        },
        wait_for="h_queue",
    )
    mo.md(f"**{g_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    g_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="1m", label="Auto-refresh")
    g_refresh
    return (g_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, g_refresh, mo):
    g_refresh

    import json as _gjson

    _rows = []
    for _run in ("dinov2_emb_6c_p1_518native", "dinov2_emb_6c_p2_640to644", "dinov2_emb_6c_p3_800to518"):
        _h = WORKDIR / "runs" / _run / "history.json"
        for _r in (_gjson.loads(_h.read_text()) if _h.exists() else [])[-1:]:
            _rows.append(f"| {_run} | {_r['epoch'] + 1}/15 | {_r['seconds']:.0f}s | {_r['val']['global/miou']:.4f} | {_r['val']['global/mf1']:.4f} |")
    _p = WORKDIR / "runs" / "g_queue" / "status.json"
    _gst = _gjson.loads(_p.read_text()) if _p.exists() else {}
    mo.md(
        f"**state:** {_gst.get('state', '-')} · **stage:** {_gst.get('stage', '-')} · **updated:** {_gst.get('updated', '-')}\n\n"
        + "| run | epoch | time/epoch | val mIoU (patches) | val mF1 |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + (f"\n\n**Result**\n\n{_gst['table']}" if _gst.get("table") else "")
        + (f"\n\n```\n{_gst['error'][-1500:]}\n```" if _gst.get("error") else "")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## H. Class head B: fusion of DINOv2 + SAM feature maps (~1.5 h, waits for G)

    Feature-map crops (frozen original DINOv2 -> PCA 256, SAM 256, prior 6, candidate
    mask) on a 16x16 grid around each click; a small conv network predicts the class
    and the mask quality. Variants: full / no DINOv2 / no SAM, 3 seeds each; evaluated
    with center and random first clicks. Results: `classhead/head_b_results.json`.
    """)
    return


@app.cell
def _(mo):
    h_button = mo.ui.run_button(label="Queue class head B (starts after G)")
    h_button
    return (h_button,)


@app.cell
def _(DEVICE, WORKDIR, h_button, mo):
    mo.stop(not h_button.value)

    from dmgseg.sam.install import ensure_sam2 as _h_sam2
    from dmgseg.classhead.run_h import h_job
    from dmgseg.prior.experiments import start_in_background as _h_start, status_writer as _h_status

    _h_sam2()
    h_message = _h_start("h_queue", WORKDIR, DEVICE, target=h_job, status=_h_status(WORKDIR, "h_queue"),
                         wait_for="g_queue")
    mo.md(f"**{h_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    h_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="1m", label="Auto-refresh")
    h_refresh
    return (h_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, h_refresh, mo):
    h_refresh

    import json as _hjson

    _p = WORKDIR / "runs" / "h_queue" / "status.json"
    _hst = _hjson.loads(_p.read_text()) if _p.exists() else {}
    mo.md(f"**state:** {_hst.get('state', '-')} · **stage:** {_hst.get('stage', '-')} · **updated:** {_hst.get('updated', '-')}"
          + (f"\n\n{_hst['table']}" if _hst.get("table") else "")
          + (f"\n\n```\n{_hst['error'][-1500:]}\n```" if _hst.get("error") else ""))
    return


@app.cell
def _(mo):
    mo.md("""
    ## I. Fine-tune SAM's decoder on the damage dataset (~1 h)

    SAM 2.1-S image encoder frozen; prompt encoder + mask decoder trained with
    simulated clicks (and boxes) on the 196 training images outside fold 0; fold 0
    is the dev set for early stopping. Then zero-shot vs fine-tuned SAM on the 44
    validation images (NoC@85/90). Result: `sam/finetuned_decoder.pt`.
    """)
    return


@app.cell
def _(mo):
    i_button = mo.ui.run_button(label="Fine-tune SAM (~1 h)")
    i_button
    return (i_button,)


@app.cell
def _(DEVICE, WORKDIR, i_button, mo):
    mo.stop(not i_button.value)

    from dmgseg.sam.install import ensure_sam2 as _i_sam2
    from dmgseg.sam.run_ft import sam_ft_job
    from dmgseg.prior.experiments import start_in_background as _i_start, status_writer as _i_status

    _i_sam2()
    i_message = _i_start("i_queue", WORKDIR, DEVICE, target=sam_ft_job, status=_i_status(WORKDIR, "i_queue"))
    mo.md(f"**{i_message}**")
    return


@app.cell(hide_code=True)
def _(mo):
    i_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="30s", label="Auto-refresh")
    i_refresh
    return (i_refresh,)


@app.cell(hide_code=True)
def _(WORKDIR, i_refresh, mo):
    i_refresh

    import json as _ijson

    _p = WORKDIR / "runs" / "i_queue" / "status.json"
    _ist = _ijson.loads(_p.read_text()) if _p.exists() else {}
    mo.md(f"**state:** {_ist.get('state', '-')} · **stage:** {_ist.get('stage', '-')} · **updated:** {_ist.get('updated', '-')}"
          + (f"\n\n{_ist['table']}" if _ist.get("table") else "")
          + (f"\n\n```\n{_ist['error'][-1500:]}\n```" if _ist.get("error") else ""))
    return


@app.cell
def _(mo):
    mo.md("""
    ## J. Class-head cards with the fine-tuned SAM (~1.5 h)

    The same cards as in F, but the masks come from the fine-tuned SAM decoder
    (section I): `classhead/cards_{train,val}_samft.npz`. Head A is then retrained
    on them (on the Mac).
    """)
    return


@app.cell
def _(mo):
    j_button = mo.ui.run_button(label="Build cards with fine-tuned SAM (~1.5 h)")
    j_button
    return (j_button,)


@app.cell
def _(DEVICE, WORKDIR, j_button, mo):
    mo.stop(not j_button.value)

    from dmgseg.classhead.run_build import cards_job as _j_cards
    from dmgseg.prior.experiments import start_in_background as _j_start, status_writer as _j_status
    from dmgseg.sam.install import ensure_sam2 as _j_sam2

    _j_sam2()
    j_message = _j_start("j2_queue", WORKDIR, DEVICE, target=_j_cards, status=_j_status(WORKDIR, "j2_queue"),
                         sam_weights="sam/finetuned_decoder.pt", suffix="_samft")
    mo.md(f"**{j_message}**")
    return


if __name__ == "__main__":
    app.run()
