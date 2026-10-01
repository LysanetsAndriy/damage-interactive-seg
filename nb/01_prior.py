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
        subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
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
    print("weights:", paths.PRIOR_WEIGHTS.name, paths.PRIOR_WEIGHTS.exists())
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
    print("torch", torch.__version__)
    return DEVICE, WORKDIR, paths, torch


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
def _(DEVICE, WORKDIR, embed_model, eval_button, mo, paths):
    mo.stop(not eval_button.value)

    import json

    from dmgseg.data.cvat import CLASS_NAMES, PAPER_KINDS
    from dmgseg.eval.metrics import summary
    from dmgseg.prior.infer import evaluate_full_images
    from dmgseg.prior.train import split_annotations
    from dmgseg.prior.unet_dinov2 import load_prior_model

    paper_model = load_prior_model(paths.PRIOR_WEIGHTS, DEVICE)
    _, val_anns = split_annotations()
    paper_eval = {}
    for _labels, _kinds in (("paper_labels", PAPER_KINDS), ("fixed_labels", ("polygon", "box", "mask"))):
        _m = evaluate_full_images(paper_model, embed_model, val_anns, _kinds,
                                  patch_size=518, stride=300, device=DEVICE)
        paper_eval[_labels] = summary(_m.compute(), CLASS_NAMES)

    _out = WORKDIR / "paper_model_eval.json"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(json.dumps(paper_eval, indent=1))
    from dmgseg import hub as _hub
    _hub.upload(_out, "runs/paper_model_eval/metrics.json")

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


if __name__ == "__main__":
    app.run()
