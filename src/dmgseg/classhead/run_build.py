"""Build the class-head cards on molab and upload them (notebook section F).

train: 246 paper-training images, out-of-fold priors, jittered first clicks
val:   44 validation images, the tool's prior (B2b), standard clicks
-> classhead/cards_{train,val}.npz on the Hub. Finished parts are skipped.
"""
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from dmgseg import hub, paths
from dmgseg.classhead.build import build_cards, save_cards
from dmgseg.data.split import load_split


def cards_job(workdir, device, status=None):
    status = status or (lambda **_: None)
    workdir = Path(workdir)
    try:
        split = load_split()
        status(state="downloading priors")
        snapshot_download(paths.HF_REPO, repo_type="dataset", local_dir=str(workdir),
                          allow_patterns=[f"priors/oof_{paths.PRIOR_RUN}/*.npz"]
                          + [f"priors/{paths.PRIOR_RUN}/{n}.npz" for n in split["val"]])
        from dmgseg.sam.predictor import SamClicker
        clicker = SamClicker("small", device)
        jobs = {
            "train": (split["train"], workdir / "priors" / f"oof_{paths.PRIOR_RUN}", True),
            "val": (split["val"], workdir / "priors" / paths.PRIOR_RUN, False),
        }
        for name, (names, prior_dir, jitter) in jobs.items():
            target = f"classhead/cards_{name}.npz"
            if HfApi().file_exists(paths.HF_REPO, target, repo_type="dataset"):
                continue
            status(state="building", stage=f"{name} ({len(names)} images)")
            cards = build_cards(names, prior_dir, clicker, jitter=jitter,
                                progress=lambda d, t, n=name: status(stage=f"{n}: {d}/{t} images") if d % 10 == 0 else None)
            out = workdir / target
            save_cards(cards, out)
            hub.upload(out, target, message=f"Class-head cards ({name}): {cards['X'].shape[0]} cards, "
                                            f"{len(cards['obj_image'])} objects")
        status(state="done")
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise
