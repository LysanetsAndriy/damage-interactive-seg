"""Compare prior checkpoints with the notebook's full-image evaluation."""
import json
from pathlib import Path

import torch

from dmgseg import hub
from dmgseg.data.cvat import CLASS_NAMES, PAPER_KINDS
from dmgseg.eval.metrics import summary
from dmgseg.prior.infer import evaluate_full_images
from dmgseg.prior.train import split_annotations
from dmgseg.prior.unet_dinov2 import load_prior_model

FIXED_KINDS = ("polygon", "box", "mask")


def export_final_weights(run_dir, push=True):
    """Save the model weights from last.pt as final.pt (same format as best.pt)."""
    run_dir = Path(run_dir)
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    torch.save(state["model"], run_dir / "final.pt")
    if push:
        hub.upload(run_dir / "final.pt", f"runs/{run_dir.name}/final.pt")
    return run_dir / "final.pt", state["epoch"] + 1


DEFAULT_EVAL = {"patch_size": 518, "stride": 300, "model_input": 518}


def checkpoint_spec(cfg, weights):
    """{weights, img_size, eval settings} for a trained run: evaluate at its training scale."""
    ev = dict(cfg.get("eval", {}))
    ev.setdefault("model_input", cfg.data.model_input)
    return {"weights": weights, "img_size": cfg.model.get("img_size"), "eval": ev}


def compare_checkpoints(checkpoints, embed_model, device, out_path, eval_cfg=None):
    """checkpoints: {name: weights path, or a checkpoint_spec dict}. Evaluates each on
    the 44 validation images with paper labels and fixed labels; writes and returns
    a JSON-able dict."""
    _, val_anns = split_annotations()
    results = {}
    for name, spec in checkpoints.items():
        if not isinstance(spec, dict):
            spec = {"weights": spec, "img_size": None, "eval": eval_cfg or DEFAULT_EVAL}
        model = load_prior_model(spec["weights"], device, img_size=spec["img_size"])
        ev = {**DEFAULT_EVAL, **spec["eval"]}
        results[name] = {"eval": ev}
        for labels, kinds in (("paper_labels", PAPER_KINDS), ("fixed_labels", FIXED_KINDS)):
            m = evaluate_full_images(model, embed_model, val_anns, kinds, device=device, **ev)
            results[name][labels] = summary(m.compute(), CLASS_NAMES)
        del model
        torch.cuda.empty_cache()
    Path(out_path).write_text(json.dumps(results, indent=1))
    return results


def results_table(results, labels="fixed_labels"):
    """Markdown table: one row per checkpoint, mean and per-class global IoU."""
    head = "| model | global mIoU | global mF1 | paper mIoU | " + " | ".join(CLASS_NAMES) + " |"
    rows = [head, "|" + "---|" * (4 + len(CLASS_NAMES))]
    for name, r in results.items():
        v = r[labels]
        if "eval" in r:
            name = f"{name} ({r['eval']['patch_size']}→{r['eval']['model_input']})"
        per_class = " | ".join(f"{v['global/iou'][c]:.3f}" for c in CLASS_NAMES)
        rows.append(f"| {name} | {v['global/miou']:.4f} | {v['global/mf1']:.4f} | "
                    f"{v['paper/miou']:.4f} | {per_class} |")
    return "\n".join(rows)
