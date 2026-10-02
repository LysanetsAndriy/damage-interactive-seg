"""SAM fine-tuning job for molab (notebook section I).

Train on the 246 paper-training images minus fold 0 (the dev set for early
stopping); then evaluate zero-shot vs fine-tuned SAM on the 44 validation images
with the simulated user (NoC@85/90, IoU@k). Uploads sam/finetuned_decoder.pt and
sam/finetune_results.json.
"""
import json
from pathlib import Path

import torch

from dmgseg import hub
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.data.split import load_kfold, load_split
from dmgseg.eval.run_interactive import evaluate
from dmgseg.sam.finetune import SamData, finetune, new_predictor, trainable_state
from dmgseg.sam.predictor import SamClicker


def sam_ft_job(workdir, device, status=None, max_images=None, epochs=12):
    status = status or (lambda **_: None)
    out = Path(workdir) / "sam_ft"
    out.mkdir(parents=True, exist_ok=True)
    try:
        split = load_split()
        dev_names = load_kfold()[0]["heldout"]
        train_names = [n for n in split["train"] if n not in set(dev_names)]
        val_names = split["val"]
        if max_images:
            train_names, dev_names, val_names = train_names[:max_images], dev_names[:max_images], val_names[:max_images]
        predictor = new_predictor("small", device)
        status(state="features", stage=f"{len(train_names)} train + {len(dev_names)} dev images")
        train_data = SamData(predictor, train_names, device)
        dev_data = SamData(predictor, dev_names, device)
        status(state="training", stage=f"{len(train_data)} train / {len(dev_data)} dev objects")
        info = finetune(predictor, train_data, dev_data, epochs=epochs,
                        log=lambda m: (print(m), status(stage=m)))
        del train_data, dev_data
        torch.cuda.empty_cache()
        torch.save(trainable_state(predictor.model), out / "finetuned_decoder.pt")

        status(state="evaluating", stage="zero-shot vs fine-tuned on validation")
        results = {"train_info": info}
        for name, clicker in (("zero-shot", SamClicker("small", device)), ("fine-tuned", SamClicker(predictor=predictor))):
            summary, _ = evaluate(clicker, val_names, max_clicks=20)
            results[name] = summary
        (out / "finetune_results.json").write_text(json.dumps(results, indent=1))
        if not max_images:
            hub.upload(out / "finetuned_decoder.pt", "sam/finetuned_decoder.pt", message="SAM 2.1-S fine-tuned decoder")
            hub.upload(out / "finetune_results.json", "sam/finetune_results.json", message="SAM fine-tuning results")
        status(state="done", table=table(results))
        return results
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise


def table(results):
    rows = ["| SAM | objects | IoU@1 | IoU@3 | IoU@5 | NoC@85 | NoC@90 | fail@90 |", "|---|---|---|---|---|---|---|---|"]
    for name in ("zero-shot", "fine-tuned"):
        a = results[name]["all"]
        m = a["miou@k"]
        rows.append(f"| {name} | {a['n']} | {m[0]:.3f} | {m[2]:.3f} | {m[4]:.3f} | {a['noc@85']:.2f} | "
                    f"{a['noc@90']:.2f} | {a['fail@90']:.1%} |")
    rows.append("\nPer class NoC@85 (zero-shot → fine-tuned): " + ", ".join(
        f"{c} {results['zero-shot'][c]['noc@85']:.1f}→{results['fine-tuned'][c]['noc@85']:.1f}"
        for c in CLASS_NAMES if results["zero-shot"].get(c)))
    return "\n".join(rows)
