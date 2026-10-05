"""SAM 2.1 image-encoder adaptation with LoRA (on top of the fine-tuned decoder).

The decoder fine-tuning (finetune.py) kept the image encoder frozen: SAM "sees" our
images exactly as before. Here low-rank adapters (LoRA, rank r) are added to the
attention projections (qkv) of every Hiera block, so the encoder's features can
adapt to damaged facades; the decoder keeps training at a lower learning rate.
The training loop is the same simulated-click protocol, but the encoder runs live
(4 images per step, up to 8 objects each).

The adapters are merged into the qkv weights at the end, so the app loads them
like the decoder weights (load_finetuned, strict=False).
"""
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dmgseg import paths
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.eval.run_interactive import MIN_AREA
from dmgseg.sam.finetune import LOW, SamData, dev_score, interactive_step, load_finetuned


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # starts as the identity change
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A).to(self.base.weight.dtype)


def add_lora(model, r=8, alpha=16):
    """Wrap every Hiera attention qkv projection. -> list of the LoRA modules."""
    mods = []
    for blk in model.image_encoder.trunk.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
        mods.append(blk.attn.qkv)
    return mods


def merged_state(model):
    """Trainable parts for the app: decoder + prompt encoder, and every block's qkv
    weight with its adapter merged in (keys as in the plain SAM 2 model)."""
    out = {k: v.detach().clone() for k, v in model.state_dict().items()
           if k.startswith(("sam_mask_decoder.", "sam_prompt_encoder."))}
    for i, blk in enumerate(model.image_encoder.trunk.blocks):
        q = blk.attn.qkv
        if isinstance(q, LoRALinear):
            out[f"image_encoder.trunk.blocks.{i}.attn.qkv.weight"] = q.merged_weight().detach().clone()
            out[f"image_encoder.trunk.blocks.{i}.attn.qkv.bias"] = q.base.bias.detach().clone()
    return out


class LiveImages:
    """Preprocessed 1024 px inputs (uint8 on the device) + 256x256 object masks."""

    def __init__(self, predictor, names, device, min_area=MIN_AREA):
        anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
        self.inputs, self.masks, self.device = [], [], device
        for name in names:
            image = np.asarray(Image.open(paths.IMAGES_DIR / name).convert("RGB"))
            x = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)       # SAM 2 squashes to a square
            self.inputs.append(torch.from_numpy(x).permute(2, 0, 1).to(device))
            ms = []
            for o in image_objects(anns[name], min_area):
                m = cv2.resize(o.mask.astype(np.float32), (LOW, LOW), interpolation=cv2.INTER_AREA) > 0.5
                if m.sum() >= 4:
                    ms.append(m)
            self.masks.append(torch.from_numpy(np.stack(ms)).to(device) if ms else None)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.norm = lambda u8: (u8.float() / 255 - mean) / std


def encode(model, x):
    """Image features with gradients (as SAM2ImagePredictor.set_image, batched)."""
    backbone_out = model.forward_image(x)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    sizes = [(256, 256), (128, 128), (64, 64)]
    B = x.shape[0]
    feats = [f.permute(1, 2, 0).reshape(B, -1, *s) for f, s in zip(vision_feats, sizes)]
    return feats[2], [feats[0], feats[1]]


class _Batch:
    """The interface interactive_step expects (masks, features(idx)) for objects of
    a few images whose features were just computed with gradients."""

    def __init__(self, embed, hr, masks, img_of):
        self.embed, self.hr, self.masks, self.img_of = embed, hr, masks, img_of

    def features(self, idx):
        im = self.img_of[idx]
        return self.embed[im], [self.hr[0][im], self.hr[1][im]]


def lora_finetune(predictor, train, dev_names, epochs=10, lr_lora=1e-4, lr_dec=1e-5, images_per_step=4,
                  objects_per_image=8, r=8, seed=0, log=print):
    model = predictor.model
    for p in model.parameters():
        p.requires_grad = False
    loras = add_lora(model, r)
    model.to(predictor.device)
    lora_params = [p for m in loras for p in (m.A, m.B)]
    dec_params = [p for n, p in model.named_parameters()
                  if n.startswith(("sam_mask_decoder.", "sam_prompt_encoder."))]
    for p in lora_params + dec_params:
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr_lora}, {"params": dec_params, "lr": lr_dec}],
                            weight_decay=0.01)
    usable = [i for i, m in enumerate(train.masks) if m is not None]
    steps = epochs * ((len(usable) + images_per_step - 1) // images_per_step)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_lora, lr_dec], total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def dev():
        model.eval()
        with torch.no_grad():
            data = SamData(predictor, dev_names, predictor.device)
            return dev_score(model, data)

    base = dev()
    log(f"start (fine-tuned decoder) dev IoU after 1/2/3 clicks: {np.round(base, 4).tolist()}")
    best, best_state, bad, history = base[0] + base[-1], merged_state(model), 0, []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(usable)
        for s in range(0, len(order), images_per_step):
            ims = order[s:s + images_per_step]
            x = torch.stack([train.norm(train.inputs[i][None])[0] for i in ims])
            masks, img_of = [], []
            for j, i in enumerate(ims):
                m = train.masks[i]
                pick = rng.choice(len(m), min(objects_per_image, len(m)), replace=False)
                masks.append(m[torch.as_tensor(pick, device=m.device)])
                img_of += [j] * len(pick)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                embed, hr = encode(model, x)
            batch = _Batch(embed, hr, torch.cat(masks), torch.tensor(img_of, device=x.device))
            idx = torch.arange(len(batch.masks), device=x.device)
            loss, _ = interactive_step(model, batch, idx, rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + dec_params, 1.0)
            opt.step()
            sched.step()
        d = dev()
        history.append({"epoch": epoch, "dev_iou": d.tolist(), "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch + 1}/{epochs}: dev IoU after 1/2/3 clicks {np.round(d, 4).tolist()} ({time.time() - t0:.0f}s)")
        if d[0] + d[-1] > best + 1e-4:
            best, bad, best_state = d[0] + d[-1], 0, merged_state(model)
        else:
            bad += 1
            if bad >= 3:
                break
    return best_state, {"start_dev": base.tolist(), "history": history}


def sam_lora_job(workdir, device, status=None, max_images=None, epochs=10):
    """molab job: LoRA on top of the fine-tuned decoder; then the fine-tuned-decoder
    SAM vs the LoRA SAM on the 44 validation images (simulated user, 20 clicks)."""
    import json
    from pathlib import Path

    from dmgseg import hub
    from dmgseg.data.split import load_kfold, load_split
    from dmgseg.eval.run_interactive import evaluate
    from dmgseg.sam.finetune import new_predictor
    from dmgseg.sam.predictor import SamClicker
    from dmgseg.sam.run_ft import table

    status = status or (lambda **_: None)
    out = Path(workdir) / "sam_lora"
    out.mkdir(parents=True, exist_ok=True)
    try:
        split = load_split()
        dev_names = load_kfold()[0]["heldout"]
        train_names = [n for n in split["train"] if n not in set(dev_names)]
        val_names = split["val"]
        if max_images:
            train_names, dev_names, val_names = train_names[:max_images], dev_names[:max_images], val_names[:max_images]
        dec = hub.download_if_exists("sam/finetuned_decoder.pt", workdir)
        predictor = new_predictor("small", device)
        load_finetuned(predictor.model, dec, device)
        status(state="loading", stage=f"{len(train_names)} train images")
        train = LiveImages(predictor, train_names, device)
        status(state="training", stage="LoRA")
        state, info = lora_finetune(predictor, train, dev_names, epochs=epochs, log=lambda m: status(stage=m))
        torch.save(state, out / "lora_merged.pt")

        status(state="evaluating", stage="fine-tuned decoder vs + encoder LoRA")
        results = {"train_info": info}
        base = SamClicker("small", device, decoder_weights=dec)
        lora = SamClicker("small", device, decoder_weights=out / "lora_merged.pt")
        for name, clicker in (("zero-shot", base), ("fine-tuned", lora)):   # names used by run_ft.table
            summary, _ = evaluate(clicker, val_names, max_clicks=20)
            results[name] = summary
        (out / "lora_results.json").write_text(json.dumps(results, indent=1))
        if not max_images:
            hub.safe_upload(out / "lora_merged.pt", "sam/lora_merged.pt", retries=6)
            hub.safe_upload(out / "lora_results.json", "sam/lora_results.json", retries=6)
        status(state="done", table=table(results).replace("zero-shot", "decoder-FT").replace("fine-tuned", "+ encoder LoRA"))
        return results
    except Exception:
        import traceback
        status(state="error", error=traceback.format_exc()[-3000:])
        raise
