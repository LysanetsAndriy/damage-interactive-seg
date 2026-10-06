"""ONNX vs PyTorch: same outputs? how fast on this Mac's CPU?

1. identical inputs -> largest output difference per network
2. end to end on validation photos (each side with its own preprocessing, as the
   apps would run): prior class agreement, SAM mask IoU for clicks / boxes /
   refinements, head A class agreement
"""
import time

import numpy as np
import torch
from PIL import Image

from dmgseg import hub, paths
from dmgseg.classhead.features import CARD_SIZE, card
from dmgseg.classhead.mlp import CardHead, predict
from dmgseg.data.split import load_split
from dmgseg.onnx.export import SamDecoder, SamEncoder
from dmgseg.onnx.runtime import OnnxHead, OnnxPrior, OnnxSam, session
from dmgseg.prior.embedding import image_embedding, load_embedding_model
from dmgseg.prior.infer import predict_probs
from dmgseg.prior.unet_dinov2 import load_prior_model
from dmgseg.sam.predictor import SamClicker

O = paths.ARTIFACTS / "onnx"
rng = np.random.default_rng(0)


def timed(f, n=3):
    f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


print("== 1. identical inputs ==")
prior = load_prior_model(hub.prior_weights(), "cpu")
x, e, p = torch.randn(2, 3, 518, 518), torch.randn(2, 1536), torch.rand(2, 2)
with torch.no_grad():
    ref = prior(x, e, p).numpy()
sp = session(O / "prior.onnx")
got = sp.run(None, {"x": x.numpy(), "emb": e.numpy(), "pos": p.numpy()})[0]
print(f"prior logits: max |diff| {np.abs(ref - got).max():.2e} (logit range {ref.min():.1f}..{ref.max():.1f})")
with torch.no_grad():
    t_torch = timed(lambda: prior(x[:1], e[:1], p[:1]), 2)
t_ort = timed(lambda: sp.run(None, {"x": x[:1].numpy(), "emb": e[:1].numpy(), "pos": p[:1].numpy()}), 2)
print(f"   one 518 crop: torch {t_torch:.2f} s, onnxruntime {t_ort:.2f} s")

emb_model = load_embedding_model("cpu")
xi = torch.randn(1, 3, 384, 384)
with torch.no_grad():
    r = emb_model(xi).numpy()
g = session(O / "embed.onnx").run(None, {"image": xi.numpy()})[0]
print(f"embedding: cosine {float((r * g).sum() / np.linalg.norm(r) / np.linalg.norm(g)):.6f}")

clicker = SamClicker("small", "cpu", decoder_weights=paths.ARTIFACTS / "sam" / "finetuned_decoder.pt")
model = clicker.predictor.model.eval()
xs = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    rf = SamEncoder(model)(xs)
se = session(O / "sam_encoder.onnx")
gf = se.run(None, {"image": xs.numpy()})
print("SAM encoder: max |diff|", [f"{np.abs(a.numpy() - b).max():.1e}" for a, b in zip(rf, gf)])
with torch.no_grad():
    t_torch = timed(lambda: SamEncoder(model)(xs), 2)
t_ort = timed(lambda: se.run(None, {"image": xs.numpy()}), 2)
print(f"   encoder 1024: torch {t_torch:.2f} s, onnxruntime {t_ort:.2f} s")

print("\n== 2. end to end on validation photos ==")
osam, oprior, ohead = OnnxSam(), OnnxPrior(), OnnxHead()
head = CardHead(CARD_SIZE)
head.load_state_dict(torch.load(paths.ARTIFACTS / "classhead" / "head_a_samft.pt"))
names = load_split()["val"][:3]
agree_prior, ious, cls_agree = [], {"click": [], "box": [], "refine": []}, []
for n in names:
    img = Image.open(paths.IMAGES_DIR / n).convert("RGB")
    rgb = np.asarray(img)
    t0 = time.perf_counter()
    pt, _ = predict_probs(prior, emb_model, img, batch_size=2)
    t1 = time.perf_counter()
    po = oprior.predict_probs(rgb)
    t2 = time.perf_counter()
    from dmgseg.tool.assign import fit_prior
    pt, po = fit_prior(pt, *rgb.shape[:2]), fit_prior(po, *rgb.shape[:2])
    agree_prior.append((pt.argmax(-1) == po.argmax(-1)).mean())
    print(f"{n} {rgb.shape[1]}x{rgb.shape[0]}: prior torch {t1 - t0:.0f} s, onnx {t2 - t1:.0f} s, "
          f"same class {agree_prior[-1]:.2%}, max prob diff {np.abs(pt - po).max():.3f}")
    clicker.set_image(rgb)
    osam.set_image(rgb)
    h, w = rgb.shape[:2]
    for _ in range(8):
        x0, y0 = int(rng.integers(w)), int(rng.integers(h))
        clicker.reset_object()
        mt = clicker.click(x0, y0)
        ct, st = clicker.candidates
        mo, so, lo = osam.predict([(x0, y0)], [1], multimask=True)
        k = int(np.argmax(so))
        ious["click"].append(float((mt & mo[k]).sum() / max((mt | mo[k]).sum(), 1)))
        # refinement with the previous mask fed back (single mask)
        x1, y1 = int(rng.integers(w)), int(rng.integers(h))
        mt2 = clicker.click(x1, y1, False)
        mo2, _, _ = osam.predict([(x0, y0), (x1, y1)], [1, 0], mask_input=lo[k], multimask=False)
        ious["refine"].append(float((mt2 & mo2[0]).sum() / max((mt2 | mo2[0]).sum(), 1)))
        bx = (max(0, x0 - 60), max(0, y0 - 60), min(w, x0 + 60), min(h, y0 + 60))
        mt3 = clicker.box_click(bx, x0, y0)
        mo3, _, _ = osam.predict([bx[:2], bx[2:], (x0, y0)], [2, 3, 1], multimask=False)
        ious["box"].append(float((mt3 & mo3[0]).sum() / max((mt3 | mo3[0]).sum(), 1)))
        cards = [card(m, pt, clicker.image_embedding, float(s), j, 1) for j, (m, s) in enumerate(zip(ct, st))]
        a, _ = predict(head, np.stack(cards))
        b, _ = ohead.score(cards)
        cls_agree.append(bool(a[0].argmax() == b[0].argmax()) and float(np.abs(a - b).max()) < 1e-4)
print({k: f"mean mask IoU torch vs onnx {np.mean(v):.4f} (min {np.min(v):.3f})" for k, v in ious.items()})
print(f"head A: same class and probs within 1e-4 in {np.mean(cls_agree):.0%} of cases")
t_t = timed(lambda: (clicker.reset_object(), clicker.click(500, 300)), 5)
t_o = timed(lambda: osam.predict([(500, 300)], [1]), 5)
print(f"SAM click (decoder + full-size masks): torch {1000 * t_t:.0f} ms, onnx {1000 * t_o:.0f} ms")
