"""Export the app's five networks to ONNX (runs on ONNX Runtime, no PyTorch needed).

    python -m dmgseg.onnx.export            # -> artifacts/onnx/*.onnx + meta.json

prior.onnx          DINOv2-Emb U-Net (B2b): crop (B,3,518,518), global embedding
                    (B,1536), position (B,2) -> logits (B,6,518,518)
embed.onnx          ConvNeXt-L CLIP: image (1,3,384,384) -> embedding (1,1536)
sam_encoder.onnx    SAM 2.1-S image encoder: image (1,3,1024,1024) -> image_embed
                    (1,256,64,64), high_res_0 (1,32,256,256), high_res_1 (1,64,128,128)
sam_decoder.onnx    prompt encoder + our fine-tuned mask decoder: features, points
                    (1,N,2) in the 1024 frame with labels (1,N) (1/0 clicks, 2/3 box
                    corners, -1 padding), previous low-res mask (1,1,256,256) and
                    has_mask (1) -> all 4 mask logits (1,4,256,256) + IoU (1,4);
                    the choice among them is done in numpy (runtime.py), exactly as
                    SAM 2 does it
head_a.onnx         class head A: cards (N,288) -> class logits (N,5), quality (N,)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from dmgseg import paths

OUT = paths.ARTIFACTS / "onnx"
OPSET = 17


class PriorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, emb, pos):
        return self.model(x, emb, pos)


class SamEncoder(nn.Module):
    """As SAM2ImagePredictor.set_image, from the normalized 1024 px input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        backbone_out = m.forward_image(x)
        _, vision_feats, _, _ = m._prepare_backbone_features(backbone_out)
        if m.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + m.no_mem_embed
        sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(vision_feats, sizes)]
        return feats[2], feats[0], feats[1]


class SamDecoder(nn.Module):
    """Prompt encoder + mask decoder, all 4 output tokens (selection outside)."""

    def __init__(self, model):
        super().__init__()
        self.pe, self.dec = model.sam_prompt_encoder, model.sam_mask_decoder

    def forward(self, image_embed, high_res_0, high_res_1, point_coords, point_labels, mask_input, has_mask):
        sparse = self.pe._embed_points(point_coords, point_labels, pad=False)
        no_mask = self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, 64, 64)
        dense = has_mask.reshape(1, 1, 1, 1) * self.pe._embed_masks(mask_input) \
            + (1 - has_mask.reshape(1, 1, 1, 1)) * no_mask
        masks, iou, _, _ = self.dec.predict_masks(
            image_embeddings=image_embed, image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            repeat_image=False, high_res_features=[high_res_0, high_res_1])
        return masks, iou


def _export(module, args, path, names_in, names_out, dynamic=None):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=names_in, output_names=names_out,
                          opset_version=OPSET, dynamic_axes=dynamic or {}, do_constant_folding=True)
    print(f"  {path.name}: {path.stat().st_size / 2**20:.0f} MB")


def export_all(out=OUT, sam_weights=None, head_weights=None):
    from dmgseg import hub
    from dmgseg.classhead.features import CARD_SIZE
    from dmgseg.classhead.mlp import CardHead
    from dmgseg.prior.embedding import load_embedding_model
    from dmgseg.prior.unet_dinov2 import load_prior_model
    from dmgseg.sam.predictor import SamClicker

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("exporting to", out)
    prior = load_prior_model(hub.prior_weights(), "cpu")
    _export(PriorWrapper(prior), (torch.randn(1, 3, 518, 518), torch.randn(1, 1536), torch.rand(1, 2)),
            out / "prior.onnx", ["x", "emb", "pos"], ["logits"],
            {"x": {0: "b"}, "emb": {0: "b"}, "pos": {0: "b"}, "logits": {0: "b"}})
    del prior
    _export(load_embedding_model("cpu"), (torch.randn(1, 3, 384, 384),), out / "embed.onnx", ["image"], ["embedding"])

    sam_weights = sam_weights or paths.ARTIFACTS / "sam" / "finetuned_decoder.pt"
    clicker = SamClicker("small", "cpu", decoder_weights=sam_weights)
    model = clicker.predictor.model.eval()
    _export(SamEncoder(model), (torch.randn(1, 3, 1024, 1024),), out / "sam_encoder.onnx", ["image"],
            ["image_embed", "high_res_0", "high_res_1"])
    feats = SamEncoder(model)(torch.randn(1, 3, 1024, 1024))
    _export(SamDecoder(model),
            (*[f.detach() for f in feats], torch.rand(1, 3, 2) * 1024, torch.tensor([[1., 0., -1.]]),
             torch.zeros(1, 1, 256, 256), torch.zeros(1)),
            out / "sam_decoder.onnx",
            ["image_embed", "high_res_0", "high_res_1", "point_coords", "point_labels", "mask_input", "has_mask"],
            ["masks", "iou"], {"point_coords": {1: "n"}, "point_labels": {1: "n"}})

    head_weights = head_weights or paths.ARTIFACTS / "classhead" / "head_a_samft.pt"
    head = CardHead(CARD_SIZE)
    head.load_state_dict(torch.load(head_weights, map_location="cpu"))
    _export(head, (torch.randn(4, CARD_SIZE),), out / "head_a.onnx", ["cards"], ["class_logits", "quality"],
            {"cards": {0: "n"}, "class_logits": {0: "n"}, "quality": {0: "n"}})

    dec = model.sam_mask_decoder
    meta = {"sam": {"stability": bool(dec.dynamic_multimask_via_stability),
                    "stability_delta": float(dec.dynamic_multimask_stability_delta),
                    "stability_thresh": float(dec.dynamic_multimask_stability_thresh),
                    "mask_threshold": float(clicker.predictor.mask_threshold), "resolution": 1024},
            "prior": {"patch_size": 518, "stride": 300, "model_input": 518},
            "embed": {"size": 384}, "opset": OPSET}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("meta:", meta["sam"])


if __name__ == "__main__":
    export_all()
