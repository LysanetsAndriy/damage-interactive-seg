"""DINOv3 U-Nets for the semantic prior (experiments V3-640, V3-640-stem, V3-CNX).

All take the same inputs as the paper's model (patch, 1536-d global image
embedding, patch position) and keep its bottleneck idea (global + position
embeddings concatenated to the deepest features). Inputs are 640 px crops at full
resolution (DINOv3: patch 16, RoPE, trained up to 768 px), no downscaling to 518.

arch="vit"       V3-640: DINOv3 ViT-L/16 in the paper's decoder. All four taps
                 come from one ViT, so they share one resolution (1/16): the
                 decoder has no high-resolution path (as in the paper's model).
arch="vit_stem"  V3-640-stem: the ViT gives the semantics at 1/16 (four taps fused
                 at the bottleneck); a small CNN stem on the RGB crop gives real
                 skip connections at 1/8, 1/4 and 1/2, so each decoder step up gets
                 fresh image detail (edges, thin roof strips, window frames).
arch="convnext"  V3-CNX: DINOv3 ConvNeXt-L, a CNN distilled from the DINOv3 teacher,
                 with a natural pyramid (1/4, 1/8, 1/16, 1/32) -> a classic U-Net.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmgseg.prior.unet_dinov2 import NUM_CLASSES

EMB_DIM = 1536                    # ConvNeXt-L CLIP global embedding (as in the paper)

DEFAULTS = {
    "vit": dict(backbone="vit_large_patch16_dinov3.lvd1689m", taps=(1, 7, 12, 20)),
    "vit_stem": dict(backbone="vit_large_patch16_dinov3.lvd1689m", taps=(5, 11, 17, 23)),
    "convnext": dict(backbone="convnext_large.dinov3_lvd1689m", taps=(0, 1, 2, 3)),
}


def conv_block(in_ch, out_ch, dropout):
    """Two 3x3 convs (as in the paper's decoder: BN, GELU, dropout)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch // 2, 3, padding=1), nn.BatchNorm2d(out_ch // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Conv2d(out_ch // 2, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(), nn.Dropout(dropout),
    )


def stem_stage(in_ch, out_ch):
    """Halve the resolution: 3x3 stride-2 conv + 3x3 conv (BN, GELU)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
    )


def up(x, like):
    return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)


class UNetDinoV3(nn.Module):
    # dropout: the paper's 0.6 on the coarse levels; lighter on the fine levels,
    # where heavy dropout would wash out the detail the skips bring
    DROP_COARSE, DROP_FINE = 0.6, 0.1

    def __init__(self, arch="vit_stem", num_classes=NUM_CLASSES, backbone=None, taps=None,
                 pretrained=False, img_size=640):
        super().__init__()
        if arch not in DEFAULTS:
            raise ValueError(f"unknown arch {arch!r}")
        self.arch, self.num_classes, self.img_size = arch, num_classes, img_size
        backbone = backbone or DEFAULTS[arch]["backbone"]
        self.taps = tuple(taps or DEFAULTS[arch]["taps"])
        extra = {"img_size": img_size} if arch != "convnext" else {}
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                         out_indices=self.taps, **extra)
        chs = self.encoder.feature_info.channels()
        getattr(self, f"_build_{arch}")(chs)

    # -- builders ---------------------------------------------------------------
    def _build_vit(self, chs):
        """The paper's decoder unchanged (c1..c4 = 1024), only the final upsampling
        goes to the input size instead of the hard-coded x4 x3.5 of patch 14."""
        c1, c2, c3, c4 = chs
        self.image_emb_fc = nn.Linear(EMB_DIM, c4 // 2)
        self.position_emb_fc = nn.Linear(2, c4 // 4)
        d = self.DROP_COARSE
        self.bottleneck = conv_block(c4 + c4 // 2 + c4 // 4, c3, d)
        self.decoder4 = conv_block(c3 + c3, c2, d)
        self.decoder3 = conv_block(c2 + c2, c1, d)
        self.decoder2 = conv_block(c1 + c1, c1 // 2, d)
        self.decoder1 = conv_block(c1 // 2, c1 // 4, d)
        self.conv_last = nn.Conv2d(c1 // 4, self.num_classes, 1)

    def _build_vit_stem(self, chs):
        # four ViT taps, each projected to 256 channels and fused at 1/16
        self.tap_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, 256, 1), nn.BatchNorm2d(256), nn.GELU())
                                      for c in chs)
        self.image_emb_fc = nn.Linear(EMB_DIM, 512)
        self.position_emb_fc = nn.Linear(2, 256)
        self.bottleneck = conv_block(256 * len(chs) + 512 + 256, 512, self.DROP_COARSE)
        # CNN stem on the RGB crop: real skips at 1/2, 1/4, 1/8 (trained from scratch)
        self.stem1, self.stem2, self.stem3 = stem_stage(3, 64), stem_stage(64, 128), stem_stage(128, 256)
        self.dec3 = conv_block(512 + 256, 256, self.DROP_COARSE)      # 1/8
        self.dec2 = conv_block(256 + 128, 128, self.DROP_FINE)        # 1/4
        self.dec1 = conv_block(128 + 64, 64, self.DROP_FINE)          # 1/2
        self.dec0 = conv_block(64, 32, self.DROP_FINE)                # full resolution
        self.conv_last = nn.Conv2d(32, self.num_classes, 1)

    def _build_convnext(self, chs):
        c1, c2, c3, c4 = chs                      # 192, 384, 768, 1536 for ConvNeXt-L
        self.image_emb_fc = nn.Linear(EMB_DIM, c4 // 2)
        self.position_emb_fc = nn.Linear(2, c4 // 4)
        self.bottleneck = conv_block(c4 + c4 // 2 + c4 // 4, c3, self.DROP_COARSE)   # 1/32
        self.dec3 = conv_block(c3 + c3, c2, self.DROP_COARSE)         # 1/16
        self.dec2 = conv_block(c2 + c2, c1, self.DROP_COARSE)         # 1/8
        self.dec1 = conv_block(c1 + c1, c1 // 2, self.DROP_FINE)      # 1/4
        self.dec0 = conv_block(c1 // 2, c1 // 4, self.DROP_FINE)      # 1/2
        self.conv_last = nn.Conv2d(c1 // 4, self.num_classes, 1)

    # -- forward ------------------------------------------------------------------
    def _embeddings(self, feat, image_embedding, position_embedding):
        B, _, H, W = feat.shape
        img_e = self.image_emb_fc(image_embedding).view(B, -1, 1, 1).expand(-1, -1, H, W)
        pos_e = self.position_emb_fc(position_embedding).view(B, -1, 1, 1).expand(-1, -1, H, W)
        return img_e, pos_e

    def forward(self, x, image_embedding, position_embedding):
        """x [B, 3, S, S] ImageNet-normalized (S = 640), image_embedding [B, 1536],
        position_embedding [B, 2] -> logits [B, num_classes, S, S]."""
        return getattr(self, f"_forward_{self.arch}")(x, image_embedding, position_embedding)

    def _forward_vit(self, x, emb, pos):
        e1, e2, e3, e4 = self.encoder(x)
        bott = self.bottleneck(torch.cat([e4, *self._embeddings(e4, emb, pos)], 1))
        d4 = self.decoder4(torch.cat([up(bott, e3), e3], 1))
        d3 = self.decoder3(torch.cat([up(d4, e2), e2], 1))
        d2 = self.decoder2(torch.cat([up(d3, e1), e1], 1))
        d1 = self.decoder1(F.interpolate(d2, scale_factor=4, mode="bilinear", align_corners=False))
        return self.conv_last(F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False))

    def _forward_vit_stem(self, x, emb, pos):
        taps = self.encoder(x)
        f = torch.cat([p(t) for p, t in zip(self.tap_proj, taps)], 1)              # 1/16
        bott = self.bottleneck(torch.cat([f, *self._embeddings(f, emb, pos)], 1))
        s1 = self.stem1(x)                                                          # 1/2
        s2 = self.stem2(s1)                                                         # 1/4
        s3 = self.stem3(s2)                                                         # 1/8
        d3 = self.dec3(torch.cat([up(bott, s3), s3], 1))
        d2 = self.dec2(torch.cat([up(d3, s2), s2], 1))
        d1 = self.dec1(torch.cat([up(d2, s1), s1], 1))
        d0 = self.dec0(up(d1, x))
        return self.conv_last(d0)

    def _forward_convnext(self, x, emb, pos):
        e1, e2, e3, e4 = self.encoder(x)                                           # 1/4 .. 1/32
        bott = self.bottleneck(torch.cat([e4, *self._embeddings(e4, emb, pos)], 1))
        d3 = self.dec3(torch.cat([up(bott, e3), e3], 1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        d0 = self.dec0(F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False))
        return self.conv_last(up(d0, x))

    # -- layer-wise learning-rate decay ---------------------------------------------
    def layer_id(self, name):
        """Depth of a parameter for layer-wise LR decay: 0 = input end of the
        encoder ... n_layers() = everything new (decoder, stem, embeddings)."""
        if not name.startswith("encoder."):
            return self.n_layers()
        if self.arch == "convnext":
            # ConvNeXt-L depths (3, 3, 27, 3): stage 2 in groups of 3 -> 12 levels,
            # as in the usual ConvNeXt layer decay
            if "stages_" not in name and "stages." not in name:
                return 0                                   # stem
            s = int(name.split("stages_")[1].split(".")[0]) if "stages_" in name else \
                int(name.split("stages.")[1].split(".")[0])
            if s < 2:
                return s + 1
            if s == 3:
                return 12
            b = int(name.split("blocks.")[1].split(".")[0]) if "blocks." in name else 0
            return 3 + b // 3
        if "blocks." in name:
            return int(name.split("blocks.")[1].split(".")[0]) + 1
        return 0                                           # patch embedding, tokens, RoPE

    def n_layers(self):
        if self.arch == "convnext":
            return 13
        return 1 + max(int(n.split("blocks.")[1].split(".")[0])
                       for n, _ in self.encoder.named_parameters() if "blocks." in n) + 1
