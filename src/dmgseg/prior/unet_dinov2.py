"""DINOv2-Emb U-Net from the paper, copied from the training notebook.

The only change: `pretrained` is a constructor argument, so loading our own
checkpoint doesn't first download the 1.2 GB timm weights.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["Other", "Building", "Roof", "Damage", "Broken Window", "Damaged roof"]
NUM_CLASSES = len(CLASS_NAMES)

# Global-context embedding model used by the paper (1536-d output).
EMBEDDING_MODEL = "convnext_large_mlp.clip_laion2b_augreg_ft_in12k_384"


class UNetDinoV2(nn.Module):
    """U-Net that uses DINOv2 (ViT-L/14) as the encoder backbone."""

    def __init__(self, num_classes=NUM_CLASSES,
                 dino_model="vit_large_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=None):
        """img_size: model input side (a multiple of 14); None = DINOv2's 518. The
        position embeddings are resampled to the new grid when the model is built."""
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size or 518

        extra = {"img_size": img_size} if img_size else {}
        self.encoder = timm.create_model(
            dino_model,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 7, 12, 20),
            **extra,
        )
        c1, c2, c3, c4 = self.encoder.feature_info.channels()

        # Global and positional embeddings, concatenated at the bottleneck.
        self.image_emb_fc = nn.Linear(1536, c4 // 2)
        self.position_emb_fc = nn.Linear(2, c4 // 4)

        self.bottleneck = self._conv_block(c4 + (c4 // 2) + (c4 // 4), c3)
        self.decoder4 = self._conv_block(c3 + c3, c2)
        self.decoder3 = self._conv_block(c2 + c2, c1)
        self.decoder2 = self._conv_block(c1 + c1, c1 // 2)
        self.decoder1 = self._conv_block(c1 // 2, c1 // 4)
        self.conv_last = nn.Conv2d(c1 // 4, num_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.6),
        )

    def forward(self, x, image_embedding, position_embedding):
        """
        Args:
            x: [B, 3, 518, 518] ImageNet-normalized patch.
            image_embedding: [B, 1536] ConvNeXt-L CLIP embedding of the full image.
            position_embedding: [B, 2] normalized (x, y) of the patch's top-left corner.
        Returns:
            [B, num_classes, 518, 518] logits.
        """
        enc1, enc2, enc3, enc4 = self.encoder(x)

        B, _, H4, W4 = enc4.shape
        img_e = self.image_emb_fc(image_embedding).view(B, -1, 1, 1).expand(-1, -1, H4, W4)
        pos_e = self.position_emb_fc(position_embedding).view(B, -1, 1, 1).expand(-1, -1, H4, W4)
        bott = self.bottleneck(torch.cat([enc4, img_e, pos_e], dim=1))

        d4 = F.interpolate(bott, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.decoder4(torch.cat([d4, enc3], dim=1))

        d3 = F.interpolate(d4, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d3, enc2], dim=1))

        d2 = F.interpolate(d3, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2, enc1], dim=1))

        d1 = F.interpolate(d2, scale_factor=4, mode="bilinear", align_corners=False)
        d1 = self.decoder1(d1)
        out = F.interpolate(d1, scale_factor=3.5, mode="bilinear", align_corners=False)

        return self.conv_last(out)


def load_prior_model(weights_path, device="cpu", img_size=None):
    model = UNetDinoV2(pretrained=False, img_size=img_size)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
