"""Global-context embedding (ConvNeXt-L CLIP, 1536-d), as in the paper."""
import timm
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

EMBEDDING_MODEL = "convnext_large_mlp.clip_laion2b_augreg_ft_in12k_384"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

full_image_transform = T.Compose([
    T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_embedding_model(device="cpu"):
    model = timm.create_model(EMBEDDING_MODEL, pretrained=True, num_classes=0)
    return model.to(device).eval()


@torch.no_grad()
def image_embedding(model, image, device="cpu"):
    """PIL image -> numpy [1536]."""
    x = full_image_transform(image).unsqueeze(0).to(device)
    return model(x).squeeze(0).float().cpu().numpy()
