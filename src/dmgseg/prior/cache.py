"""Prior probabilities on disk: uint8 per class, compressed (no PyTorch needed)."""
import numpy as np


def save_prior(path, probs):
    q = np.clip(np.rint(probs * 255), 0, 255).astype(np.uint8).transpose(2, 0, 1)
    np.savez_compressed(path, probs=q)


def load_prior(path):
    """-> float32 (H, W, C) probabilities."""
    return np.load(path)["probs"].transpose(1, 2, 0).astype(np.float32) / 255.0
