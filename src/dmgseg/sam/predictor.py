"""Click-driven wrapper around SAM 2.1 (image mode).

Follows the SAM interactive protocol: the first click asks for 3 candidate masks
(multimask) and keeps one; later clicks ask for a single mask and feed the
previous low-res logits back as the mask prompt.
"""
import numpy as np
import torch

SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


class SamClicker:
    """choose: how to pick among the 3 first-click masks:
    "score" (SAM's predicted IoU, the realistic default), "oracle" (best IoU
    against `oracle_gt`, an upper bound for analysis only), or a callable
    choose(masks, scores, x, y) -> index (e.g. a prior-based rule).
    """

    def __init__(self, size="small", device="cpu", choose="score", predictor=None, decoder_weights=None):
        """decoder_weights: fine-tuned prompt encoder + mask decoder (sam/finetune.py)."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = predictor or SAM2ImagePredictor.from_pretrained(SAM2_MODELS[size], device=device)
        if decoder_weights is not None:
            from dmgseg.sam.finetune import load_finetuned
            load_finetuned(self.predictor.model, decoder_weights, device)
        self.choose = choose
        self.oracle_gt = None
        self.reset_object()

    def set_image(self, image_rgb):
        with torch.inference_mode():
            self.predictor.set_image(np.asarray(image_rgb))
        self.reset_object()

    @property
    def image_embedding(self):
        """SAM 2 image embedding of the current image: tensor (256, 64, 64)."""
        return self.predictor._features["image_embed"][0]

    def reset_object(self):
        self.points, self.labels, self.logits = [], [], None
        self.candidates = None  # first-click masks, scores

    def click(self, x, y, positive=True):
        self.points.append((x, y))
        self.labels.append(1 if positive else 0)
        first = self.logits is None
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(self.points, dtype=np.float32),
                point_labels=np.array(self.labels),
                mask_input=None if first else self.logits[None],
                multimask_output=first,
            )
        if first:
            self.first_logits = logits
            self.candidates = (masks.astype(bool), scores)
            k = self._pick(masks.astype(bool), scores)
        else:
            k = 0
        self.logits = logits[k]
        self.last_score = float(scores[k])
        return masks[k].astype(bool)

    def box_click(self, box, x, y):
        """Box (x0, y0, x1, y1) + one positive point -> single mask, as a new object.
        The box is kept as two prompt points (labels 2 and 3, SAM's box encoding), so
        later clicks refine it like any other object."""
        self.reset_object()
        x0, y0, x1, y1 = box
        self.points = [(x0, y0), (x1, y1), (x, y)]
        self.labels = [2, 3, 1]
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(self.points, dtype=np.float32), point_labels=np.array(self.labels),
                multimask_output=False)
        self.logits = logits[0]
        self.last_score = float(scores[0])
        return masks[0].astype(bool)

    def choose_index(self, k):
        """After the first click: continue from candidate k instead of the picked one."""
        masks, scores = self.candidates
        self.logits = self.first_logits[k]
        self.last_score = float(scores[k])
        return masks[k]

    def _pick(self, masks, scores):
        if callable(self.choose):
            x, y = self.points[0]
            return int(self.choose(masks, scores, x, y))
        if self.choose == "oracle" and self.oracle_gt is not None:
            gt = self.oracle_gt
            return int(np.argmax([(m & gt).sum() / max((m | gt).sum(), 1) for m in masks]))
        return int(np.argmax(scores))
