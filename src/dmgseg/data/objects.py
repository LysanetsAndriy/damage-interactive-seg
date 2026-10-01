"""Ground-truth objects for interactive segmentation: one CVAT shape = one object."""
from dataclasses import dataclass

import numpy as np

from dmgseg.data.cvat import ALL_KINDS

OBJECT_KINDS = ("polygon", "box", "mask")  # polylines are roof-edge lines, not regions


@dataclass
class GTObject:
    image: str
    index: int        # position of the shape in the image's annotation list
    label: int
    kind: str
    mask: np.ndarray  # bool (H, W), the full shape (not cut by higher-priority shapes)

    @property
    def area(self):
        return int(self.mask.sum())


def image_objects(ann, min_area=0, kinds=OBJECT_KINDS):
    """All objects of one image, in annotation order, skipping ones below min_area."""
    out = []
    for i, shape in enumerate(ann.shapes):
        if shape.kind not in kinds or shape.kind not in ALL_KINDS:
            continue
        m = shape.render(ann.height, ann.width)
        if m.sum() >= max(min_area, 1):
            out.append(GTObject(ann.name, i, shape.label, shape.kind, m))
    return out
