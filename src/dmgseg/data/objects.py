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


def _duplicate_shapes(ann):
    """Indices of shapes whose exact geometry is drawn again with a higher-priority
    label (the dataset has 24 Roof polygons duplicated as Damaged roof, 3 Building
    as Damage). Only the higher-priority copy is visible in the semantic mask, so
    only it is kept as an object."""
    best = {}
    for i, s in enumerate(ann.shapes):
        if s.points is None:
            continue
        key = (s.kind, tuple((round(x, 1), round(y, 1)) for x, y in s.points))
        if key not in best or s.label > ann.shapes[best[key]].label:
            best[key] = i
    keep = set(best.values())
    return {i for i, s in enumerate(ann.shapes) if s.points is not None and i not in keep}


def image_objects(ann, min_area=0, kinds=OBJECT_KINDS):
    """All objects of one image, in annotation order, skipping ones below min_area
    and duplicated geometry (see _duplicate_shapes)."""
    out = []
    dropped = _duplicate_shapes(ann)
    for i, shape in enumerate(ann.shapes):
        if shape.kind not in kinds or shape.kind not in ALL_KINDS or i in dropped:
            continue
        m = shape.render(ann.height, ann.width)
        if m.sum() >= max(min_area, 1):
            out.append(GTObject(ann.name, i, shape.label, shape.kind, m))
    return out
