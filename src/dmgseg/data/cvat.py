"""Parser for the CVAT 1.1 XML export of the damage dataset.

Unlike the paper notebook, which read only <polygon> and <box>, this reads all four
shape types CVAT produced: polygon, box, mask (brush, RLE) and polyline.

Polylines (187, almost all Roof) are thin roof-edge lines, not outlines, so they
are ignored by default, as in the paper.

Semantic masks are composited in class-priority order, exactly like the notebook:
Other < Building < Roof < Damage < Broken Window < Damaged roof.
"""
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["Other", "Building", "Roof", "Damage", "Broken Window", "Damaged roof"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ALL_KINDS = ("polygon", "box", "mask", "polyline")
PAPER_KINDS = ("polygon", "box")  # what the paper's training pipeline used


@dataclass
class Shape:
    label: int
    kind: str                 # polygon | box | mask | polyline
    points: list = None       # [(x, y), ...] for polygon / box / polyline
    rle: list = None          # mask only: run lengths inside the bbox, background first
    bbox: tuple = None        # mask only: (left, top, width, height)
    z_order: int = 0

    def render(self, height, width, polyline_mode="ignore"):
        """Full-image boolean mask of this shape.

        polyline_mode: "ignore" (default) returns an empty mask; "close" joins the
        last point to the first. The dataset's polylines are open roof-edge lines
        (checked visually), so closing them paints sky or wall as Roof.
        """
        if self.kind == "mask":
            return decode_cvat_rle(self.rle, self.bbox, height, width)
        if self.kind == "polyline" and (polyline_mode == "ignore" or len(self.points) < 3):
            return np.zeros((height, width), bool)
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(self.points, fill=1)
        return np.asarray(canvas, dtype=bool)


@dataclass
class ImageAnn:
    name: str
    width: int
    height: int
    shapes: list = field(default_factory=list)

    def shapes_of(self, kinds=ALL_KINDS):
        return [s for s in self.shapes if s.kind in kinds]


def _points(s):
    return [tuple(map(float, p.split(","))) for p in s.split(";")]


def decode_cvat_rle(rle, bbox, height, width):
    """Decode a CVAT brush mask. Runs alternate background/foreground, starting with
    background, over the bbox in row-major order."""
    left, top, w, h = bbox
    flat = np.zeros(w * h, dtype=bool)
    pos, value = 0, False
    for run in rle:
        if value:
            flat[pos:pos + run] = True
        pos += run
        value = not value
    full = np.zeros((height, width), dtype=bool)
    sub = flat.reshape(h, w)
    full[top:top + h, left:left + w] = sub[:height - top, :width - left]
    return full


def parse_annotations(xml_path):
    """Return a list of ImageAnn in XML order (the order the paper split depends on)."""
    root = ET.parse(xml_path).getroot()
    images = []
    for el in root.findall("image"):
        ann = ImageAnn(el.get("name"), int(el.get("width")), int(el.get("height")))
        for sh in el:
            label = sh.get("label")
            if label is None or sh.tag not in ALL_KINDS:
                continue
            common = dict(label=LABEL_TO_ID[label], kind=sh.tag, z_order=int(sh.get("z_order", 0)))
            if sh.tag in ("polygon", "polyline"):
                ann.shapes.append(Shape(points=_points(sh.get("points")), **common))
            elif sh.tag == "box":
                x1, y1, x2, y2 = (float(sh.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                ann.shapes.append(Shape(points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], **common))
            else:  # mask
                rle = [int(v) for v in sh.get("rle").replace(" ", "").split(",") if v]
                bbox = tuple(int(float(sh.get(k))) for k in ("left", "top", "width", "height"))
                ann.shapes.append(Shape(rle=rle, bbox=bbox, **common))
        images.append(ann)
    return images


def semantic_mask(ann, kinds=ALL_KINDS, polyline_mode="ignore"):
    """Integer class map (H, W), shapes painted in class-priority order.

    kinds=PAPER_KINDS reproduces the labels the paper was trained on.
    """
    canvas = Image.new("L", (ann.width, ann.height), 0)
    draw = ImageDraw.Draw(canvas)
    for cls in range(len(CLASS_NAMES)):
        raster = [s for s in ann.shapes_of(kinds) if s.label == cls]
        polys = [s for s in raster if s.kind in ("polygon", "box")
                 or (s.kind == "polyline" and polyline_mode == "close" and len(s.points) >= 3)]
        for s in polys:
            draw.polygon(s.points, fill=cls)
        masks = [s for s in raster if s.kind == "mask"]
        if masks:
            arr = np.array(canvas)
            for s in masks:
                arr[s.render(ann.height, ann.width)] = cls
            canvas = Image.fromarray(arr)
            draw = ImageDraw.Draw(canvas)
    return np.asarray(canvas, dtype=np.uint8)
