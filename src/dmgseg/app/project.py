"""A folder of images annotated with the app, and its CVAT export.

Next to the images, `_damage_annotator/` keeps per image:
    <image>.json        saved objects (continue later)
    <image>.prior.npz   the cached DINOv2 prior (no 40 s wait next time)
"""
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np
from PIL import Image

from dmgseg.data.cvat import CLASS_NAMES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STORE = "_damage_annotator"
CVAT_COLORS = {"Other": "#d1d1d1", "Building": "#66ff66", "Roof": "#ffa500",
               "Damage": "#aa32c8", "Broken Window": "#1e5aff", "Damaged roof": "#e61e1e"}


class FolderProject:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.images = sorted(p.name for p in self.folder.iterdir()
                             if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
        self.store = self.folder / STORE
        self.store.mkdir(exist_ok=True)

    def path(self, name):
        return self.folder / name

    def state_path(self, name):
        return self.store / f"{name}.json"

    def prior_path(self, name):
        return self.store / f"{name}.prior.npz"

    def save_state(self, name, state):
        tmp = self.state_path(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self.state_path(name))

    def load_state(self, name):
        p = self.state_path(name)
        return json.loads(p.read_text()) if p.exists() else None

    def has_prior(self, name):
        return self.prior_path(name).exists()

    def save_prior(self, name, probs):
        from dmgseg.prior.cache import save_prior
        save_prior(self.prior_path(name), probs)

    def load_prior(self, name):
        from dmgseg.prior.cache import load_prior
        return load_prior(self.prior_path(name))

    def n_objects(self, name):
        st = self.load_state(name)
        return len(st["objects"]) if st else 0


# ----------------------------------------------------------------- CVAT XML
def rle_encode(mask):
    """CVAT mask RLE over the mask's bounding box: run lengths, row-major,
    starting with a (possibly empty) background run. -> (rle list, left, top, w, h)"""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    flat = mask[y0:y1, x0:x1].ravel().astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [len(flat)]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return runs, int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def mask_polygons(mask, min_points=3):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] for c in contours if len(c) >= min_points]


def visible_objects_from_state(state):
    """[(class, visible mask)] from a saved state, without SAM (rebuilds a Session-like map)."""
    from dmgseg.app.engine import Obj, paint_order
    h, w = state["height"], state["width"]
    import base64
    objs = []
    from dmgseg.app.engine import state_times
    for d, (created, decided) in zip(state["objects"], state_times(state["objects"])):
        x0, y0, x1, y1 = d["box"]
        m = np.zeros((h, w), bool)
        n = (y1 - y0) * (x1 - x0)
        bits = np.frombuffer(base64.b64decode(d["bits"]), np.uint8)
        m[y0:y1, x0:x1] = np.unpackbits(bits)[:n].reshape(y1 - y0, x1 - x0).astype(bool)
        objs.append(Obj(mask=m, ranking=d["ranking"], choice=d["choice"], manual=d["manual"], pending=d["pending"],
                        decided=decided, created=created))
    lm = np.zeros((h, w), np.uint8)
    order = paint_order(objs)
    for i in order:
        if objs[i].label is not None:
            lm[objs[i].mask] = objs[i].label
    out = []
    for o in objs:
        if o.label is None:
            continue
        vis = o.mask & (lm == o.label)
        if vis.any():
            out.append((o.label, vis))
    return out, lm


def export_cvat(project, out_path, shape="mask", only_annotated=True):
    """CVAT for images 1.1 XML for the folder. shape="mask" (exact) or "polygon"
    (outer contours). Each object is exported as its visible part, so the label map
    is reproduced in any paint order. Returns the number of images written."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
             "  <meta>", "    <task>", f"      <name>{Path(project.folder).name}</name>",
             f"      <size>{len(project.images)}</size>", "      <mode>annotation</mode>", "      <labels>"]
    for name in CLASS_NAMES:
        lines += ["        <label>", f"          <name>{name}</name>", f"          <color>{CVAT_COLORS[name]}</color>",
                  "          <type>any</type>", "          <attributes>", "          </attributes>", "        </label>"]
    lines += ["      </labels>", "    </task>", "  </meta>"]
    written = 0
    for i, name in enumerate(project.images):
        state = project.load_state(name)
        if state is None or not state["objects"]:
            if only_annotated:
                continue
            w, h = Image.open(project.path(name)).size
            lines.append(f'  <image id="{i}" name={quoteattr(name)} width="{w}" height="{h}">')
            lines.append("  </image>")
            continue
        objs, _ = visible_objects_from_state(state)
        lines.append(f'  <image id="{i}" name={quoteattr(name)} width="{state["width"]}" height="{state["height"]}">')
        for z, (cls, m) in enumerate(objs):
            label = CLASS_NAMES[cls]
            if shape == "mask":
                rle, left, top, w, h = rle_encode(m)
                lines.append(f'    <mask label={quoteattr(label)} source="semi-auto" occluded="0" '
                             f'rle="{", ".join(map(str, rle))}" left="{left}" top="{top}" '
                             f'width="{w}" height="{h}" z_order="{z}">')
                lines.append("    </mask>")
            else:
                for poly in mask_polygons(m):
                    pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in poly)
                    lines.append(f'    <polygon label={quoteattr(label)} source="semi-auto" occluded="0" '
                                 f'points="{pts}" z_order="{z}">')
                    lines.append("    </polygon>")
        lines.append("  </image>")
        written += 1
    lines.append("</annotations>")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return written
