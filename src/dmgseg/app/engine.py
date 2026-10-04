"""Annotation engine: everything the app does, without any GUI (testable headless).

An image session holds the image, the SAM state, the semantic prior (computed in
the background, may arrive later) and a list of objects. Each object has a mask,
a class ranking from the class head (left click = first, right click = next) and
the SAM prompt history, so it can be refined later.

The class head never assigns "Other" (unlabeled pixels are Other), but the user can
mark an object as Other (key 0, or the last step of right-click cycling); such
objects are painted on top and erase the labels behind them.
"""
import base64
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from dmgseg.classhead.features import CARD_SIZE, REFINED, card
from dmgseg.classhead.mlp import CardHead, predict
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.tool.assign import CLICKABLE, class_ranking, fit_prior
from dmgseg.tool.lasso import grab, loop_object, polygon_mask, stroke_points
from dmgseg.tool.prelabel import prelabel as _prelabel

OTHER = 0
# Paint lower-priority classes first, like the dataset. Objects the user marks as
# "Other" (trees, cars, poles in front of a building) are painted LAST, so they cut
# their area out of whatever is behind them; unlabeled pixels are Other as well.
PAINT_ORDER = list(CLICKABLE) + [OTHER]
CYCLE_END = [OTHER]              # right-click cycling ends with "Other"


def paint_order(objects):
    """Indices of the objects in painting order (painted later = on top).

    Every object has a time: when the user last chose its class (number key, menu,
    right click), else when it was created. Each class decision starts a new
    layer: it covers everything older (pressing 1 on a building paints all of it
    Building, over prior blobs too), while objects made afterwards (a window
    clicked on that building) go into the new layer. Within a layer the dataset's
    class priority holds: Building < Roof < Damage < Broken Window < Damaged roof.
    Other is painted last (it cuts its area out); pending objects not at all."""
    import bisect
    decisions = sorted(o.decided for o in objects if o.decided is not None)

    def key(i):
        o = objects[i]
        if o.label == OTHER:
            return (1, 0, 0, i)
        t = o.decided if o.decided is not None else (o.created or 0)
        return (0, bisect.bisect_right(decisions, t), PAINT_ORDER.index(o.label), i)

    return sorted((i for i in range(len(objects)) if objects[i].label is not None), key=key)
COLORS = {0: (160, 160, 160), 1: (40, 170, 60), 2: (255, 165, 0), 3: (170, 50, 200),
          4: (30, 90, 255), 5: (230, 30, 30)}


@dataclass
class Obj:
    mask: np.ndarray
    ranking: list                     # clickable class ids, most likely first
    choice: int = 0                   # index into ranking (right click = +1)
    manual: int | None = None         # class set with a number key
    sam_score: float = 0.0
    cand_type: int = 0
    click_no: int = 1
    points: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    logits: np.ndarray | None = None
    pending: bool = False             # class not yet known (prior still running)
    auto: bool = False                # created by the automatic pre-label
    # True while the mask is exactly SAM's answer to `points`/`logits`; then a
    # Shift/Option click re-asks SAM for the whole object. Otherwise (pre-label
    # blobs, exact edits, clipped or reloaded masks) clicks edit the mask locally.
    sam_valid: bool = True
    decided: int | None = None        # time of the user's last class decision (paint_order)
    created: int = 0                  # time the object was made (paint_order)

    @property
    def label(self):
        """Current class; None while the class is still pending (not painted yet)."""
        if self.manual is not None:
            return self.manual
        if self.pending:
            return None
        return self.ranking[self.choice % len(self.ranking)] if self.ranking else None


def mask_to_logits(mask, size=256, scale=10.0):
    """A mask as SAM's low-res mask prompt (SAM 2 squashes the image to a square),
    so SAM refinements after an exact edit start from the edited shape."""
    m = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return ((m - 0.5) * 2 * scale).astype(np.float32)


def state_times(objs):
    """(created, decided) per saved object. Saves made before objects had times:
    the list order is the creation order, and a class decision is placed right
    after its own object was made (make an object, then press a key)."""
    if all("created" in d for d in objs):
        return [(d["created"], d.get("decided")) for d in objs]
    return [(2 * i + 2, None if d.get("decided") is None and d.get("manual") is None and not d.get("choice")
             else 2 * i + 3) for i, d in enumerate(objs)]


class ClassHead:
    """Head A (MLP on description cards)."""

    def __init__(self, weights):
        self.model = CardHead(CARD_SIZE)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def score(self, cards):
        """-> class probabilities (n, 5) in CLICKABLE order, quality (n,)."""
        return predict(self.model, np.stack(cards).astype(np.float32))


class Session:
    def __init__(self, image, clicker, head=None):
        self.image = np.asarray(image)
        self.h, self.w = self.image.shape[:2]
        self.clicker = clicker
        self.head = head
        self.prior = None
        self.objects: list[Obj] = []
        self.active: int | None = None
        self._undo: list = []
        self.prelabeled = False          # the automatic draft was made (do not redo it)
        clicker.set_image(self.image)
        self.embed = clicker.image_embedding

    # -- prior -------------------------------------------------------------
    def set_prior(self, prior):
        """Called when the background prior is ready: classify pending objects."""
        self.prior = fit_prior(prior, self.h, self.w)
        for o in self.objects:
            if o.pending:
                o.ranking = self._ranking(o)
                o.pending = False

    def _ranking(self, o):
        if self.prior is None:
            return list(CLICKABLE) + CYCLE_END
        if self.head is None:
            return class_ranking(self.prior, o.mask) + CYCLE_END
        probs, _ = self.head.score([card(o.mask, self.prior, self.embed, o.sam_score, o.cand_type, o.click_no)])
        return [CLICKABLE[i] for i in np.argsort(-probs[0])] + CYCLE_END

    # -- automatic first draft ------------------------------------------------
    def prelabel(self, use_head=False, **kw):
        """Add objects for the prior's blobs, snapped to shapes by SAM. Returns how many.
        The class comes from the blob (head A was trained on click masks and is worse
        on these); use_head=True switches to head A."""
        if self.prior is None:
            return 0
        self._snapshot()
        n = 0
        for p in _prelabel(self.prior, self.clicker, **kw):
            o = Obj(mask=p.mask, ranking=[], sam_score=p.sam_score, cand_type=REFINED, click_no=1,
                    points=p.points, labels=p.labels, logits=mask_to_logits(p.mask), auto=True,
                    sam_valid=False, created=self._tick())
            if use_head and self.head is not None:
                o.ranking = self._ranking(o)
            else:
                o.ranking = [p.prior_class] + [c for c in CLICKABLE if c != p.prior_class] + CYCLE_END
            self.objects.append(o)
            n += 1
        self.active = None
        self.prelabeled = True
        return n

    # -- actions -------------------------------------------------------------
    def _snapshot(self):
        self._undo.append((copy.deepcopy(self.objects), self.active))
        del self._undo[:-50]

    def undo(self):
        if self._undo:
            self.objects, self.active = self._undo.pop()

    def object_at(self, x, y):
        """Topmost object under (x, y) in paint order, or None."""
        hits = [i for i, o in enumerate(self.objects) if o.mask[y, x] and o.label is not None]
        if not hits:
            return None
        rank = {i: r for r, i in enumerate(paint_order(self.objects))}
        return max(hits, key=lambda i: rank[i])

    def new_object(self, x, y):
        """Left click: SAM's 3 candidates; the head picks the mask and the class."""
        self._snapshot()
        c = self.clicker
        c.reset_object()
        c.click(x, y, True)
        masks, scores = c.candidates
        if self.head is not None and self.prior is not None:
            cards = [card(m, self.prior, self.embed, float(s), k, 1) for k, (m, s) in enumerate(zip(masks, scores))]
            probs, quality = self.head.score(cards)
            k = int(np.argmax(quality))
            ranking = [CLICKABLE[i] for i in np.argsort(-probs[k])] + CYCLE_END
            pending = False
        else:
            k = int(np.argmax(scores))
            ranking = (class_ranking(self.prior, masks[k]) if self.prior is not None else list(CLICKABLE)) + CYCLE_END
            pending = self.prior is None
        mask = c.choose_index(k)
        o = Obj(mask=mask, ranking=ranking, sam_score=float(scores[k]), cand_type=k, click_no=1,
                points=list(c.points), labels=list(c.labels), logits=c.logits.copy(), pending=pending,
                created=self._tick())
        self.objects.append(o)
        self.active = len(self.objects) - 1
        return self.active

    def refine(self, x, y, positive, index=None):
        """Shift-click (positive) / Shift+right (negative): refine an object's mask."""
        index = self.active if index is None else index
        if index is None:
            return None
        self._snapshot()
        o = self.objects[index]
        c = self.clicker
        if not o.sam_valid:
            self._local_edit(o, x, y, positive)
            self.active = index
            return index
        c.points, c.labels, c.logits = list(o.points), list(o.labels), o.logits
        mask = c.click(x, y, positive)
        o.mask, o.points, o.labels, o.logits = mask, list(c.points), list(c.labels), c.logits.copy()
        o.sam_score, o.cand_type, o.click_no = c.last_score, REFINED, min(o.click_no + 1, 3)
        if o.manual is None and o.choice == 0:
            o.ranking = self._ranking(o)
        self.active = index
        return index

    # -- drawing -------------------------------------------------------------
    def _add(self, mask, prior_class, sam_score, points, labels, click_no=1, sam_valid=False):
        o = Obj(mask=mask, ranking=[], sam_score=sam_score, cand_type=REFINED, click_no=click_no,
                points=list(points), labels=list(labels), logits=mask_to_logits(mask), sam_valid=sam_valid,
                created=self._tick())
        if prior_class is not None:
            o.ranking = [prior_class] + [c for c in CLICKABLE if c != prior_class] + CYCLE_END
        else:
            o.ranking = self._ranking(o)
            o.pending = self.prior is None
        self.objects.append(o)
        return len(self.objects) - 1

    def lasso(self, polygon, group=False):
        """A loop drawn around something. Default: that one object, shaped by SAM
        and chosen by the loop (tool/lasso.loop_object). group=True (Cmd + loop):
        every object the prior finds inside (tool/lasso.grab).
        Returns the new object indices (one undo step)."""
        region = polygon_mask(polygon, self.h, self.w)
        if region.sum() < 16:
            return []
        self._snapshot()
        if not group:
            mask, score, pts, labels, from_sam = loop_object(region, self.clicker)
            i = self._add(mask, None, score, pts, labels, sam_valid=from_sam)
            if from_sam:
                self.objects[i].logits = self.clicker.logits.copy()
            self.active = i
            return [i]
        lm = self.label_map()
        new = []
        for g in grab(region, self.prior, self.clicker, lm):
            i = self._add(g.mask, g.prior_class, g.sam_score, g.points, g.labels)
            o = self.objects[i]
            if g.prior_class is None and o.label is not None and (lm[o.mask] == o.label).mean() >= 0.7:
                self.objects.pop()            # the loop is already labeled with this class
                continue
            new.append(i)
        if not new:
            self._undo.pop()
        else:
            self.active = new[-1]
        return new

    def scribble(self, stroke, positive=True, index=None):
        """An open stroke. Without index: a new object through points along it.
        With index: refine that object with all those points (grow / shrink)."""
        pts = [(min(max(x, 0), self.w - 1), min(max(y, 0), self.h - 1)) for x, y in stroke_points(stroke)]
        self._snapshot()
        c = self.clicker
        if index is None:
            mask = c.prompt(pts, [1] * len(pts))
            o = Obj(mask=mask, ranking=[], sam_score=c.last_score, cand_type=REFINED, click_no=2,
                    points=list(c.points), labels=list(c.labels), logits=c.logits.copy(), created=self._tick())
            o.ranking = self._ranking(o)
            o.pending = self.prior is None
            self.objects.append(o)
            self.active = len(self.objects) - 1
            return self.active
        o = self.objects[index]
        if not o.sam_valid:                   # local: add / remove the piece along the line
            piece = c.prompt(pts, [1] * len(pts))
            o.mask = (o.mask | piece) if positive else (o.mask & ~piece)
            o.logits, o.cand_type = mask_to_logits(o.mask), REFINED
            c.reset_object()
            self.active = index
            return index
        logits = o.logits if o.logits is not None else mask_to_logits(o.mask)
        mask = c.prompt(list(o.points) + pts, list(o.labels) + [int(positive)] * len(pts), logits)
        o.mask, o.points, o.labels, o.logits = mask, list(c.points), list(c.labels), c.logits.copy()
        o.sam_score, o.cand_type, o.click_no = c.last_score, REFINED, min(o.click_no + 1, 3)
        if o.manual is None and o.choice == 0:
            o.ranking = self._ranking(o)
        self.active = index
        return index

    def edit_area(self, polygon, add=True, index=None):
        """Exact pixel edit with a loop (no SAM). add: the area joins object `index`
        (a new object if None). Cut: the area leaves object `index`, or every object
        when index is None; objects left empty are removed. Returns the object index."""
        region = polygon_mask(polygon, self.h, self.w)
        if not region.any():
            return index
        self._snapshot()
        if add:
            if index is None:
                return self._add_exact(region)
            targets = [index]
        else:
            targets = [index] if index is not None else [i for i, o in enumerate(self.objects)
                                                         if (o.mask & region).any()]
        for i in targets:
            o = self.objects[i]
            o.mask = (o.mask | region) if add else (o.mask & ~region)
            o.logits, o.cand_type, o.sam_valid = mask_to_logits(o.mask), REFINED, False
        keep = [i for i, o in enumerate(self.objects) if o.mask.any()]
        if len(keep) < len(self.objects):
            active = self.objects[index] if index is not None and index in keep else None
            self.objects = [self.objects[i] for i in keep]
            index = self.objects.index(active) if active is not None else None
        self.active = index
        return index

    def _add_exact(self, region):
        ys, xs = np.nonzero(region)
        i = self._add(region, None, 0.0, [(xs.min(), ys.min()), (xs.max() + 1, ys.max() + 1)], [2, 3])
        self.active = i
        return i

    def _local_edit(self, o, x, y, positive):
        """Shift/Option click on a mask that is not SAM's own: SAM segments the
        piece under the cursor (one click, 3 candidates) and only that piece is
        added to / removed from the mask; the rest of the mask stays as it is."""
        c = self.clicker
        c.reset_object()
        c.click(x, y, True)
        masks, scores = c.candidates
        area = o.mask.sum()
        limit = area if positive else 0.6 * area      # a piece, not a whole new object
        ok = [k for k in range(len(masks)) if masks[k].sum() <= limit]
        k = max(ok, key=lambda k: scores[k]) if ok else int(np.argmin([m.sum() for m in masks]))
        o.mask = (o.mask | masks[k]) if positive else (o.mask & ~masks[k])
        o.logits, o.cand_type = mask_to_logits(o.mask), REFINED
        c.reset_object()

    def delete_many(self, indices):
        """Remove several objects in one undo step."""
        drop = set(indices)
        if not drop:
            return
        self._snapshot()
        self.objects = [o for i, o in enumerate(self.objects) if i not in drop]
        self.active = None

    def _tick(self):
        """The session clock: one step per creation / class decision."""
        return 1 + max((max(x.decided or 0, x.created) for x in self.objects), default=0)

    def _decide(self, o):
        o.decided = self._tick()

    def next_class(self, index):
        """Right click on an object: the next class in its ranking."""
        self._snapshot()
        o = self.objects[index]
        o.manual = None
        o.choice = (o.choice + 1) % len(o.ranking)
        self._decide(o)
        self.active = index

    def set_class(self, index, class_id):
        self._snapshot()
        self.objects[index].manual = class_id
        self._decide(self.objects[index])
        self.active = index

    def set_classes(self, indices, class_id):
        """Set the class of several objects in one undo step."""
        if not indices:
            return
        self._snapshot()
        for i in indices:
            self.objects[i].manual = class_id
            self._decide(self.objects[i])
        self.active = indices[-1]

    def delete(self, index):
        self._snapshot()
        del self.objects[index]
        self.active = None

    # -- saving and loading ----------------------------------------------------
    def to_state(self):
        """Everything needed to continue later (masks stored as packed bits in their box)."""
        objs = []
        for o in self.objects:
            ys, xs = np.nonzero(o.mask)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            objs.append({
                "box": [x0, y0, x1, y1],
                "bits": base64.b64encode(np.packbits(o.mask[y0:y1, x0:x1]).tobytes()).decode("ascii"),
                "ranking": list(map(int, o.ranking)), "choice": int(o.choice),
                "manual": None if o.manual is None else int(o.manual), "sam_score": float(o.sam_score),
                "cand_type": int(o.cand_type), "click_no": int(o.click_no),
                "points": [[float(a), float(b)] for a, b in o.points], "labels": list(map(int, o.labels)),
                "pending": bool(o.pending), "auto": bool(o.auto), "decided": o.decided,
                "created": int(o.created)})
        return {"version": 1, "width": self.w, "height": self.h, "prelabeled": self.prelabeled, "objects": objs}

    def load_state(self, state):
        if (state["width"], state["height"]) != (self.w, self.h):
            raise ValueError("saved annotation does not match the image size")
        self.objects = []
        for d, (created, decided) in zip(state["objects"], state_times(state["objects"])):
            x0, y0, x1, y1 = d["box"]
            mask = np.zeros((self.h, self.w), bool)
            n = (y1 - y0) * (x1 - x0)
            bits = np.frombuffer(base64.b64decode(d["bits"]), np.uint8)
            mask[y0:y1, x0:x1] = np.unpackbits(bits)[:n].reshape(y1 - y0, x1 - x0).astype(bool)
            self.objects.append(Obj(mask=mask, ranking=d["ranking"], choice=d["choice"], manual=d["manual"],
                                    sam_score=d["sam_score"], cand_type=d["cand_type"], click_no=d["click_no"],
                                    points=[tuple(p) for p in d["points"]], labels=d["labels"],
                                    logits=mask_to_logits(mask), pending=d["pending"], auto=d["auto"],
                                    decided=decided, created=created,
                                    sam_valid=False))
        self.prelabeled = state.get("prelabeled", False)
        self.active, self._undo = None, []

    # -- output --------------------------------------------------------------
    def label_map(self):
        """(H, W) class ids; objects painted in paint_order (Other = 0)."""
        out = np.zeros((self.h, self.w), np.uint8)
        order = paint_order(self.objects)
        for i in order:
            o = self.objects[i]
            if o.label is not None:
                out[o.mask] = o.label
        return out

    def counts(self):
        labels = [o.label for o in self.objects if o.label is not None]
        return {c: labels.count(c) for c in list(CLICKABLE) + [OTHER]}

    def other_mask(self):
        """Pixels explicitly marked as Other (shown hatched gray in the overlay)."""
        lm = self.label_map()
        m = np.zeros((self.h, self.w), bool)
        for o in self.objects:
            if o.label == OTHER:
                m |= o.mask
        return m & (lm == OTHER)

    def export(self, png_path, json_path=None, source_name=""):
        """Writes three files:
        <name>.png          class id per pixel (0 Other ... 5 Damaged roof), saved as a
                            palette PNG: viewers show the class colors, while reading it
                            with PIL (np.array(Image.open(...))) gives the ids 0-5
        <name>_overlay.jpg  the photo with the class colors, for looking at
        <name>.json         objects with class, area, bbox and outline polygons"""
        from PIL import Image as _Image
        lm = self.label_map()
        pal = _Image.fromarray(lm, mode="P")
        palette = [0] * 768
        for c, rgb in COLORS.items():
            palette[3 * c:3 * c + 3] = list(rgb)
        pal.putpalette(palette)
        pal.save(str(png_path))
        _Image.fromarray(self.overlay(alpha=0.5, active_outline=False)).save(
            str(Path(png_path).with_name(Path(png_path).stem + "_overlay.jpg")), quality=90)
        objs = []
        for o in self.objects:
            if o.label is None:
                continue
            contours, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ys, xs = np.nonzero(o.mask)
            objs.append({"class": CLASS_NAMES[o.label], "area": int(o.mask.sum()),
                         "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                         "polygons": [c[:, 0, :].tolist() for c in contours if len(c) >= 3],
                         "manual_class": o.manual is not None, "right_clicks": o.choice})
        json_path = json_path or Path(png_path).with_suffix(".json")
        Path(json_path).write_text(json.dumps({"image": source_name, "width": self.w, "height": self.h,
                                               "objects": objs}, indent=1))
        return png_path, json_path

    def visible_objects(self):
        """[(class id, visible mask)]: the part of each object that shows in the
        label map. Disjoint across classes, so any paint order reproduces the label
        map (needed for CVAT, where Other is painted below everything)."""
        lm = self.label_map()
        out = []
        for o in self.objects:
            if o.label is None:
                continue
            vis = o.mask & (lm == o.label)
            if vis.any():
                out.append((o.label, vis))
        return out

    def overlay(self, alpha=0.45, active_outline=True):
        """RGB image with the class colors blended in (for display and tests)."""
        lm = self.label_map()
        out = self.image.astype(np.float32).copy()
        for c in CLICKABLE:
            m = lm == c
            out[m] = (1 - alpha) * out[m] + alpha * np.array(COLORS[c])
        other = self.other_mask()
        if other.any():
            yy, xx = np.mgrid[:self.h, :self.w]
            stripes = ((xx + yy) // 6) % 2 == 0          # hatched gray = marked as Other
            m = other & stripes
            out[m] = (1 - alpha) * out[m] + alpha * np.array(COLORS[OTHER])
            m = other & ~stripes
            out[m] = (1 - alpha / 2) * out[m] + (alpha / 2) * np.array(COLORS[OTHER])
        out = out.astype(np.uint8)
        if active_outline and self.active is not None and self.active < len(self.objects):
            cnt, _ = cv2.findContours(self.objects[self.active].mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnt, -1, (255, 255, 0), 2)
        return out
