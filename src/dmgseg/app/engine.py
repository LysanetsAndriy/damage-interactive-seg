"""Annotation engine: everything the app does, without any GUI (testable headless).

An image session holds the image, the SAM state, the semantic prior (computed in
the background, may arrive later) and a list of objects. Each object has a mask,
a class ranking from the class head (left click = first, right click = next) and
the SAM prompt history, so it can be refined later.

The class head never assigns "Other" (unlabeled pixels are Other), but the user can
mark an object as Other (key 0, or the last step of right-click cycling); such
objects are painted on top and erase the labels behind them.
"""
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
from dmgseg.tool.prelabel import prelabel as _prelabel

OTHER = 0
# Paint lower-priority classes first, like the dataset. Objects the user marks as
# "Other" (trees, cars, poles in front of a building) are painted LAST, so they cut
# their area out of whatever is behind them; unlabeled pixels are Other as well.
PAINT_ORDER = list(CLICKABLE) + [OTHER]
CYCLE_END = [OTHER]              # right-click cycling ends with "Other"
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

    @property
    def label(self):
        """Current class; None while the class is still pending (not painted yet)."""
        if self.manual is not None:
            return self.manual
        if self.pending:
            return None
        return self.ranking[self.choice % len(self.ranking)] if self.ranking else None


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
        clicker.set_image(self.image)
        self.embed = clicker.image_embedding

    # -- prior -------------------------------------------------------------
    def set_prior(self, prior):
        """Called when the background prior is ready: classify pending objects."""
        self.prior = fit_prior(prior, self.h, self.w)
        for o in self.objects:
            if o.pending or (o.manual is None and o.choice == 0):
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
                    points=p.points, labels=p.labels, logits=p.logits, auto=True)
            if use_head and self.head is not None:
                o.ranking = self._ranking(o)
            else:
                o.ranking = [p.prior_class] + [c for c in CLICKABLE if c != p.prior_class] + CYCLE_END
            self.objects.append(o)
            n += 1
        self.active = None
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
        return max(hits, key=lambda i: (PAINT_ORDER.index(self.objects[i].label), i))

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
                points=list(c.points), labels=list(c.labels), logits=c.logits.copy(), pending=pending)
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
        c.points, c.labels, c.logits = list(o.points), list(o.labels), o.logits
        mask = c.click(x, y, positive)
        o.mask, o.points, o.labels, o.logits = mask, list(c.points), list(c.labels), c.logits.copy()
        o.sam_score, o.cand_type, o.click_no = c.last_score, REFINED, min(o.click_no + 1, 3)
        if o.manual is None and o.choice == 0:
            o.ranking = self._ranking(o)
        self.active = index
        return index

    def next_class(self, index):
        """Right click on an object: the next class in its ranking."""
        self._snapshot()
        o = self.objects[index]
        o.manual = None
        o.choice = (o.choice + 1) % len(o.ranking)
        self.active = index

    def set_class(self, index, class_id):
        self._snapshot()
        self.objects[index].manual = class_id
        self.active = index

    def delete(self, index):
        self._snapshot()
        del self.objects[index]
        self.active = None

    # -- output --------------------------------------------------------------
    def label_map(self):
        """(H, W) class ids; objects painted in class-priority order (Other = 0)."""
        out = np.zeros((self.h, self.w), np.uint8)
        order = sorted(range(len(self.objects)),
                       key=lambda i: (PAINT_ORDER.index(self.objects[i].label)
                                      if self.objects[i].label is not None else -1, i))
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
        """PNG: class id per pixel. JSON: objects with class, area, bbox, polygon."""
        lm = self.label_map()
        cv2.imwrite(str(png_path), lm)
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
