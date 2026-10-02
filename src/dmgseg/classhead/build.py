"""Build description cards by running the simulated user with SAM.

Per object (clickable classes only, area >= MIN_AREA, duplicates removed):
- first click(s): the 3 SAM candidates -> 3 cards (cand_type 0/1/2, click 1);
  on training images also a second, randomly placed first click (augmentation);
- clicks 2 and 3 continuing from SAM's own choice -> refined masks (cand_type 3).
Targets per card: the object's class and the mask's IoU with the object.

    cards = build_cards(names, prior_dir, clicker, jitter=True)
    save_cards(cards, path)
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from dmgseg import paths
from dmgseg.data.cvat import parse_annotations
from dmgseg.data.objects import image_objects
from dmgseg.eval.run_interactive import MIN_AREA
from dmgseg.eval.simulator import iou, next_click
from dmgseg.prior.kfold import load_prior
from dmgseg.tool.assign import CLICKABLE, fit_prior
from dmgseg.classhead.features import REFINED, card


def _random_interior_click(mask, rng):
    """A random point well inside the object (distance >= 30% of the max depth)."""
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    ys, xs = np.nonzero(dt >= 0.3 * dt.max())
    i = rng.integers(len(ys))
    return int(xs[i]), int(ys[i])


def build_cards(names, prior_dir, clicker, jitter=False, refine_clicks=2, seed=0, image_dir=None):
    anns = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}
    rng = np.random.default_rng(seed)
    X, label, target_iou, obj, cand, click, objects = [], [], [], [], [], [], []
    for name in tqdm(names, desc="cards"):
        ann = anns[name]
        image = np.asarray(Image.open(Path(image_dir or paths.IMAGES_DIR) / name).convert("RGB"))
        prior = fit_prior(load_prior(Path(prior_dir) / f"{name}.npz"), *image.shape[:2])
        clicker.set_image(image)
        embed = clicker.image_embedding
        for o in image_objects(ann, MIN_AREA):
            if o.label not in CLICKABLE:
                continue
            oid = len(objects)
            objects.append((name, o.index, o.label, o.area))

            def add(mask, score, ctype, cno):
                X.append(card(mask, prior, embed, score, ctype, cno))
                label.append(o.label); target_iou.append(iou(mask, o.mask))
                obj.append(oid); cand.append(ctype); click.append(cno)

            starts = [next_click(None, o.mask)[:2]]
            if jitter:
                starts.append(_random_interior_click(o.mask, rng))
            for s_i, (x, y) in enumerate(starts):
                clicker.reset_object()
                pred = clicker.click(x, y, True)
                masks, scores = clicker.candidates
                for k in range(len(masks)):
                    add(masks[k], float(scores[k]), k, 1)
                if s_i == 0:  # refine only from the standard (center) first click
                    for c in range(refine_clicks):
                        cx, cy, pos = next_click(pred, o.mask)
                        pred = clicker.click(cx, cy, pos)
                        add(pred, clicker.last_score, REFINED, 2 + c)
    return {
        "X": np.stack(X).astype(np.float32), "label": np.array(label, np.int64),
        "iou": np.array(target_iou, np.float32), "obj": np.array(obj, np.int64),
        "cand": np.array(cand, np.int64), "click": np.array(click, np.int64),
        "obj_image": np.array([o[0] for o in objects]), "obj_shape": np.array([o[1] for o in objects]),
        "obj_label": np.array([o[2] for o in objects]), "obj_area": np.array([o[3] for o in objects]),
    }


def save_cards(cards, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cards)


def load_cards(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
