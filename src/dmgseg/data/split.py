"""Reproduce the paper's 246/44 train/validation split.

The notebook kept images that have at least one polygon or box, in XML order, and
called sklearn's train_test_split(test_size=0.15, random_state=1212).
"""
import json

from sklearn.model_selection import KFold, train_test_split

from dmgseg import paths
from dmgseg.data.cvat import PAPER_KINDS, parse_annotations

PAPER_SEED = 1212
PAPER_VAL_FRACTION = 0.15


def paper_split(images):
    kept = [a.name for a in images if a.shapes_of(PAPER_KINDS)]
    train, val = train_test_split(kept, test_size=PAPER_VAL_FRACTION, random_state=PAPER_SEED)
    return {"train": train, "val": val}


def write_split_file(out=paths.SPLIT_FILE):
    split = paper_split(parse_annotations(paths.ANNOTATIONS_XML))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1, ensure_ascii=False))
    return split


def load_split(path=paths.SPLIT_FILE):
    return json.loads(path.read_text())


KFOLD_FILE = paths.SPLIT_FILE.parent / "kfold5_train.json"


def kfold_split(train_names, k=5, seed=PAPER_SEED):
    """k folds over the paper's TRAINING images only (validation stays untouched).
    Returns a list of {"train": [...], "heldout": [...]}."""
    folds = []
    for tr, ho in KFold(n_splits=k, shuffle=True, random_state=seed).split(train_names):
        folds.append({"train": [train_names[i] for i in tr], "heldout": [train_names[i] for i in ho]})
    return folds


def load_kfold(path=KFOLD_FILE):
    return json.loads(path.read_text())


if __name__ == "__main__":
    s = write_split_file()
    print(f"train={len(s['train'])} val={len(s['val'])} -> {paths.SPLIT_FILE}")
    folds = kfold_split(s["train"])
    KFOLD_FILE.write_text(json.dumps(folds, indent=1, ensure_ascii=False))
    print("folds (train/heldout):", [(len(f["train"]), len(f["heldout"])) for f in folds], "->", KFOLD_FILE)
