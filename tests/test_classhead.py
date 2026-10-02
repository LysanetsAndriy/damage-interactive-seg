import numpy as np
import torch

from dmgseg.classhead.features import CARD_SIZE, card, group_slices
from dmgseg.classhead.mlp import CardHead, evaluate_cards


def test_card_layout_and_values():
    h, w = 64, 80
    mask = np.zeros((h, w), bool)
    mask[10:30, 20:60] = True                      # a 20 x 40 rectangle
    prior = np.zeros((h, w, 6), np.float32)
    prior[..., 4] = 1.0                            # prior says Broken Window everywhere
    embed = torch.ones(256, 64, 64)
    v = card(mask, prior, embed, 0.7, 1, 1)
    s = group_slices()
    assert v.shape == (CARD_SIZE,) and np.isfinite(v).all()
    assert np.isclose(v[s["prior_in"]][4], 1.0)    # mean prob of class 4
    assert np.isclose(v[s["prior_in"]][6 + 4], 1.0)  # all pixels argmax class 4
    assert np.allclose(v[s["sam"]], 1.0)
    g = v[s["geometry"]]
    assert np.isclose(g[4], 1.0)                   # a rectangle fills its bbox
    assert np.isclose(g[3], np.log(40 / 20))       # aspect
    info = v[s["sam_info"]]
    assert np.isclose(info[0], 0.7) and info[1 + 1] == 1 and np.isclose(info[5], 1 / 3)


def test_evaluate_cards_runs_on_synthetic_cards():
    rng = np.random.default_rng(0)
    n_obj, rows = 6, []
    X, label, iou, obj, cand, click = [], [], [], [], [], []
    for o in range(n_obj):
        for k, c in [(0, 1), (1, 1), (2, 1), (3, 2), (3, 3)]:
            X.append(rng.normal(size=CARD_SIZE).astype(np.float32)); label.append(1 + o % 5)
            iou.append(rng.random()); obj.append(o); cand.append(k); click.append(c)
    cards = {"X": np.stack(X), "label": np.array(label), "iou": np.array(iou, np.float32),
             "obj": np.array(obj), "cand": np.array(cand), "click": np.array(click)}
    m = evaluate_cards(CardHead(CARD_SIZE), cards)["all"]
    assert m["n"] == n_obj
    assert 0 <= m["head/iou1"] <= m["oracle/iou1"] + 1e-9
