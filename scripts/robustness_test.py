"""Robustness of the tool to where the user clicks first (validation, 44 images).

center   = the standard protocol (deepest point of the object)
random   = random point at >= 30 % of the object's depth (a typical user)
anywhere = any pixel inside the object (a careless user)
Rules (prior average + SAM score) and head A are evaluated on the same masks.

    python scripts/robustness_test.py
"""
import json
from pathlib import Path

import torch

from dmgseg import paths
from dmgseg.classhead.build import build_cards, load_cards, save_cards
from dmgseg.classhead.features import CARD_SIZE
from dmgseg.classhead.mlp import CardHead, evaluate_cards
from dmgseg.classhead.train_a import fetch
from dmgseg.data.split import load_split
from dmgseg.sam.predictor import SamClicker

OUT = paths.ARTIFACTS / "classhead"


def main():
    head = CardHead(CARD_SIZE)
    head.load_state_dict(torch.load(OUT / "head_a.pt"))
    head.eval()
    results = {"center": evaluate_cards(head, fetch("val"))}
    clicker = None
    for start in ("random", "anywhere"):
        path = OUT / f"cards_val_{start}.npz"
        if not path.exists():
            clicker = clicker or SamClicker("small", "cpu")
            cards = build_cards(load_split()["val"], paths.ARTIFACTS / "priors" / paths.PRIOR_RUN, clicker,
                                start=start, seed=1)
            save_cards(cards, path)
        results[start] = evaluate_cards(head, load_cards(path))
    print(f"{'first click':12}{'objects':>9} | {'IoU 1st click: SAM':>19}{'head':>7} | "
          f"{'right class: rules':>19}{'head':>7} | {'manual: rules':>14}{'head':>7}")
    for start, r in results.items():
        a = r["all"]
        print(f"{start:12}{a['n']:>9} | {a['sam/iou1']:19.3f}{a['head/iou1']:7.3f} | "
              f"{a['prior/top1@refined']:19.1%}{a['head/top1@refined']:7.1%} | "
              f"{a['prior/manual@refined']:14.1%}{a['head/manual@refined']:7.1%}", flush=True)
    (OUT / "robustness.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
