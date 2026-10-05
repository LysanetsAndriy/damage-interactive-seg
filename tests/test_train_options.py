import math

import torch

from dmgseg.prior.train import make_scheduler, param_groups
from dmgseg.prior.unet_dinov2 import UNetDinoV2

MODEL = UNetDinoV2(pretrained=False)


def test_layer_decay_groups():
    groups = param_groups(MODEL, 4e-5, 1e-3, layer_decay=0.8)
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == len(list(MODEL.parameters()))
    lrs = [g["lr"] for g in groups]
    assert lrs == sorted(lrs)                                   # deeper -> larger lr
    assert math.isclose(lrs[-1], 4e-5)                          # decoder side
    assert math.isclose(lrs[-2], 4e-5 * 0.8)                    # last encoder block
    assert math.isclose(lrs[0], 4e-5 * 0.8 ** 22)               # embeddings
    assert len(param_groups(MODEL, 4e-5, 1e-3)) == 1            # default: one group


def test_cosine_warmup_then_decay():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched, per_step = make_scheduler(opt, {"type": "cosine", "warmup_epochs": 1, "min_lr_ratio": 0.01}, 10, 5)
    assert per_step
    lrs = []
    for _ in range(50):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9] <= 1.0              # warm-up
    assert math.isclose(lrs[10], 1.0)          # peak right after warm-up
    assert lrs[-1] < 0.05                      # decays to ~min_lr_ratio
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))


def test_plateau_is_default():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched, per_step = make_scheduler(opt, {"patience": 2, "factor": 0.4}, 10, 5)
    assert not per_step and isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


import pytest  # noqa: E402


@pytest.mark.parametrize("name", ["v3_640", "v3_640_stem", "v3_cnx"])
def test_dinov3_configs_build_train_groups_and_full_size_output(name):
    """Each DINOv3 config builds (random weights, small input), gives full-size
    logits, and layer-wise LR decay covers every parameter once, smallest LR at the
    input end of the encoder, full LR for the new layers."""
    import torch
    from dmgseg.config import load_config
    from dmgseg.prior.train import param_groups
    from dmgseg.prior.unet_dinov2 import build_prior_model
    from dmgseg import paths
    cfg = load_config(paths.PROJECT_ROOT / "configs" / f"prior_dinov3_6c_{name}.yaml")
    model = build_prior_model(cfg.model, pretrained=False, img_size=128)
    groups = param_groups(model, 1e-4, 1e-3, cfg.train.layer_decay)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in model.parameters() if p.requires_grad)
    lrs = [g["lr"] for g in groups]
    assert lrs == sorted(lrs) and lrs[-1] == pytest.approx(1e-4) and lrs[0] < 1e-5
    head = {id(p) for p in model.conv_last.parameters()}
    assert any(head & {id(p) for p in g["params"]} for g in groups if g["lr"] == pytest.approx(1e-4))
    with torch.no_grad():
        y = model.eval()(torch.randn(1, 3, 128, 128), torch.randn(1, 1536), torch.rand(1, 2))
    assert y.shape == (1, 6, 128, 128)
