"""Self-test of an installed / packaged app (no window): loads the models, computes
the prior of a synthetic facade, clicks, drafts and exports CVAT XML.

    "Damage Annotator.app/Contents/MacOS/Damage Annotator" --selftest
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def synthetic_facade(h=600, w=800, seed=0):
    """A grey building with a roof, dark windows and a burnt patch (no files needed)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (170, 200, 230), np.uint8)                 # sky
    img[150:, 100:700] = (190, 180, 165)                                 # wall
    img[120:150, 90:710] = (120, 70, 50)                                 # roof
    for y in range(190, h - 40, 70):
        for x in range(140, 680, 80):
            img[y:y + 40, x:x + 35] = (40, 45, 55)                       # windows
    img[300:420, 420:560] = (60, 50, 45)                                 # burnt patch
    return np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)


def run():
    from dmgseg import paths
    from dmgseg.app.engine import Session
    from dmgseg.app.gui import Models
    from dmgseg.app.project import FolderProject, export_cvat
    from dmgseg.data.cvat import CLASS_NAMES
    t0 = time.time()
    print(f"backend: {paths.BACKEND} | models: {paths.ONNX_DIR}", flush=True)
    m = Models().load()
    print(f"models loaded in {time.time() - t0:.1f} s | PyTorch imported: {'torch' in sys.modules}", flush=True)
    img = synthetic_facade()
    t = time.time()
    prior = m.compute_prior(img)
    print(f"prior {prior.shape} in {time.time() - t:.1f} s", flush=True)
    s = Session(img, m.clicker, m.head)
    s.set_prior(prior)
    t = time.time()
    i = s.new_object(157, 210)
    print(f"click on a window -> {CLASS_NAMES[s.objects[i].label]}, {s.objects[i].area} px, "
          f"{1000 * (time.time() - t):.0f} ms", flush=True)
    n = s.prelabel(keep_small_px=25)
    print(f"draft: {n} objects", flush=True)
    with tempfile.TemporaryDirectory() as d:
        from PIL import Image
        Image.fromarray(img).save(Path(d) / "facade.png")
        proj = FolderProject(d)
        proj.save_state("facade.png", s.to_state())
        k = export_cvat(proj, Path(d) / "cvat.xml")
        print(f"CVAT export: {k} image(s), {(Path(d) / 'cvat.xml').stat().st_size} bytes", flush=True)
    print(f"SELFTEST OK in {time.time() - t0:.0f} s", flush=True)
    return 0
