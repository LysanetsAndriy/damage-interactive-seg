"""Headless check of the folder workflow (QT_QPA_PLATFORM=offscreen):
open a folder, pre-label + clicks on image 1, next image (prior pre-computed in the
background), back (objects restored from disk), CVAT export read back with the
dataset parser.
"""
import functools
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

from dmgseg import paths  # noqa: E402
from dmgseg.app.gui import MainWindow, apply_classic_style  # noqa: E402
from dmgseg.app.project import export_cvat  # noqa: E402
from dmgseg.data.cvat import parse_annotations, semantic_mask  # noqa: E402
from dmgseg.data.split import load_split  # noqa: E402


def wait(app, cond, timeout=900):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


class Heartbeat:
    """Longest gap between 20 ms timer ticks = longest time the window was frozen."""

    def __init__(self):
        self.last, self.worst = time.perf_counter(), 0.0
        self.timer = QtCore.QTimer(interval=20, timeout=self.tick)
        self.timer.start()

    def tick(self):
        now = time.perf_counter()
        self.worst = max(self.worst, now - self.last)
        self.last = now

    def take(self):
        w, self.worst = self.worst, 0.0
        return w


def main():
    folder = Path(tempfile.mkdtemp(prefix="annotator_"))
    names = load_split()["val"][:3]
    for n in names:
        shutil.copy(paths.IMAGES_DIR / n, folder / n)
    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = MainWindow()
    win.resize(1500, 880)
    win.show()
    wait(app, win.ready)
    hb = Heartbeat()
    hb.take()
    win.open_folder(str(folder))
    first = win.current_name
    wait(app, lambda: win.session is not None and win.session.prior is not None and win.session.prelabeled)
    print(f"image 1 ({first}): draft with {len(win.session.objects)} objects; "
          f"longest freeze while opening + drafting: {hb.take():.2f} s")
    for x, y in [(200, 200), (300, 250)]:
        win.on_click(x, y, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    n1 = len(win.session.objects)
    lm1 = win.session.label_map().copy()
    # the second image's prior should be pre-computed while we were on the first
    wait(app, lambda: win.project.has_prior(win.project.images[1]))
    t0 = time.perf_counter()
    win.step(1)
    wait(app, lambda: win.session is not None)
    print(f"switch to image 2: ready to click after {time.perf_counter() - t0:.2f} s")
    wait(app, lambda: win.session is not None and win.session.prior is not None and win.session.prelabeled)
    print(f"image 2: prior '{win.prior_label.text().strip()}', draft with {len(win.session.objects)} objects")
    t0 = time.perf_counter()
    win.step(-1)
    wait(app, lambda: win.session is not None and win.current_name == first and win.session.prior is not None)
    print(f"switch back to image 1: ready after {time.perf_counter() - t0:.2f} s; "
          f"longest freeze so far: {hb.take():.2f} s")
    print(f"back to image 1: {len(win.session.objects)} objects restored (expected {n1}); "
          f"label map identical: {np.array_equal(win.session.label_map(), lm1)}")
    print("image list:", [win.image_list.item(i).text() for i in range(win.image_list.count())])
    win.grab().save("artifacts/figures/app_v1.png")
    win.save_current()
    out = folder / "cvat.xml"
    n = export_cvat(win.project, out, shape="mask")
    anns = {a.name: a for a in parse_annotations(out)}
    same = np.array_equal(semantic_mask(anns[first]), lm1)
    print(f"CVAT export: {n} images; image 1 read back with the dataset parser == app label map: {same}")
    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
