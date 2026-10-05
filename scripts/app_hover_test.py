"""Headless check of the hover preview: the cursor rests on 8 validation objects;
how long until the outline appears, and does it match what a click then creates?"""
import functools
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

from dmgseg import paths  # noqa: E402
from dmgseg.app.gui import MainWindow, apply_classic_style  # noqa: E402
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations  # noqa: E402
from dmgseg.data.objects import image_objects  # noqa: E402
from dmgseg.tool.prelabel import deepest_point  # noqa: E402

NAME = "download29_1.png"


def wait(app, cond, timeout=600):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


def main():
    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = MainWindow()
    win.resize(1400, 850)
    win.show()
    wait(app, win.ready)
    win.auto_check.setChecked(False)
    win.open_image(str(paths.IMAGES_DIR / NAME))
    wait(app, lambda: win.session is not None and win.session.prior is not None)
    ann = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}[NAME]
    objs = [o for o in image_objects(ann, 400) if o.label != 0][:8]
    shown = []
    win.canvas._hover.setPath = (lambda orig: (lambda p: (shown.append(time.perf_counter()), orig(p))))(win.canvas._hover.setPath)
    times, agree, classes = [], [], []
    for o in objs:
        x, y = deepest_point(o.mask)
        shown.clear()
        t0 = time.perf_counter()
        win.on_hover(x, y)
        wait(app, lambda: win.canvas._hover.path().elementCount() > 0, timeout=30)
        times.append(time.perf_counter() - t0)
        label = win.canvas._hover_label.text()
        win.on_click(x, y, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        made = win.session.objects[-1]
        # the preview outline vs the clicked mask: compare via the preview call itself
        m, c = win.session.preview(x, y, win.models.hover)
        agree.append((m & made.mask).sum() / max((m | made.mask).sum(), 1))
        classes.append((label, CLASS_NAMES[made.label]))
    print(f"hover -> outline shown: median {1000 * np.median(times):.0f} ms, max {1000 * max(times):.0f} ms "
          f"(includes the 60 ms rest delay)")
    print(f"preview mask vs clicked mask IoU: {np.round(agree, 3).tolist()}")
    print(f"preview class == clicked class: {sum(a == b for a, b in classes)}/{len(classes)}  {classes}")
    win.on_hover(*deepest_point(objs[0].mask))
    wait(app, lambda: win.canvas._hover.path().elementCount() > 0, timeout=30)
    win.grab().save("artifacts/figures/app_hover.png")


if __name__ == "__main__":
    main()
