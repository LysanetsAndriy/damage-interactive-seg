"""Headless check of the app: open a validation image, click objects, save a screenshot.

    QT_QPA_PLATFORM=offscreen python scripts/app_screenshot.py --out artifacts/figures/app.png
"""
import argparse
import functools
import os
import time

print = functools.partial(print, flush=True)  # noqa: A001

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from dmgseg import paths  # noqa: E402
from dmgseg.app.gui import MainWindow, apply_classic_style  # noqa: E402
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations  # noqa: E402
from dmgseg.data.objects import image_objects  # noqa: E402
from dmgseg.data.split import load_split  # noqa: E402
from dmgseg.eval.simulator import next_click  # noqa: E402


def wait(app, cond, timeout=600):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=load_split()["val"][2])
    ap.add_argument("--clicks", type=int, default=8)
    ap.add_argument("--out", default="artifacts/figures/app.png")
    args = ap.parse_args()

    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = MainWindow()
    win.resize(1400, 860)
    win.show()
    t = time.time()
    wait(app, win.ready)
    print(f"models loaded in {time.time() - t:.0f}s")
    win.open_image(str(paths.IMAGES_DIR / args.image))
    wait(app, lambda: win.session is not None)
    t = time.time()
    wait(app, lambda: win.session.prior is not None)
    print(f"prior in {time.time() - t:.0f}s")

    ann = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}[args.image]
    objs = sorted([o for o in image_objects(ann, 600) if o.label != 0], key=lambda o: -o.area)[:args.clicks]
    for o in objs:
        x, y, _ = next_click(None, o.mask)
        win.on_click(x, y, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        got = win.session.objects[win.session.active].label
        print(f"clicked {CLASS_NAMES[o.label]:14} -> {CLASS_NAMES[got]}")
    # one right click on the first object, to show the class switch
    x, y, _ = next_click(None, objs[0].mask)
    win.on_click(x, y, QtCore.Qt.RightButton, QtCore.Qt.NoModifier)
    app.processEvents()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    win.grab().save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
