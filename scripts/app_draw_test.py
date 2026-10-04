"""Headless check of the drawing gestures with real mouse events on the canvas:
a loop around one window (one object), Cmd+loop around a group of windows (grab),
a scribble (one object), Shift+loop (exact add), Option+loop (exact cut), an
Option+click on a pre-label object (local cut), multi-delete from the list. Saves artifacts/figures/app_lasso.png
(mid-drawing) and app_lasso_result.png."""
import functools
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from dmgseg import paths  # noqa: E402
from dmgseg.app.gui import MainWindow, apply_classic_style  # noqa: E402
from dmgseg.data.cvat import CLASS_NAMES, parse_annotations  # noqa: E402
from dmgseg.data.objects import image_objects  # noqa: E402

NAME = "download14_1.png"


def wait(app, cond, timeout=900):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


def mouse(canvas, kind, xy, mods=QtCore.Qt.NoModifier):
    pos = QtCore.QPointF(canvas.mapFromScene(QtCore.QPointF(*xy)))
    t = {"press": QtCore.QEvent.MouseButtonPress, "move": QtCore.QEvent.MouseMove,
         "release": QtCore.QEvent.MouseButtonRelease}[kind]
    btn = QtCore.Qt.NoButton if kind == "move" else QtCore.Qt.LeftButton
    e = QtGui.QMouseEvent(t, pos, canvas.mapToGlobal(pos.toPoint()), btn, QtCore.Qt.LeftButton, mods)
    {"press": canvas.mousePressEvent, "move": canvas.mouseMoveEvent, "release": canvas.mouseReleaseEvent}[kind](e)


def draw(app, canvas, pts, mods=QtCore.Qt.NoModifier, shot=None):
    mouse(canvas, "press", pts[0], mods)
    for p in pts[1:]:
        mouse(canvas, "move", p, mods)
    app.processEvents()
    if shot:
        canvas.window().grab().save(shot)
    mouse(canvas, "release", pts[-1], mods)
    app.processEvents()


def ring(cx, cy, rx, ry, n=40):
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n)]


def main():
    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = MainWindow()
    win.resize(1500, 880)
    win.show()
    wait(app, win.ready)
    win.auto_check.setChecked(False)
    win.open_image(str(paths.IMAGES_DIR / NAME))
    wait(app, lambda: win.session is not None and win.session.prior is not None)
    s = win.session
    ann = {a.name: a for a in parse_annotations(paths.ANNOTATIONS_XML)}[NAME]
    wins = [o.mask for o in image_objects(ann, 100) if o.label == 4]
    c = np.array([np.argwhere(m).mean(0) for m in wins])          # (y, x)
    k = int(np.argmin(np.hypot(*(c - c.mean(0)).T)))
    near = np.argsort(np.hypot(*(c - c[k]).T))[:4]
    union = np.any([wins[j] for j in near], axis=0)
    ys, xs = np.nonzero(union)
    cx, cy, rx, ry = xs.mean(), ys.mean(), (xs.max() - xs.min()) * 0.65 + 10, (ys.max() - ys.min()) * 0.65 + 10
    one = wins[k]
    oy, ox = np.nonzero(one)
    draw(app, win.canvas, ring(ox.mean(), oy.mean(), (ox.ptp() / 2) * 1.3 + 4, (oy.ptp() / 2) * 1.3 + 4))
    o = s.objects[-1]
    iou = (o.mask & one).sum() / (o.mask | one).sum()
    print(f"loop around one window -> {CLASS_NAMES[o.label]}, IoU with it {iou:.2f} | {win.msg.text()}")
    draw(app, win.canvas, ring(cx, cy, rx, ry), QtCore.Qt.ControlModifier, shot="artifacts/figures/app_lasso.png")
    print(f"Cmd+loop around {len(near)} GT windows -> {len(s.objects) - 1} objects: "
          f"{[CLASS_NAMES[o.label] for o in s.objects[1:]]} | status: {win.msg.text()}")
    n = len(s.objects)
    y0 = int(ys.max() + 2 * ry)
    draw(app, win.canvas, [(cx - rx, min(y0, s.h - 5)), (cx, min(y0, s.h - 5)), (cx + rx, min(y0, s.h - 5))])
    print(f"scribble -> {len(s.objects) - n} new object: {CLASS_NAMES[s.objects[-1].label]} | {win.msg.text()}")
    a0 = int(s.objects[-1].mask.sum())
    draw(app, win.canvas, ring(cx + rx, y0, 25, 25), QtCore.Qt.ShiftModifier)
    a1 = int(s.objects[-1].mask.sum())
    draw(app, win.canvas, ring(cx + rx, y0, 25, 25), QtCore.Qt.AltModifier)
    a2 = int(s.objects[-1].mask.sum())
    print(f"Shift+loop: {a0} -> {a1} px; Option+loop: -> {a2} px | {win.msg.text()}")
    win.grab().save("artifacts/figures/app_lasso_result.png")
    s.undo(); s.undo()
    print("two undos restore the scribble object:", int(s.objects[-1].mask.sum()) == a0)
    # pre-label, then an Option+click on a big auto object cuts only a piece
    win.run_prelabel()                                   # in the background now
    wait(app, lambda: s.prelabeled)
    big = max((i for i, o in enumerate(s.objects) if o.auto), key=lambda i: s.objects[i].mask.sum())
    m = s.objects[big].mask
    yy, xx = np.nonzero(m)
    j = len(yy) // 2
    before = int(m.sum())
    win.on_click(int(xx[j]), int(yy[j]), QtCore.Qt.LeftButton, QtCore.Qt.AltModifier)
    print(f"Option+click on auto object #{big + 1} ({CLASS_NAMES[s.objects[big].label]}): "
          f"{before} -> {int(s.objects[big].mask.sum())} px | {win.msg.text()}")
    n = len(s.objects)
    win.refresh(save=False)
    win.select_tiny(limit=500)
    k = len(win.object_list.selectedItems())
    win.delete_active()
    print(f"select < 500 px + Delete: {n} -> {len(s.objects)} objects ({k} selected)")


if __name__ == "__main__":
    main()
