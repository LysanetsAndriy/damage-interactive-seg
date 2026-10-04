"""Frames of the toy slot machine for a quick visual check (offscreen):
idle, handle down, spinning, reels stopping, jackpot confetti.
Saves artifacts/figures/slots_frames.png."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets  # noqa: E402

from dmgseg.app.gui import apply_classic_style  # noqa: E402
from dmgseg.app.slots import SlotMachine  # noqa: E402


def run_until(app, t_end):
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def main():
    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = QtWidgets.QMainWindow()
    split = QtWidgets.QSplitter()
    left = QtWidgets.QWidget()
    lv = QtWidgets.QVBoxLayout(left)
    imgs = QtWidgets.QGroupBox("Images")
    il = QtWidgets.QVBoxLayout(imgs)
    lst = QtWidgets.QListWidget()
    for n in ["✓126  A70EA485-003E.jpg", "✓ 56  image.jpg", "✓ 30  images.jpeg", " •    okishIDdukZPW7MV.jpg"]:
        lst.addItem(n)
    il.addWidget(lst)
    lv.addWidget(imgs, 1)
    box = QtWidgets.QGroupBox("Lucky Labels")
    bl = QtWidgets.QVBoxLayout(box)
    sm = SlotMachine()
    bl.addWidget(sm)
    lv.addWidget(box)
    canvas = QtWidgets.QLabel()
    canvas.setStyleSheet("background:#808080")
    split.addWidget(left)
    split.addWidget(canvas)
    split.setSizes([215, 500])
    win.setCentralWidget(split)
    win.resize(720, 640)
    win.show()
    app.processEvents()
    frames = [("idle", win.grab())]
    t0 = time.perf_counter()
    sm.pull_handle(outcome=(4, 4, 4))
    for name, t in [("handle down", .2), ("spinning", .45), ("reels stopping", 1.2), ("jackpot", 2.15),
                    ("confetti", 2.5)]:
        run_until(app, t0 + t)
        frames.append((name, win.grab()))
    # crop the left part of each frame, side by side
    w, h = 300, 640
    sheet = QtGui.QPixmap(w * len(frames), h + 22)
    sheet.fill(QtGui.QColor("white"))
    p = QtGui.QPainter(sheet)
    for i, (name, pm) in enumerate(frames):
        p.drawPixmap(i * w, 22, pm.copy(0, 0, w, h))
        p.drawText(i * w + 6, 16, name)
    p.end()
    sheet.save("artifacts/figures/slots_frames.png")
    print("result", tuple(r.result() for r in sm.reels), "jackpots", sm.jackpots, "msg", sm.msg)


if __name__ == "__main__":
    main()
