"""Headless check of the Auto pre-label box: draft on open; untick removes the
untouched draft (own objects stay, clicks still classified); tick drafts again;
untick while drafting drops the late result; Ctrl+Z restores a removed draft."""
import functools
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
print = functools.partial(print, flush=True)  # noqa: A001

from PySide6 import QtCore, QtWidgets  # noqa: E402

from dmgseg import paths  # noqa: E402
from dmgseg.app.gui import MainWindow, apply_classic_style  # noqa: E402
from dmgseg.data.cvat import CLASS_NAMES  # noqa: E402

NAME = "download29_1.png"


def wait(app, cond, timeout=600):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


def pause(app, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def main():
    app = QtWidgets.QApplication([])
    apply_classic_style(app)
    win = MainWindow()
    win.show()
    wait(app, win.ready)
    win.open_image(str(paths.IMAGES_DIR / NAME))
    wait(app, lambda: win.session is not None and win.session.prelabeled)
    s = win.session
    print(f"1. opened with the box on: {s.draft_count()} draft objects")
    win.on_click(300, 300, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    mine = s.objects[-1]
    print(f"2. own click -> {CLASS_NAMES[mine.label]} (auto={mine.auto})")
    win.auto_check.setChecked(False)
    print(f"3. untick: {s.draft_count()} draft objects left, {len(s.objects)} objects "
          f"(own kept: {mine in s.objects}) | {win.msg.text()}")
    win.on_click(600, 350, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    print(f"4. click with the box off -> {CLASS_NAMES[s.objects[-1].label]} (class from DINO + head)")
    win.auto_check.setChecked(True)
    busy = win.progress.maximum() == 0
    wait(app, lambda: win.session.draft_count() > 0)
    print(f"5. tick: draft back with {s.draft_count()} objects (progress bar moving while drafting: {busy}); "
          f"own objects still there: {sum(not o.auto for o in s.objects)}")
    win.auto_check.setChecked(False)
    win.auto_check.setChecked(True)
    win.auto_check.setChecked(False)          # untick while the new draft is being computed
    pause(app, 8)
    print(f"6. tick+untick while drafting, 8 s later: {s.draft_count()} draft objects (should be 0)")
    win.undo()
    print(f"7. Ctrl+Z after removing: {s.draft_count()} draft objects restored")


if __name__ == "__main__":
    main()
