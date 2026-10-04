"""The toy slot machine: a pull spins the reels to the drawn result; three of a
kind counts a jackpot and throws confetti; pulls during a spin are ignored."""
import os
import random
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

QtCore = pytest.importorskip("PySide6.QtCore")
from PySide6 import QtWidgets  # noqa: E402

from dmgseg.app.slots import SlotMachine  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def spin(app, sm, outcome=None, timeout=5):
    got = []
    sm.finished.connect(got.append)
    assert sm.pull_handle(outcome)
    assert not sm.pull_handle()                   # ignored while the handle / reels move
    t0 = time.perf_counter()
    while not got and time.perf_counter() - t0 < timeout:
        app.processEvents()
        time.sleep(0.005)
    sm.finished.disconnect()
    return got[0], time.perf_counter() - t0


def test_jackpot_pair_and_random(app):
    win = QtWidgets.QWidget()
    win.resize(400, 400)
    sm = SlotMachine(win, rng=random.Random(1))
    sm.resize(200, sm.heightForWidth(200))
    win.show()
    res, seconds = spin(app, sm, (4, 4, 4))
    assert res == (4, 4, 4) and sm.jackpots == 1 and sm.confetti is not None and sm.confetti.parts
    assert seconds < 2.6                          # a short spin
    res, _ = spin(app, sm, (2, 3, 2))
    assert res == (2, 3, 2) and sm.jackpots == 1 and sm.msg == "So close!"
    for _ in range(3):
        res, _ = spin(app, sm)
        assert all(r in (0, 1, 2, 3, 4, 5) for r in res)
    assert sm.spins == 5
