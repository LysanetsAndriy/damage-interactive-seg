"""Canvas input: wheel scrolls, Ctrl(Cmd)+wheel zooms, Shift+wheel scrolls sideways,
Space+drag pans without drawing (offscreen Qt, real events)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

QtCore = pytest.importorskip("PySide6.QtCore")
from PySide6 import QtGui, QtWidgets  # noqa: E402

from dmgseg.app.gui import Canvas  # noqa: E402

Qt = QtCore.Qt


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wheel(c, dy=0, dx=0, mods=Qt.NoModifier):
    pos = QtCore.QPointF(c.viewport().width() / 2, c.viewport().height() / 2)
    e = QtGui.QWheelEvent(pos, c.viewport().mapToGlobal(pos), QtCore.QPoint(0, 0), QtCore.QPoint(dx, dy),
                          Qt.NoButton, mods, Qt.NoScrollPhase, False)
    c.wheelEvent(e)


def state(c):
    return round(c.transform().m11(), 4), c.horizontalScrollBar().value(), c.verticalScrollBar().value()


def test_wheel_scrolls_ctrl_wheel_zooms_space_pans(app):
    c = Canvas()
    c.resize(400, 300)
    c.show()
    c.show_rgb(np.zeros((1000, 1500, 3), np.uint8))
    c.zoom(1.0 / c.transform().m11(), at_cursor=False)       # 100 %: the image is bigger than the view
    app.processEvents()
    z0, h0, v0 = state(c)
    wheel(c, dy=-120)                                         # wheel down: scroll, no zoom
    z1, h1, v1 = state(c)
    assert z1 == z0 and v1 > v0 and h1 == h0
    wheel(c, dy=-120, mods=Qt.ShiftModifier)                  # Shift+wheel: sideways
    z2, h2, v2 = state(c)
    assert z2 == z0 and h2 > h1 and v2 == v1
    wheel(c, dy=120, mods=Qt.ControlModifier)                 # Ctrl(Cmd)+wheel: zoom in
    assert state(c)[0] > z0
    # Space + left drag pans and does not draw
    drawn = []
    c.drawn.connect(lambda *a: drawn.append(a))
    c.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier))
    before = state(c)
    for kind, x in (("press", 200), ("move", 150), ("move", 100), ("release", 100)):
        t = {"press": QtCore.QEvent.MouseButtonPress, "move": QtCore.QEvent.MouseMove,
             "release": QtCore.QEvent.MouseButtonRelease}[kind]
        p = QtCore.QPointF(x, 150)
        e = QtGui.QMouseEvent(t, p, c.viewport().mapToGlobal(p.toPoint()),
                              Qt.NoButton if kind == "move" else Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        getattr(c, {"press": "mousePressEvent", "move": "mouseMoveEvent", "release": "mouseReleaseEvent"}[kind])(e)
    c.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier))
    assert state(c)[1] > before[1] and not drawn
    c.zoom(1000)                                              # clamped
    assert c.transform().m11() <= Canvas.MAX_ZOOM + 1e-6
