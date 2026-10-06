"""App icon (.icns) drawn with the app's own class icons: a Windows-98-grey tile with a
navy title bar and a cracked window. -> packaging/build/icon.icns"""
import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from dmgseg.app.slots import draw_symbol

OUT = Path(__file__).parent / "build"


def render(size):
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    m = size * 0.06
    tile = QtCore.QRectF(m, m, size - 2 * m, size - 2 * m)
    p.setPen(QtGui.QPen(QtGui.QColor("#404040"), max(1, size / 128)))
    p.setBrush(QtGui.QColor("#c0c0c0"))
    p.drawRoundedRect(tile, size * 0.12, size * 0.12)
    bar = QtCore.QRectF(tile.x() + size * 0.05, tile.y() + size * 0.05, tile.width() - size * 0.1, size * 0.13)
    p.setBrush(QtGui.QColor("#000080"))
    p.setPen(QtCore.Qt.NoPen)
    p.drawRect(bar)
    body = QtCore.QRectF(tile.x() + size * 0.1, bar.bottom() + size * 0.04, tile.width() - size * 0.2,
                         tile.bottom() - bar.bottom() - size * 0.12)
    draw_symbol(p, 4, body)                         # the "Broken Window" icon
    p.end()
    return img


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    iconset = OUT / "icon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for s in (16, 32, 128, 256, 512):
        render(s).save(str(iconset / f"icon_{s}x{s}.png"))
        render(2 * s).save(str(iconset / f"icon_{s}x{s}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT / "icon.icns")], check=True)
    render(512).save(str(OUT / "icon.png"))
    print("icon:", OUT / "icon.icns")


if __name__ == "__main__":
    main()
