"""Damage Annotator: a classic-Windows-style desktop app for click annotation.

    python -m dmgseg.app

Mouse (on the image):
    left click            new object (SAM mask; class and mask chosen by the class head)
    right click / Ctrl+click   next class for the object under the cursor
    Shift+click           grow the active object (positive refinement)
    Alt(Option)+click     shrink the active object (negative refinement)
    wheel                 zoom, middle button / Space+drag: pan
Keys: 1-5 set the class of the active object, 0 marks it as Other (e.g. a tree in
front of the building: it is cut out of what is behind), Delete/Backspace removes
it, Ctrl+Z undo.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from dmgseg.app.engine import COLORS, ClassHead, Session
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.tool.assign import CLICKABLE

TITLE = "Damage Annotator"
GRAY, DARK, NAVY = "#c0c0c0", "#808080", "#000080"


# ----------------------------------------------------------------- look & feel
def apply_classic_style(app):
    """Windows 95/98 look: Qt's built-in 'Windows' style + the classic gray palette."""
    app.setStyle(QtWidgets.QStyleFactory.create("Windows"))
    pal = QtGui.QPalette()
    c = QtGui.QColor
    for role, color in [
        (QtGui.QPalette.Window, GRAY), (QtGui.QPalette.Button, GRAY), (QtGui.QPalette.Base, "#ffffff"),
        (QtGui.QPalette.AlternateBase, "#e0e0e0"), (QtGui.QPalette.WindowText, "#000000"),
        (QtGui.QPalette.ButtonText, "#000000"), (QtGui.QPalette.Text, "#000000"),
        (QtGui.QPalette.Highlight, NAVY), (QtGui.QPalette.HighlightedText, "#ffffff"),
        (QtGui.QPalette.Light, "#ffffff"), (QtGui.QPalette.Midlight, "#dfdfdf"),
        (QtGui.QPalette.Mid, DARK), (QtGui.QPalette.Dark, DARK), (QtGui.QPalette.Shadow, "#000000"),
        (QtGui.QPalette.ToolTipBase, "#ffffe1"), (QtGui.QPalette.ToolTipText, "#000000"),
    ]:
        pal.setColor(role, c(color))
    app.setPalette(pal)
    families = set(QtGui.QFontDatabase.families())
    for name in ("MS Sans Serif", "Microsoft Sans Serif", "Tahoma", "Verdana", "Geneva"):
        if name in families:
            app.setFont(QtGui.QFont(name, 11))
            break
    app.setStyleSheet("""
        QStatusBar::item { border: 1px solid; border-color: #808080 #ffffff #ffffff #808080; }
        QGroupBox { font-weight: bold; }
        QListWidget { border: 2px inset #808080; }
    """)


class CaptionBar(QtWidgets.QLabel):
    """A Windows 98 style title strip inside the window (navy gradient)."""

    def __init__(self, text):
        super().__init__("  " + text)
        self.setFixedHeight(22)
        self.setStyleSheet("color: white; font-weight: bold; padding-left: 4px;"
                           "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000080, stop:1 #1084d0);")


# ----------------------------------------------------------------- workers
class Worker(QtCore.QObject):
    """Runs a function in a QThread; emits done(result) or failed(message)."""
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(self.fn(*self.args, progress=self.progress.emit))
        except Exception as e:  # shown in the status bar
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class Models:
    """SAM, the class head and the semantic prior, loaded once in the background."""

    def __init__(self):
        self.clicker = self.head = self.prior_model = self.embed_model = None

    def load(self, progress=None):
        from dmgseg import hub, paths
        from dmgseg.prior.embedding import load_embedding_model
        from dmgseg.prior.unet_dinov2 import load_prior_model
        from dmgseg.sam.predictor import SamClicker
        steps = 4
        self.clicker = SamClicker("small", "cpu")
        progress and progress(1, steps)
        head = paths.ARTIFACTS / "classhead" / "head_a.pt"
        if not head.exists():
            hub.download_if_exists("classhead/head_a.pt", paths.ARTIFACTS)
        self.head = ClassHead(head)
        progress and progress(2, steps)
        self.prior_model = load_prior_model(hub.prior_weights(), "cpu")
        progress and progress(3, steps)
        self.embed_model = load_embedding_model("cpu")
        progress and progress(4, steps)
        return self

    def compute_prior(self, image, progress=None):
        from dmgseg.prior.infer import predict_probs
        probs, _ = predict_probs(self.prior_model, self.embed_model, Image.fromarray(image),
                                 batch_size=2, progress=progress)
        return probs


# ----------------------------------------------------------------- canvas
class Canvas(QtWidgets.QGraphicsView):
    clicked = QtCore.Signal(int, int, object, object)   # x, y, button, modifiers
    moved = QtCore.Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = self.scene().addPixmap(QtGui.QPixmap())
        self.setBackgroundBrush(QtGui.QColor(DARK))
        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._pan = None

    def show_rgb(self, rgb, fit=False):
        h, w = rgb.shape[:2]
        img = QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.item.setPixmap(QtGui.QPixmap.fromImage(img))
        self.scene().setSceneRect(0, 0, w, h)
        if fit:
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def _image_xy(self, e):
        p = self.mapToScene(e.position().toPoint())
        return int(p.x()), int(p.y())

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MiddleButton or (e.button() == QtCore.Qt.LeftButton and
                                                     QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.MetaModifier):
            self._pan = e.position()
            return
        x, y = self._image_xy(e)
        r = self.item.pixmap().rect()
        if 0 <= x < r.width() and 0 <= y < r.height():
            self.clicked.emit(x, y, e.button(), e.modifiers())

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        self.moved.emit(*self._image_xy(e))

    def mouseReleaseEvent(self, e):
        self._pan = None


def swatch(class_id, size=12):
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor(*COLORS[class_id]))
    p = QtGui.QPainter(pm)
    p.setPen(QtGui.QColor("#000000"))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QtGui.QIcon(pm)


# ----------------------------------------------------------------- main window
class MainWindow(QtWidgets.QMainWindow):
    # Worker threads never touch widgets: they emit _ui(fn, arg) and the GUI thread
    # runs fn(arg) (a plain lambda connected to a worker signal would run in the
    # worker's thread).
    _ui = QtCore.Signal(object, object)

    def __init__(self, models=None):
        super().__init__()
        self._ui.connect(lambda fn, arg: fn(*arg) if isinstance(arg, tuple) else fn(arg))
        self.setWindowTitle(TITLE)
        self.resize(1280, 820)
        self.models = models
        self.session = None
        self.image_path = None
        self.alpha = 0.45
        self.show_prior = False
        self._threads = []

        self.canvas = Canvas()
        self.canvas.clicked.connect(self.on_click)
        self.canvas.moved.connect(lambda x, y: self.pos_label.setText(f" x={x} y={y} "))

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        split = QtWidgets.QSplitter()
        split.addWidget(self.canvas)
        split.addWidget(self._side_panel())
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 260])
        lay.addWidget(split, 1)
        self.setCentralWidget(central)

        self._menus()
        self._toolbar()
        self._statusbar()
        self.setAcceptDrops(True)
        self.refresh()
        if self.models is None:
            self.models = Models()
            self.status("Loading models (SAM, class head, prior)...")
            self.run_bg(self.models.load, on_done=lambda m: self.status("Ready. File > Open Image..."),
                        progress=lambda d, t: self.progress.setValue(int(100 * d / t)))

    # -- layout pieces ---------------------------------------------------------
    def _side_panel(self):
        side = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(4, 0, 0, 0)
        box = QtWidgets.QGroupBox("Classes")
        bl = QtWidgets.QVBoxLayout(box)
        self.class_list = QtWidgets.QListWidget()
        self.class_list.setFixedHeight(6 * 22 + 8)
        self.class_list.itemClicked.connect(self.on_class_item)
        bl.addWidget(self.class_list)
        v.addWidget(box)
        box2 = QtWidgets.QGroupBox("Objects")
        bl2 = QtWidgets.QVBoxLayout(box2)
        self.object_list = QtWidgets.QListWidget()
        self.object_list.currentRowChanged.connect(self.on_object_selected)
        bl2.addWidget(self.object_list)
        v.addWidget(box2, 1)
        hint = QtWidgets.QLabel("Left click: new object\nRight click: next class\nShift+click: grow\n"
                                "Option+click: shrink\n1-5: set class, 0: Other\n⌫ (Backspace): delete")
        hint.setFrameShape(QtWidgets.QFrame.Panel)
        hint.setFrameShadow(QtWidgets.QFrame.Sunken)
        v.addWidget(hint)
        return side

    def _menus(self):
        # Title strip above the menu bar, as in Windows 98 (the menu is kept inside
        # the window instead of macOS's global menu bar).
        mb = QtWidgets.QMenuBar()
        mb.setNativeMenuBar(False)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(2, 2, 2, 0)
        tl.setSpacing(0)
        self.caption = CaptionBar(f"{TITLE} - no image")
        tl.addWidget(self.caption)
        tl.addWidget(mb)
        self.setMenuWidget(top)
        f = mb.addMenu("&File")
        self.act_open = f.addAction("&Open Image...", self.open_dialog, QtGui.QKeySequence.Open)
        self.act_save = f.addAction("&Export Mask...", self.export_dialog, QtGui.QKeySequence.Save)
        f.addSeparator()
        f.addAction("E&xit", self.close, QtGui.QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        self.act_undo = e.addAction("&Undo", self.undo, QtGui.QKeySequence.Undo)
        act_del = e.addAction("&Delete Object", self.delete_active)
        # On a Mac keyboard the big delete key is Backspace; Delete is Fn+Backspace.
        act_del.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_Backspace), QtGui.QKeySequence.Delete])
        vmenu = mb.addMenu("&View")
        vmenu.addAction("Fit to &Window", lambda: self.canvas.fitInView(self.canvas.item, QtCore.Qt.KeepAspectRatio), "F")
        self.act_prior = vmenu.addAction("Show &Prior Map", self.toggle_prior, "P")
        self.act_prior.setCheckable(True)
        h = mb.addMenu("&Help")
        h.addAction("&Shortcuts", lambda: QtWidgets.QMessageBox.information(self, "Shortcuts", __doc__.split("Mouse")[1]))
        h.addAction("&About", lambda: QtWidgets.QMessageBox.about(
            self, "About", f"<b>{TITLE}</b><br>SAM 2.1 + DINOv2 prior + class head.<br>Lab work, KNU 2026."))

    def _toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        st = self.style()
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Open", self.open_dialog)
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Export", self.export_dialog)
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_ArrowBack), "Undo", self.undo)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Color strength: "))
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(self.alpha * 100))
        s.setFixedWidth(120)
        tip = "How strongly the class colors cover the photo (0 % = photo only, 100 % = colors only)"
        s.setToolTip(tip)
        pct = QtWidgets.QLabel(f" {int(self.alpha * 100)} % ")
        s.valueChanged.connect(lambda v: (setattr(self, "alpha", v / 100), pct.setText(f" {v} % "), self.redraw()))
        tb.addWidget(s)
        tb.addWidget(pct)
        tb.addSeparator()
        cb = QtWidgets.QCheckBox("Prior map")
        cb.toggled.connect(lambda on: (self.act_prior.setChecked(on), self.toggle_prior()))
        self.prior_check = cb
        tb.addWidget(cb)

    def _statusbar(self):
        sb = self.statusBar()
        self.msg = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(True)
        self.prior_label = QtWidgets.QLabel(" Prior: - ")
        self.time_label = QtWidgets.QLabel(" - ms ")
        self.pos_label = QtWidgets.QLabel(" x=- y=- ")
        sb.addWidget(self.msg, 1)
        for w in (self.prior_label, self.progress, self.time_label, self.pos_label):
            sb.addPermanentWidget(w)

    # -- helpers ---------------------------------------------------------------
    def status(self, text):
        self.msg.setText(" " + text)

    def run_bg(self, fn, *args, on_done=None, progress=None):
        thread = QtCore.QThread(self)
        worker = Worker(fn, *args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        if on_done:
            worker.done.connect(lambda r: self._ui.emit(on_done, r))
        worker.failed.connect(lambda m: self._ui.emit(lambda msg: self.status("Error: " + msg), m))
        if progress:
            worker.progress.connect(lambda d, t: self._ui.emit(progress, (d, t)))
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda: self._threads.remove((thread, worker)))
        self._threads.append((thread, worker))
        thread.start()
        return thread

    def ready(self):
        """All models loaded (SAM loads first, the prior's embedding model last)."""
        return self.models is not None and self.models.embed_model is not None

    # -- image -----------------------------------------------------------------
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.open_image(path)

    def open_image(self, path):
        if not self.ready():
            self.status("Models are still loading, please wait...")
            return
        self.image_path = Path(path)
        image = np.asarray(Image.open(path).convert("RGB"))
        self.status(f"Reading image with SAM: {self.image_path.name}")
        self.caption.setText(f"  {TITLE} - {self.image_path.name}")
        self.canvas.show_rgb(image, fit=True)

        def make_session(progress=None):
            return Session(image, self.models.clicker, self.models.head)

        def got_session(session):
            self.session = session
            self.status("Ready: click objects (the class appears when the prior is done)")
            self.prior_label.setText(" Prior: running ")
            self.refresh()
            t0 = time.time()
            self.run_bg(self.models.compute_prior, image,
                        on_done=lambda p: self.got_prior(p, time.time() - t0),
                        progress=lambda d, t: self.progress.setValue(int(100 * d / t)))

        self.progress.setValue(0)
        self.run_bg(make_session, on_done=got_session)

    def got_prior(self, prior, seconds):
        if self.session is None:
            return
        self.session.set_prior(prior)
        self.prior_label.setText(f" Prior: done ({seconds:.0f} s) ")
        self.progress.setValue(100)
        self.status("Prior ready: classes assigned")
        self.refresh()

    # -- interaction -------------------------------------------------------------
    def on_click(self, x, y, button, mods):
        s = self.session
        if s is None:
            return
        t0 = time.time()
        right = button == QtCore.Qt.RightButton or (button == QtCore.Qt.LeftButton and mods & QtCore.Qt.ControlModifier)
        if right:
            i = s.object_at(x, y)
            if i is None:
                self.status("Right click: no object here")
                return
            s.next_class(i)
            self.status(f"Object #{i + 1}: {CLASS_NAMES[s.objects[i].label]}")
        elif mods & QtCore.Qt.ShiftModifier and s.active is not None:
            s.refine(x, y, True)
            self.status("Grew the active object")
        elif mods & QtCore.Qt.AltModifier and s.active is not None:
            s.refine(x, y, False)
            self.status("Shrank the active object")
        else:
            i = s.new_object(x, y)
            o = s.objects[i]
            self.status(f"Object #{i + 1}: " + ("class pending (prior running)" if o.pending
                                               else CLASS_NAMES[o.label]))
        self.time_label.setText(f" {1000 * (time.time() - t0):.0f} ms ")
        self.refresh()

    def keyPressEvent(self, e):
        s = self.session
        key = e.key()
        if s is not None and s.active is not None and QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_5:
            s.set_class(s.active, CLICKABLE[key - QtCore.Qt.Key_1])
            self.refresh()
        elif s is not None and s.active is not None and key == QtCore.Qt.Key_0:
            s.set_class(s.active, 0)
            self.status(f"Object #{s.active + 1} marked as Other (cut out of what is behind it)")
            self.refresh()
        else:
            super().keyPressEvent(e)

    def on_class_item(self, item):
        s = self.session
        if s is not None and s.active is not None:
            s.set_class(s.active, item.data(QtCore.Qt.UserRole))
            self.refresh()

    def on_object_selected(self, row):
        if self.session is not None and 0 <= row < len(self.session.objects) and row != self.session.active:
            self.session.active = row
            self.redraw()

    def undo(self):
        if self.session:
            self.session.undo()
            self.refresh()

    def delete_active(self):
        if self.session and self.session.active is not None:
            self.session.delete(self.session.active)
            self.refresh()

    def toggle_prior(self):
        self.show_prior = self.act_prior.isChecked()
        self.prior_check.blockSignals(True)
        self.prior_check.setChecked(self.show_prior)
        self.prior_check.blockSignals(False)
        self.redraw()

    def export_dialog(self):
        if self.session is None:
            return
        default = str(self.image_path.with_name(self.image_path.stem + "_mask.png"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Mask", default, "PNG (*.png)")
        if path:
            png, js = self.session.export(path, source_name=self.image_path.name)
            self.status(f"Saved {Path(png).name} and {Path(js).name}")

    # -- drawing -----------------------------------------------------------------
    def redraw(self):
        s = self.session
        if s is None:
            return
        if self.show_prior and s.prior is not None:
            colors = np.array([COLORS[c] for c in range(6)], np.float32)
            rgb = (0.4 * s.image + 0.6 * colors[s.prior.argmax(-1)]).astype(np.uint8)
        else:
            rgb = s.overlay(alpha=self.alpha)
        self.canvas.show_rgb(rgb)

    def refresh(self):
        s = self.session
        self.class_list.clear()
        counts = s.counts() if s else {c: 0 for c in list(CLICKABLE) + [0]}
        for key, c in [(n, c) for n, c in enumerate(CLICKABLE, start=1)] + [(0, 0)]:
            it = QtWidgets.QListWidgetItem(swatch(c), f"{key}  {CLASS_NAMES[c]:<14} {counts[c]:>4}")
            it.setData(QtCore.Qt.UserRole, c)
            self.class_list.addItem(it)
        self.object_list.blockSignals(True)
        self.object_list.clear()
        if s:
            for i, o in enumerate(s.objects):
                name = "(pending)" if o.pending else CLASS_NAMES[o.label]
                flag = " *" if o.manual is not None else ""
                it = QtWidgets.QListWidgetItem(swatch(o.label if o.label is not None else 0),
                                               f"#{i + 1}  {name}{flag}  ({int(o.mask.sum())} px)")
                self.object_list.addItem(it)
            if s.active is not None:
                self.object_list.setCurrentRow(s.active)
        self.object_list.blockSignals(False)
        self.redraw()


def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName(TITLE)
    apply_classic_style(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        QtCore.QTimer.singleShot(0, lambda: _open_when_ready(win, sys.argv[1]))
    return app.exec()


def _open_when_ready(win, path):
    if win.ready():
        win.open_image(path)
    else:
        QtCore.QTimer.singleShot(500, lambda: _open_when_ready(win, path))


if __name__ == "__main__":
    sys.exit(main())
