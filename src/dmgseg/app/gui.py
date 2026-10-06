"""Damage Annotator: a classic-Windows-style desktop app for click annotation.

    python -m dmgseg.app                    # then File > Open Folder (or Open Image)
    python -m dmgseg.app path/to/folder     # open a folder directly

Folder mode: work is saved automatically to <folder>/_damage_annotator/ (also on
switching images and on quitting); priors are pre-computed in the background for
the next images; Page Up / Page Down = previous / next image; File > Export CVAT XML
writes the whole folder in CVAT for images 1.1 format.

Shortcuts: see shortcuts_text() (Help > Shortcuts); they use Cmd/Option on macOS and
Ctrl/Alt on Windows and Linux (Qt maps Ctrl to Cmd on a Mac).
"""
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from dmgseg.app.engine import COLORS, ClassHead, Session, blend_labels
from dmgseg.app.project import FolderProject, export_cvat, visible_objects_from_state
from dmgseg.app.slots import SlotMachine
from dmgseg.data.cvat import CLASS_NAMES
from dmgseg.tool.assign import CLICKABLE

TITLE = "Damage Annotator"
MAC = sys.platform == "darwin"
CTRL = "Cmd" if MAC else "Ctrl"      # Qt's ControlModifier is the Cmd key on a Mac
ALT = "Option" if MAC else "Alt"


def shortcuts_text():
    return f"""Mouse on the image
  left click                new object (SAM mask; class from the class head)
  right click / {CTRL}+click   next class of the object under the cursor
  Shift+click               grow the selected object
  {ALT}+click / Shift+right click   shrink (cut a piece from) the object under the cursor
  draw a loop               the object inside it
  {CTRL}+loop                  every object the prior finds inside
  draw a line               one object along the line
  Shift / {ALT} + loop        add / cut exactly the drawn area
  Shift / {ALT} + line        grow / shrink the selected object along the line

View
  wheel / two-finger swipe  scroll (Shift+wheel: left-right)
  {CTRL}+wheel, pinch         zoom at the cursor
  {CTRL}+plus / minus / 0     zoom in / out / 100 %;  F: fit to window
  Space+drag, middle drag   pan;  P: show the prior map
  U                         spotlight the areas where the prior is least sure
  H                         hover preview: outline of what a click would select

Keys
  1-5   class of the selected objects, 0: Other (cut out of what is behind)
  Backspace / Delete   delete the selected objects;  {CTRL}+Z undo
  Page Up / Page Down, {CTRL}+Left / Right   previous / next image (folder)
  {CTRL}+S save, {CTRL}+E export mask, {CTRL}+Shift+E export CVAT
  {CTRL}+L draft now, {CTRL}+Shift+L remove the untouched draft

Auto pre-label box: on = draft objects from the DINO map (now and for each image);
off = remove the untouched draft and label by hand (classes still come from DINO).
Draft objects you edit or re-classify become yours and are never removed.

On pre-label objects Shift/{ALT} clicks add/remove only the piece under the cursor.
Linux: if the desktop uses {ALT}+drag to move windows, use Shift+right click to shrink."""
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

    FEATURE_CACHE = 6                    # images whose SAM features stay in memory (~16 MB each)

    def __init__(self):
        self.clicker = self.head = self.prior_model = self.embed_model = None
        self._features = OrderedDict()   # image path -> SAM features
        self._cache_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self.draft_lock = threading.Lock()

    def cached_features(self, key):
        with self._cache_lock:
            f = self._features.get(key)
            if f is not None:
                self._features.move_to_end(key)
            return f

    def sam_features(self, key, image):
        """SAM features of an image: from memory, or computed by the background
        encoder (a twin of the user's clicker, so it never disturbs the user)."""
        f = self.cached_features(key)
        if f is not None:
            return f
        with self._encode_lock:
            f = self.cached_features(key)          # another thread may have just made it
            if f is None:
                self.encoder.set_image(image)
                f = self.encoder.features()
                with self._cache_lock:
                    self._features[key] = f
                    while len(self._features) > self.FEATURE_CACHE:
                        self._features.popitem(last=False)
        return f

    def load(self, progress=None):
        from dmgseg import hub, paths
        from dmgseg.prior.embedding import load_embedding_model
        from dmgseg.prior.unet_dinov2 import load_prior_model
        from dmgseg.sam.predictor import SamClicker
        steps = 4
        sam_ft = paths.ARTIFACTS / "sam" / "finetuned_decoder.pt"
        if not sam_ft.exists():
            hub.download_if_exists("sam/finetuned_decoder.pt", paths.ARTIFACTS)
        # fine-tuned SAM decoder (27 % fewer clicks to 85 % IoU); zero-shot if absent
        self.clicker = SamClicker("small", "cpu", decoder_weights=sam_ft if sam_ft.exists() else None)
        self.sam_finetuned = sam_ft.exists()
        self.encoder = self.clicker.twin()   # reads images in the background
        self.drafter = self.clicker.twin()   # makes the automatic draft in the background
        self.hover = self.clicker.twin()     # hover preview (background thread)
        progress and progress(1, steps)
        # head A trained on masks of the same SAM the app uses
        name = "head_a_samft.pt" if self.sam_finetuned else "head_a.pt"
        head = paths.ARTIFACTS / "classhead" / name
        if not head.exists():
            hub.download_if_exists(f"classhead/{name}", paths.ARTIFACTS)
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


class PriorPrefetcher:
    """One background thread that prepares a folder, in this order:
        1. the DINOv2 prior of the current image (cached on disk),
        2. SAM's reading of the next and previous images (kept in memory), so
           switching to them is instant,
        3. the priors of the other images, starting after the current one.
    Callbacks run in this thread; the window forwards them to the GUI thread."""

    def __init__(self, models, project, on_done, on_progress):
        self.models, self.project = models, project
        self.on_done, self.on_progress = on_done, on_progress
        self.current = None
        self.failed = set()
        self.cv = threading.Condition()
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, name="prior_prefetch", daemon=True)
        self.thread.start()

    def set_current(self, name):
        with self.cv:
            self.current = name
            self.cv.notify()

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify()

    def pending(self):
        return sum(1 for n in self.project.images if not self.project.has_prior(n) and n not in self.failed)

    def _next(self):
        """-> ("prior" | "sam", name) or None."""
        todo = lambda n: not self.project.has_prior(n) and n not in self.failed
        imgs = self.project.images
        cur = self.current if self.current in imgs else None
        if cur and todo(cur):
            return "prior", cur
        if cur:
            i = imgs.index(cur)
            for j in (i + 1, i - 1):
                if 0 <= j < len(imgs) and self.models.cached_features(str(self.project.path(imgs[j]))) is None \
                        and imgs[j] not in self.failed:
                    return "sam", imgs[j]
        start = imgs.index(cur) if cur else 0
        n = next((n for n in imgs[start:] + imgs[:start] if todo(n)), None)
        return ("prior", n) if n else None

    def _loop(self):
        while True:
            with self.cv:
                if self.stopped:
                    return
                job = self._next()
                if job is None:
                    self.cv.wait(5)
                    continue
            kind, name = job
            try:
                image = np.asarray(Image.open(self.project.path(name)).convert("RGB"))
                if kind == "sam":
                    self.models.sam_features(str(self.project.path(name)), image)
                    continue
                probs = self.models.compute_prior(image, progress=lambda d, t, n=name: self.on_progress(n, d, t))
                self.project.save_prior(name, probs)
                self.on_done(name, probs)
            except Exception:
                import traceback
                traceback.print_exc()
                self.failed.add(name)
                if kind == "prior":
                    self.on_done(name, None)


class HoverWorker:
    """Computes the hover preview in its own thread. Only the latest cursor position
    matters: requests made while one is running replace each other."""

    def __init__(self, models, on_result):
        self.models, self.on_result = models, on_result
        self.cv = threading.Condition()
        self.request = None                  # (session, features, x, y, token)
        self.features = None                 # features currently set on the hover clicker
        threading.Thread(target=self._loop, name="hover_preview", daemon=True).start()

    def submit(self, session, features, x, y, token):
        with self.cv:
            self.request = (session, features, x, y, token)
            self.cv.notify()

    def _loop(self):
        while True:
            with self.cv:
                while self.request is None:
                    self.cv.wait()
                session, features, x, y, token = self.request
                self.request = None
            try:
                if features is not self.features:
                    self.models.hover.set_features(features)
                    self.features = features
                with torch_inference():
                    mask, cls = session.preview(x, y, self.models.hover)
                self.on_result(token, mask, cls)
            except Exception:
                import traceback
                traceback.print_exc()


def torch_inference():
    import torch
    return torch.inference_mode()


def mask_path(mask, max_points=4000):
    """QPainterPath of a mask's outer contours (computed on its bounding box)."""
    import cv2
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    path = QtGui.QPainterPath()
    if len(rows) == 0:
        return path
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    cnts, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE, offset=(int(x0), int(y0)))
    for c in cnts:
        c = c[:, 0, :]
        if len(c) > max_points:
            c = c[:: len(c) // max_points + 1]
        if len(c) >= 2:
            path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x) + .5, float(y) + .5) for x, y in c]))
            path.closeSubpath()
    return path


# ----------------------------------------------------------------- canvas
class Canvas(QtWidgets.QGraphicsView):
    """Shows the image. A left click (press + release without moving) emits
    `clicked`; a left drag draws a stroke and emits `drawn(points, is_loop, mods)`."""
    clicked = QtCore.Signal(int, int, object, object)   # x, y, button, modifiers
    drawn = QtCore.Signal(object, bool, object)         # [(x, y)], loop?, modifiers
    moved = QtCore.Signal(int, int)
    DRAG_PX = 6                                          # screen px before a press becomes a drag

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = self.scene().addPixmap(QtGui.QPixmap())
        self.setBackgroundBrush(QtGui.QColor(DARK))
        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._pan = None
        self._space = False              # Space held: left drag pans
        self._stroke = None              # image points of the stroke being drawn
        self._press = None               # (screen pos, modifiers) of the left press
        self._path = self.scene().addPath(QtGui.QPainterPath())
        self._path.setZValue(10)
        pen = QtGui.QPen(QtGui.QColor("#00e5ff"), 2, QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self._hover = self.scene().addPath(QtGui.QPainterPath(), pen)
        self._hover.setZValue(9)
        self._hover_label = self.scene().addSimpleText("")
        self._hover_label.setBrush(QtGui.QColor("#00e5ff"))
        self._hover_label.setPen(QtGui.QPen(QtGui.QColor("#000000"), 0.8))   # readable on any color
        f = QtGui.QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() + 1)
        self._hover_label.setFont(f)
        self._hover_label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self._hover_label.setZValue(11)

    def show_rgb(self, rgb, fit=False):
        h, w = rgb.shape[:2]
        img = QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.item.setPixmap(QtGui.QPixmap.fromImage(img))
        self.scene().setSceneRect(0, 0, w, h)
        if fit:
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    # -- view: scrolling and zooming (same on macOS, Windows, Linux) ------------
    MIN_ZOOM, MAX_ZOOM = 0.05, 40.0

    def zoom(self, factor, at_cursor=True):
        z = self.transform().m11()
        factor = min(max(factor, self.MIN_ZOOM / z), self.MAX_ZOOM / z)
        if not at_cursor:
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

    def zoom_reset(self):
        self.zoom(1.0 / self.transform().m11(), at_cursor=False)

    def fit(self):
        self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        """Wheel / two-finger swipe scrolls; Ctrl(Cmd)+wheel zooms at the cursor
        (a Windows/Linux touchpad pinch also arrives as Ctrl+wheel)."""
        ad, pd = e.angleDelta(), e.pixelDelta()
        if e.modifiers() & QtCore.Qt.ControlModifier:
            d = ad.y() or ad.x()
            if d:
                self.zoom(1.0015 ** d)        # one wheel notch (120) = x1.2
            return
        if not pd.isNull():                   # trackpad: exact pixels
            dx, dy = pd.x(), pd.y()
        else:                                 # mouse wheel: 120 per notch
            dx, dy = ad.x() * 0.5, ad.y() * 0.5
        if e.modifiers() & QtCore.Qt.ShiftModifier and dx == 0:
            dx, dy = dy, 0                    # Shift+wheel: left-right (macOS already swaps)
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() - int(dx))
        v.setValue(v.value() - int(dy))

    def viewportEvent(self, e):
        # macOS trackpad pinch / smart zoom (Windows/Linux pinch comes as Ctrl+wheel)
        if e.type() == QtCore.QEvent.NativeGesture:
            g = e.gestureType()
            if g == QtCore.Qt.ZoomNativeGesture:
                self.zoom(1.0 + e.value())
                return True
            if g == QtCore.Qt.SmartZoomNativeGesture:
                self.fit()
                return True
        return super().viewportEvent(e)

    left = QtCore.Signal()

    def show_hover(self, path, text, x, y):
        self._hover.setPath(path)
        self._hover_label.setText(text)
        self._hover_label.setPos(x + 2, y + 2)

    def clear_hover(self):
        self._hover.setPath(QtGui.QPainterPath())
        self._hover_label.setText("")

    def leaveEvent(self, e):
        self.left.emit()
        super().leaveEvent(e)

    def enterEvent(self, e):
        self.setFocus()                       # so Space (pan) reaches the canvas
        super().enterEvent(e)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space:
            if not e.isAutoRepeat():
                self._space = True
                self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            return
        e.ignore()                            # everything else goes to the window

    def keyReleaseEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.viewport().unsetCursor()
            return
        e.ignore()

    def _image_xy(self, e):
        p = self.mapToScene(e.position().toPoint())
        return int(p.x()), int(p.y())

    def _inside(self, x, y):
        r = self.item.pixmap().rect()
        return 0 <= x < r.width() and 0 <= y < r.height()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MiddleButton or (e.button() == QtCore.Qt.LeftButton and self._space):
            self._pan = e.position()
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            return
        x, y = self._image_xy(e)
        if not self._inside(x, y):
            return
        if e.button() == QtCore.Qt.LeftButton:
            self._press = (e.position(), e.modifiers())
            self._drag = False           # set once the mouse moves away (a loop ends where it began)
            p = self.mapToScene(e.position().toPoint())
            self._stroke = [(p.x(), p.y())]
        else:
            self.clicked.emit(x, y, e.button(), e.modifiers())

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._stroke is not None:
            p = self.mapToScene(e.position().toPoint())
            self._stroke.append((p.x(), p.y()))
            self._drag = self._drag or self._dragging(e)
            if self._drag:
                self._draw_stroke()
        self.moved.emit(*self._image_xy(e))

    def _dragging(self, e):
        d = e.position() - self._press[0]
        return abs(d.x()) + abs(d.y()) > self.DRAG_PX

    def _draw_stroke(self):
        from dmgseg.tool.lasso import is_loop
        mods = self._press[1] | QtWidgets.QApplication.keyboardModifiers()
        color = QtGui.QColor("#30e030" if mods & QtCore.Qt.ShiftModifier else
                             "#30e0ff" if mods & QtCore.Qt.ControlModifier else
                             "#ff3030" if mods & QtCore.Qt.AltModifier else "#ffff00")
        path = QtGui.QPainterPath(QtCore.QPointF(*self._stroke[0]))
        for x, y in self._stroke[1:]:
            path.lineTo(x, y)
        fill = QtGui.QColor(color)
        if is_loop(self._stroke):
            path.closeSubpath()
            fill.setAlpha(60)
        else:
            fill.setAlpha(0)
        pen = QtGui.QPen(color, 2, QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self._path.setPen(pen)
        self._path.setBrush(fill)
        self._path.setPath(path)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            if self._space:
                self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            return
        if self._stroke is None:
            return
        from dmgseg.tool.lasso import is_loop
        stroke, (pos, mods), drag = self._stroke, self._press, self._drag or self._dragging(e)
        self._stroke = self._press = None
        self._path.setPath(QtGui.QPainterPath())
        if not drag:
            x, y = int(stroke[0][0]), int(stroke[0][1])
            self.clicked.emit(x, y, QtCore.Qt.LeftButton, mods)
        else:
            mods = mods | QtWidgets.QApplication.keyboardModifiers()
            self.drawn.emit(stroke, bool(is_loop(stroke)), mods)


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
        self.project = None              # FolderProject when a folder is open
        self.current_name = None         # image name inside the project
        self.prefetcher = None
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_current)

        self.canvas = Canvas()
        self.canvas.clicked.connect(self.on_click)
        self.canvas.drawn.connect(self.on_drawn)
        self.canvas.moved.connect(self.on_hover)
        self.canvas.left.connect(self.end_hover)
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._request_hover)
        self._hover_xy, self._hover_token, self.hover_worker = None, 0, None
        self.canvas.moved.connect(lambda x, y: self.pos_label.setText(f" x={x} y={y} "))

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        split = QtWidgets.QSplitter()
        split.addWidget(self._images_panel())
        split.addWidget(self.canvas)
        split.addWidget(self._side_panel())
        split.setStretchFactor(1, 1)
        split.setSizes([215, 885, 260])
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
            self.run_bg(self.models.load, on_done=lambda m: self.status(
                "Ready (SAM: " + ("fine-tuned" if m.sam_finetuned else "zero-shot") + "). File > Open Image..."),
                        progress=lambda d, t: self.progress.setValue(int(100 * d / t)))

    # -- layout pieces ---------------------------------------------------------
    def _images_panel(self):
        box = QtWidgets.QGroupBox("Images")
        v = QtWidgets.QVBoxLayout(box)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.itemActivated.connect(lambda it: self.goto(it.data(QtCore.Qt.UserRole)))
        self.image_list.itemClicked.connect(lambda it: self.goto(it.data(QtCore.Qt.UserRole)))
        self.image_list.setToolTip("✓ n: annotated (n objects)\n•  prior ready\n   not started")
        v.addWidget(self.image_list)
        box.setVisible(False)
        self.images_box = box
        # a toy slot machine for breaks (off by default: View > Slot Machine)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(box, 1)
        self.slots_box = QtWidgets.QGroupBox("Lucky Labels")
        sl = QtWidgets.QVBoxLayout(self.slots_box)
        sl.setContentsMargins(4, 4, 4, 4)
        self.slots = SlotMachine()
        sl.addWidget(self.slots)
        lv.addStretch(0)
        lv.addWidget(self.slots_box)
        self.slots_box.setVisible(False)
        return left

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
        self.object_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.object_list.currentRowChanged.connect(self.on_object_selected)
        self.object_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(self.object_menu)
        bl2.addWidget(self.object_list)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Delete")
        b.setToolTip("Delete the selected objects (Backspace)")
        b.clicked.connect(self.delete_active)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Select tiny")
        b.setToolTip("Select objects smaller than 50 px (then Delete)")
        b.clicked.connect(self.select_tiny)
        row.addWidget(b)
        bl2.addLayout(row)
        v.addWidget(box2, 1)
        hint = QtWidgets.QLabel(f"Left click: new object\nRight click: next class\nShift+click: grow\n"
                                f"{ALT}+click: shrink / cut piece\nLoop: the object inside\n"
                                f"{CTRL}+loop: all objects inside\nLine: object along it\n"
                                f"Shift/{ALT}+loop: add/cut area\n1-5: set class, 0: Other\n"
                                f"Backspace: delete selected\n{CTRL}+wheel / pinch: zoom\n"
                                f"Space+drag: pan, H: hover preview")
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
        mb.setVisible(True)      # on macOS a menu bar inside the window stays hidden otherwise
        self.menu_bar = mb
        f = mb.addMenu("&File")
        self.act_open = f.addAction("&Open Image...", self.open_dialog, QtGui.QKeySequence.Open)
        f.addAction("Open &Folder...", self.open_folder_dialog, "Ctrl+Shift+O")
        f.addSeparator()
        f.addAction("&Save", self.save_current, QtGui.QKeySequence.Save)
        f.addAction("&Export Mask...", self.export_dialog, "Ctrl+E")
        f.addAction("Export &CVAT XML (folder)...", self.export_cvat_dialog, "Ctrl+Shift+E")
        f.addSeparator()
        # Page Up/Down are missing on Mac laptops: Cmd+Left/Right as well
        a = f.addAction("&Previous Image", lambda: self.step(-1))
        a.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_PageUp), QtGui.QKeySequence("Ctrl+Left")])
        a = f.addAction("&Next Image", lambda: self.step(1))
        a.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_PageDown), QtGui.QKeySequence("Ctrl+Right")])
        f.addSeparator()
        f.addAction("E&xit", self.close, QtGui.QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        self.act_undo = e.addAction("&Undo", self.undo, QtGui.QKeySequence.Undo)
        act_del = e.addAction("&Delete Object", self.delete_active)
        # On a Mac keyboard the big delete key is Backspace; Delete is Fn+Backspace.
        act_del.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_Backspace), QtGui.QKeySequence.Delete])
        vmenu = mb.addMenu("&View")
        vmenu.addAction("Fit to &Window", self.canvas.fit, "F")
        self.act_hover = vmenu.addAction("&Hover Preview", self.end_hover, "H")
        self.act_hover.setCheckable(True)
        self.act_hover.setChecked(True)
        self.act_hover.setToolTip("Outline the object a click would select, before clicking")
        self.act_slots = vmenu.addAction("&Slot Machine")
        self.act_slots.setCheckable(True)
        self.act_slots.setChecked(False)
        self.act_slots.toggled.connect(self.slots_box.setVisible)
        a = vmenu.addAction("Zoom &In", lambda: self.canvas.zoom(1.25, at_cursor=False))
        a.setShortcuts([QtGui.QKeySequence.ZoomIn, QtGui.QKeySequence("Ctrl+=")])
        vmenu.addAction("Zoom &Out", lambda: self.canvas.zoom(0.8, at_cursor=False), QtGui.QKeySequence.ZoomOut)
        vmenu.addAction("&Actual Size (100 %)", self.canvas.zoom_reset, "Ctrl+0")
        self.act_prior = vmenu.addAction("Show &Prior Map", self.toggle_prior, "P")
        self.act_prior.setCheckable(True)
        self.act_uncertain = vmenu.addAction("&Uncertain Areas", self.redraw, "U")
        self.act_uncertain.setCheckable(True)
        self.act_uncertain.setToolTip("Dim everything except where the prior is least sure "
                                      "(about half of its mistakes are there)")
        t = mb.addMenu("&Tools")
        t.addAction("&Auto Pre-label (draft now)", lambda: self.run_prelabel(force=True), "Ctrl+L")
        t.addAction("&Remove Draft", self.remove_draft, "Ctrl+Shift+L")
        h = mb.addMenu("&Help")
        h.addAction("&Shortcuts", self.show_shortcuts, QtGui.QKeySequence.HelpContents)
        h.addAction("&About", lambda: QtWidgets.QMessageBox.about(
            self, "About", f"<b>{TITLE}</b><br>SAM 2.1 + DINOv2 prior + class head.<br>Lab work, KNU 2026."))

    def _toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        st = self.style()
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), "Folder", self.open_folder_dialog)
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Open", self.open_dialog)
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save", self.save_current)
        export = QtWidgets.QToolButton()
        export.setText("Export")
        export.setIcon(st.standardIcon(QtWidgets.QStyle.SP_DriveFDIcon))
        export.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        em = QtWidgets.QMenu(export)
        em.addAction("CVAT XML for the whole folder... (Ctrl+Shift+E)", self.export_cvat_dialog)
        em.addAction("Mask of this image (PNG + overlay + JSON)... (Ctrl+E)", self.export_dialog)
        export.setMenu(em)
        export.setToolTip("Export your labels in a readable format (CVAT XML like the paper's dataset, or PNG masks)")
        tb.addWidget(export)
        tb.addSeparator()
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward), "Prev", lambda: self.step(-1))
        tb.addAction(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward), "Next", lambda: self.step(1))
        tb.addSeparator()
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
        self.auto_check = QtWidgets.QCheckBox("Auto pre-label")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip(
            "On: draft objects from the DINO map automatically (now and for every image).\n"
            "Off: remove the untouched draft (Ctrl+Z brings it back) and label by hand;\n"
            "clicked objects still get their class from DINO and the class head.\n"
            "Draft objects you edit or re-classify become yours and are never removed.")
        self.auto_check.toggled.connect(self.on_auto_toggled)
        tb.addWidget(self.auto_check)
        tb.addSeparator()
        cb = QtWidgets.QCheckBox("Prior map")
        cb.setToolTip("View only: show DINO's raw class map instead of your objects (P)")
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

    def open_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder of Images")
        if path:
            self.open_folder(path)

    def open_folder(self, path):
        if not self.ready():
            self.status("Models are still loading, please wait...")
            QtCore.QTimer.singleShot(500, lambda: self.open_folder(path))
            return
        self.save_current()
        self.project = FolderProject(path)
        if not self.project.images:
            self.status(f"No images in {path}")
            self.project = None
            return
        if self.prefetcher:
            self.prefetcher.stop()
        self.prefetcher = PriorPrefetcher(
            self.models, self.project,
            on_done=lambda n, p: self._ui.emit(self._prefetch_done, (n, p)),
            on_progress=lambda n, d, t: self._ui.emit(self._prefetch_progress, (n, d, t)))
        self.images_box.setVisible(True)
        self._fill_image_list()
        first = next((n for n in self.project.images if self.project.n_objects(n) == 0), self.project.images[0])
        self.goto(first)

    def goto(self, name):
        if self.project is not None and name in self.project.images and name != self.current_name:
            self.open_image(str(self.project.path(name)))

    def step(self, delta):
        if self.project is None or self.current_name is None:
            return
        imgs = self.project.images
        i = imgs.index(self.current_name) + delta
        if 0 <= i < len(imgs):
            self.goto(imgs[i])

    def open_image(self, path):
        if not self.ready():
            self.status("Models are still loading, please wait...")
            return
        self.save_current()
        path = Path(path)
        in_project = self.project is not None and path.parent == self.project.folder
        name = path.name if in_project else None
        self.current_name = name
        self.image_path = path
        self.session = None
        self.end_hover()
        image = np.asarray(Image.open(path).convert("RGB"))
        self.caption.setText(f"  {TITLE} - {path.name}")
        self.progress.setValue(0)
        state = self.project.load_state(name) if name else None
        key = str(path)
        cached = self.models.cached_features(key)
        # show the saved labels at once; SAM is only needed for new clicks
        if cached is None and state and state["objects"] and (state["height"], state["width"]) == image.shape[:2]:
            _, lm = visible_objects_from_state(state)
            self.canvas.show_rgb(blend_labels(image, lm, self.alpha), fit=True)
        else:
            self.canvas.show_rgb(image, fit=True)
        if in_project:
            self._mark_current()
            self.prefetcher.set_current(name)

        def got_features(features):
            if self.image_path != path:          # the user moved on meanwhile
                return
            session = Session(image, self.models.clicker, self.models.head, features=features)
            if state:
                session.load_state(state)
            self.session = session
            restored = f"restored {len(session.objects)} objects; " if state else ""
            if name and self.project.has_prior(name):
                self.got_prior(self.project.load_prior(name), cached=True)     # refreshes
            elif name:
                self.refresh(save=False)
                self.prior_label.setText(" Prior: computing ")
                self.status(restored + "click objects (classes appear when the prior is done)")
            else:
                self.refresh(save=False)
                self.prior_label.setText(" Prior: computing ")
                self.status("Ready: click objects (the class appears when the prior is done)")
                t0 = time.time()
                self.run_bg(self.models.compute_prior, image,
                            on_done=lambda p: self.got_prior(p, seconds=time.time() - t0) if self.image_path == path else None,
                            progress=lambda d, t: self.progress.setValue(int(100 * d / t)))

        if cached is not None:
            got_features(cached)
        else:
            self.status(f"SAM is reading {path.name} (a few seconds; the saved labels are shown already)")
            self.run_bg(lambda progress=None: self.models.sam_features(key, image), on_done=got_features)

    def got_prior(self, prior, seconds=None, cached=False):
        if self.session is None or self.session.prior is not None:
            return
        self.session.set_prior(prior)
        self.prior_label.setText(" Prior: cached " if cached else f" Prior: done ({seconds:.0f} s) " if seconds else " Prior: done ")
        self.progress.setValue(100)
        self.status("Prior ready")
        self.refresh()
        if self.auto_check.isChecked() and not self.session.prelabeled:
            self.run_prelabel()

    def on_auto_toggled(self, on):
        """The Auto pre-label box acts at once: off removes the untouched draft of
        this image (undoable) and stops drafting; on drafts this image now."""
        s = self.session
        if not on:
            self._drafting = None                # a draft still being computed is dropped
            self._busy(False)
            if s is not None and s.draft_count():
                n = s.remove_draft()
                self.status(f"Auto pre-label off: removed {n} untouched draft objects (Ctrl+Z restores). "
                            "Clicks still get their class from DINO.")
                self.refresh()
            else:
                self.status("Auto pre-label off: label by hand (classes still come from DINO)")
        elif s is not None and s.prior is not None and not s.draft_count():
            self.run_prelabel(force=True)

    def remove_draft(self):
        s = self.session
        if s is not None:
            n = s.remove_draft()
            self.status(f"Removed {n} untouched draft objects (Ctrl+Z restores)" if n else "No draft objects")
            self.refresh()

    def _busy(self, on):
        """Moving progress bar while the draft is computed."""
        if on:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    # -- folder: background priors, saving, list -------------------------------------
    def _prefetch_done(self, name, probs):
        self._update_image_item(name)
        if probs is not None and name == self.current_name and self.session is not None:
            self.got_prior(probs)
        left = self.prefetcher.pending() if self.prefetcher else 0
        if name != self.current_name:
            self.prior_label.setText(f" Priors to pre-compute: {left} " if left else " All priors ready ")

    def _prefetch_progress(self, name, done, total):
        if name == self.current_name:
            self.progress.setValue(int(100 * done / total))

    def save_current(self):
        if self.project is None or self.current_name is None or self.session is None:
            return
        self.project.save_state(self.current_name, self.session.to_state())
        self._update_image_item(self.current_name)

    def _item_text(self, name):
        n = self.project.n_objects(name)
        mark = f"✓{n:>3}" if n else (" •  " if self.project.has_prior(name) else "    ")
        return f"{mark}  {name}"

    def _fill_image_list(self):
        self.image_list.clear()
        for name in self.project.images:
            it = QtWidgets.QListWidgetItem(self._item_text(name))
            it.setData(QtCore.Qt.UserRole, name)
            self.image_list.addItem(it)

    def _update_image_item(self, name):
        if self.project is None or name not in self.project.images:
            return
        self.image_list.item(self.project.images.index(name)).setText(self._item_text(name))

    def _mark_current(self):
        self.image_list.blockSignals(True)
        self.image_list.setCurrentRow(self.project.images.index(self.current_name))
        self.image_list.blockSignals(False)

    def export_cvat_dialog(self):
        if self.project is None:
            self.status("Open a folder first (File > Open Folder)")
            return
        self.save_current()
        kind, ok = QtWidgets.QInputDialog.getItem(
            self, "Export CVAT XML", "Shapes:", ["Masks (exact, CVAT 2.x)", "Polygons (outlines)"], 0, False)
        if not ok:
            return
        default = str(self.project.folder / "annotations_cvat.xml")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CVAT XML", default, "XML (*.xml)")
        if path:
            n = export_cvat(self.project, path, shape="mask" if kind.startswith("Masks") else "polygon")
            self.status(f"CVAT XML with {n} annotated images saved to {Path(path).name}")

    def closeEvent(self, e):
        self.save_current()
        if self.prefetcher:
            self.prefetcher.stop()
        super().closeEvent(e)

    def run_prelabel(self, force=False):
        """The automatic draft, made in a background thread (it takes seconds on big
        images); clicking keeps working meanwhile and the draft is added when ready.
        force: also when this image had a draft before (Tools menu, re-ticking the box)."""
        s = self.session
        if s is None or s.prior is None:
            self.status("Pre-label needs the prior: wait for 'Prior: done'")
            return
        if getattr(self, "_drafting", None) is s:
            return
        if s.draft_count() and force:
            s.remove_draft()                     # replace, never duplicate, the draft
        self._drafting = s
        features, prior, t0 = self.models.clicker.features(), s.prior, time.time()
        self._busy(True)
        big = s.h * s.w > 2_500_000
        self.status("Auto pre-label: drafting objects from the DINO map"
                    + (" (large image: ~10 s)" if big else "") + "... you can click meanwhile")

        def work(progress=None):
            from dmgseg.tool.prelabel import prelabel
            with self.models.draft_lock:
                d = self.models.drafter
                d.set_features(features)
                return prelabel(prior, d, keep_small_px=25)    # no 1-20 px specks

        def done(props):
            if getattr(self, "_drafting", None) is not s:
                return                           # cancelled (box unticked) or replaced
            self._drafting = None
            self._busy(False)
            if self.session is not s or not (force or self.auto_check.isChecked()):
                return                           # moved to another image / automatic draft, box unticked
            n = s.apply_prelabel(props)
            self.status(f"Auto pre-label: {n} objects in {time.time() - t0:.0f} s. "
                        "Correct with clicks; Ctrl+Z removes the draft")
            self.refresh()

        self.run_bg(work, on_done=done)

    # -- interaction -------------------------------------------------------------
    def on_click(self, x, y, button, mods):
        self.end_hover()
        s = self.session
        if s is None:
            return
        t0 = time.time()
        # right click; Ctrl(Cmd)+click and, on a Mac, Control+click (Qt: Meta) do the same
        right = button == QtCore.Qt.RightButton or (
            button == QtCore.Qt.LeftButton and mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
        shrink = mods & QtCore.Qt.AltModifier or (right and mods & QtCore.Qt.ShiftModifier)
        if shrink:
            # the active object if the click is on it, else the object under the cursor
            i = s.active if s.active is not None and s.objects[s.active].mask[y, x] else s.object_at(x, y)
            if i is None:
                self.status(f"{ALT}+click: no object here")
                return
            local = not s.objects[i].sam_valid
            s.refine(x, y, False, i)
            self.status(f"Cut the piece under the cursor from object #{i + 1}" if local
                        else f"Shrank object #{i + 1}")
        elif right:
            i = s.object_at(x, y)
            if i is None:
                self.status("Right click: no object here")
                return
            s.next_class(i)
            self.status(f"Object #{i + 1}: {CLASS_NAMES[s.objects[i].label]}")
        elif mods & QtCore.Qt.ShiftModifier:
            if s.active is None:
                self.status("Shift+click grows the selected object: select one first")
                return
            local = not s.objects[s.active].sam_valid
            s.refine(x, y, True)
            self.status("Added the piece under the cursor" if local else "Grew the active object")
        else:
            i = s.new_object(x, y)
            o = s.objects[i]
            self.status(f"Object #{i + 1}: " + ("class pending (prior running)" if o.pending
                                               else CLASS_NAMES[o.label]))
        self.time_label.setText(f" {1000 * (time.time() - t0):.0f} ms ")
        self.refresh()

    # -- hover preview -------------------------------------------------------------
    def on_hover(self, x, y):
        self.end_hover(keep_timer=True)
        if self.session is None or not self.act_hover.isChecked() or self.canvas._stroke is not None \
                or QtWidgets.QApplication.mouseButtons() != QtCore.Qt.NoButton:
            return
        r = self.canvas.item.pixmap().rect()
        if not (0 <= x < r.width() and 0 <= y < r.height()):
            return
        self._hover_xy = (x, y)
        self._hover_timer.start(60)          # once the cursor rests a moment

    def _request_hover(self):
        if self.session is None or self._hover_xy is None:
            return
        if self.hover_worker is None:
            self.hover_worker = HoverWorker(self.models, lambda *r: self._ui.emit(self._show_hover, r))
        self._hover_token += 1
        x, y = self._hover_xy
        self.hover_worker.submit(self.session, self.models.clicker.features(), x, y,
                                 (self._hover_token, self.session))

    def _show_hover(self, token, mask, cls):
        n, session = token
        if n != self._hover_token or session is not self.session or self._hover_xy is None:
            return                            # the cursor moved on meanwhile
        x, y = self._hover_xy
        self.canvas.show_hover(mask_path(mask), CLASS_NAMES[cls] if cls is not None else "", x, y)

    def end_hover(self, keep_timer=False):
        self._hover_token += 1
        if not keep_timer:
            self._hover_timer.stop()
            self._hover_xy = None
        self.canvas.clear_hover()

    def on_drawn(self, stroke, loop, mods):
        """A drawn stroke: loop = grab what is inside, open stroke = scribble.
        Shift / Option: loop = add / cut exactly that area, scribble = SAM grow / shrink."""
        s = self.session
        if s is None:
            return
        t0 = time.time()
        add, cut = bool(mods & QtCore.Qt.ShiftModifier), bool(mods & QtCore.Qt.AltModifier)
        group = bool(mods & QtCore.Qt.ControlModifier)          # Cmd on a Mac
        if loop and (add or cut):
            i = s.edit_area(stroke, add=add, index=s.active)
            what = "Added the area to" if add else "Cut the area from"
            self.status(f"{what} object #{i + 1}" if i is not None else
                        ("New object from the drawn area" if add else "Cut the area from all objects"))
        elif loop and not group:
            i = s.lasso(stroke)[0]
            o = s.objects[i]
            kept_loop = not o.sam_valid and o.sam_score == 0
            self.status(f"Loop -> object #{i + 1}: " + ("class pending" if o.pending else CLASS_NAMES[o.label])
                        + (" (no SAM shape fits the loop: kept the loop's own shape)" if kept_loop else ""))
        elif loop:
            self.status("Lasso: looking for objects inside the loop...")
            QtWidgets.QApplication.processEvents()
            new = s.lasso(stroke, group=True)
            if new:
                names = [CLASS_NAMES[s.objects[i].label] if s.objects[i].label is not None else "pending"
                         for i in new]
                summary = ", ".join(f"{names.count(n)} {n}" for n in dict.fromkeys(names))
                self.status(f"Lasso: {len(new)} new object(s): {summary}")
            else:
                self.status("Lasso: nothing new inside the loop (already labeled)")
        elif (add or cut) and s.active is not None:
            s.scribble(stroke, positive=add, index=s.active)
            self.status("Grew the active object along the stroke" if add else
                        "Shrank the active object along the stroke")
        else:
            i = s.scribble(stroke)
            o = s.objects[i]
            self.status(f"Scribble -> object #{i + 1}: " + ("class pending" if o.pending else CLASS_NAMES[o.label]))
        self.time_label.setText(f" {1000 * (time.time() - t0):.0f} ms ")
        self.refresh()

    def keyPressEvent(self, e):
        s = self.session
        key = e.key()
        rows = self.selected_objects() if s is not None else []
        if rows and QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_5:
            s.set_classes(rows, CLICKABLE[key - QtCore.Qt.Key_1])
            self.refresh()
        elif rows and key == QtCore.Qt.Key_0:
            s.set_classes(rows, 0)
            self.status(f"{len(rows)} object(s) marked as Other (cut out of what is behind)")
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

    def show_shortcuts(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Shortcuts")
        box.setText(f"<pre>{shortcuts_text()}</pre>")
        box.exec()

    def selected_objects(self):
        """Rows selected in the Objects list, or the active object."""
        rows = sorted({self.object_list.row(it) for it in self.object_list.selectedItems()})
        if not rows and self.session is not None and self.session.active is not None:
            rows = [self.session.active]
        return rows

    def delete_active(self):
        s = self.session
        rows = self.selected_objects() if s else []
        if rows:
            s.delete_many(rows)
            self.status(f"Deleted {len(rows)} object(s) (Ctrl+Z brings them back)")
            self.refresh()

    def select_tiny(self, limit=50):
        s = self.session
        if s is None:
            return
        self.object_list.clearSelection()
        n = 0
        for i, o in enumerate(s.objects):
            if o.area < limit:
                self.object_list.item(i).setSelected(True)
                n += 1
        self.status(f"Selected {n} object(s) smaller than {limit} px: press Delete to remove them")

    def object_menu(self, pos):
        s = self.session
        if s is None or self.object_list.itemAt(pos) is None:
            return
        rows = self.selected_objects()
        m = QtWidgets.QMenu(self)
        m.addAction(f"Delete ({len(rows)})", self.delete_active)
        sub = m.addMenu("Set class")
        for key, c in [(n, c) for n, c in enumerate(CLICKABLE, start=1)] + [(0, 0)]:
            sub.addAction(swatch(c), f"{key}  {CLASS_NAMES[c]}",
                          lambda c=c: (s.set_classes(self.selected_objects(), c), self.refresh()))
        m.exec(self.object_list.mapToGlobal(pos))

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
            self.status(f"Saved {Path(png).name} (class mask), {Path(png).stem}_overlay.jpg (preview) "
                        f"and {Path(js).name}")

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
        if self.act_uncertain.isChecked():
            u = s.uncertain_areas()
            if u is not None:                    # spotlight: dim what the prior is sure about
                rgb = rgb.copy()
                rgb[~u] = (rgb[~u] * 0.35).astype(np.uint8)
        self.canvas.show_rgb(rgb)

    def refresh(self, save=True):
        if save and self.project is not None and self.session is not None:
            self._save_timer.start(1500)      # auto-save shortly after each change
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
                flag = (" *" if o.manual is not None else "") + (" (auto)" if o.auto else "")
                it = QtWidgets.QListWidgetItem(swatch(o.label if o.label is not None else 0),
                                               f"#{i + 1}  {name}{flag}  ({o.area} px)")
                self.object_list.addItem(it)
            if s.active is not None:
                self.object_list.setCurrentRow(s.active)
        self.object_list.blockSignals(False)
        self.redraw()


def main(argv=None):
    # keep the menu bar inside the window (Windows 98 look) instead of macOS's top bar
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeMenuBar, True)
    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName(TITLE)
    apply_classic_style(app)
    win = MainWindow()
    settings = QtCore.QSettings("dmgseg", TITLE)
    win.auto_check.setChecked(settings.value("auto_prelabel", True, type=bool))
    win.auto_check.toggled.connect(lambda on: settings.setValue("auto_prelabel", on))
    win.act_slots.setChecked(settings.value("slot_machine", False, type=bool))
    win.act_slots.toggled.connect(lambda on: settings.setValue("slot_machine", on))
    win.show()
    if len(sys.argv) > 1:
        QtCore.QTimer.singleShot(0, lambda: _open_when_ready(win, sys.argv[1]))
    return app.exec()


def _open_when_ready(win, path):
    if win.ready():
        (win.open_folder if Path(path).is_dir() else win.open_image)(path)
    else:
        QtCore.QTimer.singleShot(500, lambda: _open_when_ready(win, path))


if __name__ == "__main__":
    sys.exit(main())
