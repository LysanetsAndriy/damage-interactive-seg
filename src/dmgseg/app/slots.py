"""Lucky Labels: a toy slot machine for short breaks while labeling.

Just for fun: no money, no stakes, nothing is saved. Three reels show the six
annotation classes as little icons. Pull the handle (click it, or drag the red knob
down): the reels spin, stop one after another with a small overshoot, and three of
a kind throws confetti over the whole window.

Everything is drawn with QPainter (no image files), animated by one 60 fps timer
that runs only while something moves.
"""
import math
import random
import time

from PySide6 import QtCore, QtGui, QtWidgets

from dmgseg.app.engine import COLORS
from dmgseg.data.cvat import CLASS_NAMES

SYMBOLS = [1, 2, 3, 4, 5, 0]
SHORT = {0: "Tree", 1: "Building", 2: "Roof", 3: "Damage", 4: "Window", 5: "Dmg roof"}
GOLD = QtGui.QColor("#f5c518")
JACKPOT_P, PAIR_P = 0.12, 0.33            # chance of three / two of a kind
NO_LUCK = ["No luck. Label on!", "Almost... not.", "Try again", "The house wins", "Next one!"]


# ----------------------------------------------------------------- symbols
def _c(class_id, alpha=255):
    c = QtGui.QColor(*COLORS[class_id])
    c.setAlpha(alpha)
    return c


def draw_symbol(p, class_id, r):
    """A small icon for a class inside rect r (QRectF)."""
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    col = _c(class_id)
    p.setPen(QtGui.QPen(col.darker(170), max(1.0, w / 26)))
    P = QtCore.QPointF
    if class_id == 1:                                   # building with windows
        body = QtCore.QRectF(x + w * .24, y + h * .1, w * .52, h * .82)
        p.setBrush(col)
        p.drawRect(body)
        p.setBrush(QtGui.QColor("#e8f2ff"))
        for i in range(3):
            for j in range(2):
                p.drawRect(QtCore.QRectF(body.x() + body.width() * (.14 + .46 * j),
                                         body.y() + body.height() * (.08 + .24 * i),
                                         body.width() * .3, body.height() * .15))
        p.setBrush(col.darker(140))
        p.drawRect(QtCore.QRectF(body.center().x() - w * .07, body.bottom() - h * .2, w * .14, h * .2))
    elif class_id == 2:                                 # roof on a wall
        p.setBrush(QtGui.QColor("#e9dcc0"))
        p.drawRect(QtCore.QRectF(x + w * .24, y + h * .5, w * .52, h * .4))
        p.setBrush(col)
        p.drawPolygon(QtGui.QPolygonF([P(x + w * .1, y + h * .55), P(x + w * .5, y + h * .12),
                                       P(x + w * .9, y + h * .55)]))
        p.setBrush(QtGui.QColor("#8b5a2b"))
        p.drawRect(QtCore.QRectF(x + w * .66, y + h * .2, w * .1, h * .2))
    elif class_id == 3:                                 # damage: a burst with a crack
        pts = []
        for k in range(16):
            a = math.pi * 2 * k / 16
            rr = (.42 if k % 2 == 0 else .22) * min(w, h)
            pts.append(P(x + w / 2 + rr * math.cos(a), y + h / 2 + rr * math.sin(a)))
        p.setBrush(col)
        p.drawPolygon(QtGui.QPolygonF(pts))
        p.setPen(QtGui.QPen(QtGui.QColor("#2b0a33"), max(1.5, w / 18)))
        p.drawPolyline(QtGui.QPolygonF([P(x + w * .38, y + h * .3), P(x + w * .5, y + h * .45),
                                        P(x + w * .43, y + h * .55), P(x + w * .6, y + h * .72)]))
    elif class_id == 4:                                 # broken window
        f = QtCore.QRectF(x + w * .18, y + h * .14, w * .64, h * .72)
        p.setBrush(QtGui.QColor("#f4f4f4"))
        p.drawRect(f)
        g = f.adjusted(w * .06, h * .06, -w * .06, -h * .06)
        p.setBrush(col)
        p.drawRect(g)
        p.setPen(QtGui.QPen(QtGui.QColor("#f4f4f4"), max(1.0, w / 30)))
        p.drawLine(P(g.center().x(), g.top()), P(g.center().x(), g.bottom()))
        p.drawLine(P(g.left(), g.center().y()), P(g.right(), g.center().y()))
        c0 = P(g.x() + g.width() * .35, g.y() + g.height() * .35)
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), max(1.0, w / 34)))
        for a in (0.3, 1.4, 2.5, 3.6, 4.8, 5.6):
            p.drawLine(c0, P(c0.x() + math.cos(a) * w * .22, c0.y() + math.sin(a) * h * .22))
    elif class_id == 5:                                 # damaged roof: a hole in it
        p.setBrush(QtGui.QColor("#e9dcc0"))
        p.drawRect(QtCore.QRectF(x + w * .24, y + h * .5, w * .52, h * .4))
        p.setBrush(col)
        p.drawPolygon(QtGui.QPolygonF([P(x + w * .1, y + h * .55), P(x + w * .5, y + h * .12),
                                       P(x + w * .9, y + h * .55)]))
        p.setBrush(QtGui.QColor("#1a1a1a"))
        p.drawPolygon(QtGui.QPolygonF([P(x + w * .42, y + h * .3), P(x + w * .5, y + h * .36),
                                       P(x + w * .58, y + h * .28), P(x + w * .62, y + h * .44),
                                       P(x + w * .46, y + h * .5), P(x + w * .38, y + h * .42)]))
    else:                                               # Other: a tree
        p.setBrush(QtGui.QColor("#7a5230"))
        p.drawRect(QtCore.QRectF(x + w * .45, y + h * .55, w * .1, h * .33))
        p.setBrush(QtGui.QColor("#6f8f6a"))
        for cx, cy, rr in ((.5, .32, .2), (.35, .46, .16), (.65, .46, .16)):
            p.drawEllipse(P(x + w * cx, y + h * cy), w * rr, h * rr)
    p.restore()


# ----------------------------------------------------------------- reels
def ease_out_back(t, s=1.4):
    t -= 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


class Reel:
    def __init__(self, rng):
        self.strip = SYMBOLS * 3
        rng.shuffle(self.strip)
        self.n = len(self.strip)
        self.pos = float(rng.randrange(self.n))
        self.state = "idle"                     # idle / spin / stopping
        self.speed = 0.0

    def symbol(self, k):
        return self.strip[k % self.n]

    def result(self):
        return self.symbol(round(self.pos))

    def start(self, speed):
        self.state, self.speed = "spin", speed

    def stop_on(self, symbol, now, min_travel=1):
        k = math.ceil(self.pos) + min_travel
        while self.symbol(k) != symbol:
            k += 1
        self.p0, self.p1, self.t0 = self.pos, float(k), now
        # the ease starts at about the spinning speed (f'(0) = s + 3 for
        # ease_out_back), but a reel never takes more than 0.75 s to stop
        self.dur = min(0.75, max(0.35, (1.4 + 3) * (self.p1 - self.p0) / self.speed))
        self.state = "stopping"

    def update(self, now, dt):
        if self.state == "spin":
            self.pos += self.speed * dt
        elif self.state == "stopping":
            t = (now - self.t0) / self.dur
            if t >= 1:
                self.pos, self.state = self.p1, "idle"
            else:
                self.pos = self.p0 + (self.p1 - self.p0) * ease_out_back(t)


# ----------------------------------------------------------------- confetti
class Confetti(QtWidgets.QWidget):
    """Transparent layer over the whole window; paper bits fly out and fall."""

    def __init__(self, top):
        super().__init__(top)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.parts = []
        self._last = 0.0
        self.timer = QtCore.QTimer(self, interval=16, timeout=self._step)
        self.hide()

    def burst(self, origin, n=160, rng=random):
        colors = [_c(c) for c in SYMBOLS] + [GOLD, GOLD, QtGui.QColor("#ffffff")]
        for _ in range(n):
            a = rng.uniform(-math.pi * .95, -math.pi * .05)        # upward fan
            v = rng.uniform(250, 820)
            self.parts.append({
                "x": origin.x(), "y": origin.y(), "vx": math.cos(a) * v, "vy": math.sin(a) * v,
                "rot": rng.uniform(0, 360), "vrot": rng.uniform(-720, 720), "flip": rng.uniform(0, 6.3),
                "vflip": rng.uniform(4, 14), "w": rng.uniform(5, 10), "h": rng.uniform(3, 6),
                "color": rng.choice(colors), "life": rng.uniform(2.0, 3.2), "round": rng.random() < .25})
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        if not self.timer.isActive():
            self._last = time.perf_counter()
            self.timer.start()

    def _step(self):
        now = time.perf_counter()
        dt = min(now - self._last, 0.05)
        self._last = now
        for q in self.parts:
            q["vy"] += 900 * dt
            q["vx"] *= 1 - 1.2 * dt                                  # air drag
            q["vy"] *= 1 - 0.9 * dt
            q["vx"] += math.sin(q["flip"]) * 40 * dt                  # flutter
            q["x"] += q["vx"] * dt
            q["y"] += q["vy"] * dt
            q["rot"] += q["vrot"] * dt
            q["flip"] += q["vflip"] * dt
            q["life"] -= dt
        h = self.height()
        self.parts = [q for q in self.parts if q["life"] > 0 and q["y"] < h + 20]
        if not self.parts:
            self.timer.stop()
            self.hide()
        self.setGeometry(self.parentWidget().rect())
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        for q in self.parts:
            c = QtGui.QColor(q["color"])
            c.setAlphaF(max(0.0, min(1.0, q["life"] / 0.6)))
            p.save()
            p.translate(q["x"], q["y"])
            p.rotate(q["rot"])
            sx = abs(math.cos(q["flip"])) * 0.85 + 0.15                # paper turning over
            p.scale(sx, 1.0)
            p.setBrush(c)
            if q["round"]:
                p.drawEllipse(QtCore.QPointF(0, 0), q["h"] / 1.4, q["h"] / 1.4)
            else:
                p.drawRect(QtCore.QRectF(-q["w"] / 2, -q["h"] / 2, q["w"], q["h"]))
            p.restore()


# ----------------------------------------------------------------- the machine
class SlotMachine(QtWidgets.QWidget):
    finished = QtCore.Signal(tuple)          # the three symbols (class ids)

    PULL_DOWN, SPRING_BACK = 0.22, 0.5       # seconds
    STOPS = (0.55, 0.9, 1.25)                # when each reel starts to stop (s after the pull)

    def __init__(self, parent=None, rng=None):
        super().__init__(parent)
        self.rng = rng or random.Random()
        self.reels = [Reel(self.rng) for _ in range(3)]
        self.pull = 0.0
        self.handle = "idle"                 # idle / down / up / drag
        self.spinning = False
        self.spins = self.jackpots = 0
        self.msg = "PULL THE HANDLE"
        self.sub = None                      # second LCD line (None: the counters)
        self.win_until = 0.0
        self.win_symbol = None
        self.confetti = None
        self._press = None
        self._last = time.perf_counter()
        self.timer = QtCore.QTimer(self, interval=16, timeout=self._tick)
        self.setMinimumWidth(170)
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setMouseTracking(True)
        self.setToolTip("A toy slot machine: pull the handle (no money, just for fun)")

    HANDLE_W, LCD_H = 26, 40

    @classmethod
    def _cell(cls, W):
        return min((W - cls.HANDLE_W - 6 - 26) / 3, 64)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, W):
        # body margin + marquee + reels window + LCD
        return int(3 + 6 + 30 + 12 + 1.5 * self._cell(W) + 5 + 10 + self.LCD_H + 8 + 3)

    def sizeHint(self):
        return QtCore.QSize(200, self.heightForWidth(200))

    def resizeEvent(self, e):
        if self.height() != self.heightForWidth(self.width()):
            self.setFixedHeight(self.heightForWidth(self.width()))
        super().resizeEvent(e)

    # -- geometry ----------------------------------------------------------------
    def _geo(self):
        W, H = self.width(), self.height()
        hw = self.HANDLE_W
        body = QtCore.QRectF(3, 3, W - hw - 6, H - 6)
        marquee = QtCore.QRectF(body.x() + 6, body.y() + 6, body.width() - 12, 30)
        cell = self._cell(W)
        rw, rh = cell, cell * 1.5
        total = 3 * rw + 2 * 5
        x0 = body.center().x() - total / 2
        ry = marquee.bottom() + 12
        reels = [QtCore.QRectF(x0 + i * (rw + 5), ry, rw, rh) for i in range(3)]
        window = QtCore.QRectF(x0 - 5, ry - 5, total + 10, rh + 10)
        lcd = QtCore.QRectF(body.x() + 8, window.bottom() + 10, body.width() - 16, self.LCD_H)
        pivot = QtCore.QPointF(body.right() + hw / 2 + 1, window.center().y() + 4)
        top_y = body.y() + 14
        bottom_y = min(pivot.y() + (pivot.y() - top_y) * 0.85, H - 12)
        return dict(body=body, marquee=marquee, reels=reels, window=window, lcd=lcd, cell=cell,
                    pivot=pivot, top_y=top_y, bottom_y=bottom_y)

    def _knob(self, g):
        y = g["top_y"] + (g["bottom_y"] - g["top_y"]) * self.pull
        return QtCore.QPointF(g["pivot"].x(), y)

    # -- actions -----------------------------------------------------------------
    def pull_handle(self, outcome=None):
        """Animate the handle and spin. outcome: force a result (tests)."""
        if self.spinning or self.handle in ("down", "up"):
            return False
        self._outcome = outcome
        self.handle, self._h0 = "down", time.perf_counter()
        self._run()
        return True

    def _start_spin(self, now):
        self.spinning = True
        self.spins += 1
        self.win_symbol = None
        self.msg, self.sub = "SPINNING...", None
        if self._outcome is None:
            r = self.rng.random()
            if r < JACKPOT_P:
                s = self.rng.choice(SYMBOLS)
                self._outcome = (s, s, s)
            elif r < JACKPOT_P + PAIR_P:
                a, b = self.rng.sample(SYMBOLS, 2)
                o = [a, a, b]
                self.rng.shuffle(o)
                self._outcome = tuple(o)
            else:
                self._outcome = tuple(self.rng.sample(SYMBOLS, 3))
        for i, reel in enumerate(self.reels):
            reel.start(17 + 3 * i + self.rng.uniform(-1.5, 1.5))
        self._spin_t0 = now

    def _finish(self, now):
        self.spinning = False
        res = tuple(r.result() for r in self.reels)
        if res[0] == res[1] == res[2]:
            self.jackpots += 1
            self.win_symbol = res[0]
            self.win_until = now + 2.8
            self.msg, self.sub = "JACKPOT!", f"3 x {SHORT[res[0]]}"
            self._throw_confetti()
        elif len(set(res)) == 2:
            pair = max(set(res), key=res.count)
            self.msg, self.sub = "So close!", f"2 x {SHORT[pair]}"
            self.win_until = now + 0.9
        else:
            self.msg = self.rng.choice(NO_LUCK)
        self.finished.emit(res)

    def _throw_confetti(self):
        top = self.window()
        if self.confetti is None or self.confetti.parentWidget() is not top:
            self.confetti = Confetti(top)
        g = self._geo()
        origin = self.mapTo(top, g["window"].center().toPoint())
        self.confetti.burst(QtCore.QPointF(origin), 170, self.rng)
        QtCore.QTimer.singleShot(280, lambda: self.confetti.burst(QtCore.QPointF(origin), 90, self.rng))

    def _run(self):
        if not self.timer.isActive():
            self._last = time.perf_counter()
            self.timer.start()

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self._last, 0.05)
        self._last = now
        if self.handle == "down":
            t = min(1.0, (now - self._h0) / self.PULL_DOWN)
            self.pull = t * t
            if t >= 1:
                self._start_spin(now)
                self.handle, self._h0 = "up", now
        elif self.handle == "up":
            t = min(1.0, (now - self._h0) / self.SPRING_BACK)
            self.pull = max(0.0, 1 - ease_out_back(t, 2.2))      # snaps back past the top a bit
            if t >= 1:
                self.pull, self.handle = 0.0, "idle"
        for r in self.reels:
            r.update(now, dt)
        if self.spinning:
            for i, reel in enumerate(self.reels):
                if reel.state == "spin" and now - self._spin_t0 >= self.STOPS[i]:
                    reel.stop_on(self._outcome[i], now)
            if all(r.state == "idle" for r in self.reels):
                self._finish(now)
        self.update()
        if not self.spinning and self.handle in ("idle", "drag") and now > self.win_until:
            self.timer.stop()

    # -- mouse: click the handle, or drag the knob down -----------------------------
    def _on_handle(self, pos):
        g = self._geo()
        k = self._knob(g)
        return (abs(pos.x() - k.x()) < 16 and abs(pos.y() - k.y()) < 16) or \
            (pos.x() > g["body"].right() - 2 and g["top_y"] - 12 < pos.y() < g["bottom_y"] + 12)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self._on_handle(e.position()) and not self.spinning \
                and self.handle == "idle":
            self._press = e.position()
            self.handle = "drag"

    def mouseMoveEvent(self, e):
        self.setCursor(QtCore.Qt.PointingHandCursor if self._on_handle(e.position()) else QtCore.Qt.ArrowCursor)
        if self.handle == "drag" and self._press is not None:
            g = self._geo()
            self.pull = min(1.0, max(0.0, (e.position().y() - g["top_y"]) / (g["bottom_y"] - g["top_y"])))
            self.update()

    def mouseReleaseEvent(self, e):
        if self.handle != "drag":
            return
        moved = self._press is not None and abs(e.position().y() - self._press.y()) > 4
        self._press = None
        self._outcome = None
        now = time.perf_counter()
        if not moved:                          # a click: the full pull animation
            self.handle = "idle"
            self.pull_handle()
        elif self.pull > 0.55:                 # pulled far enough: spin
            self._start_spin(now)
            self.handle, self._h0 = "up", now - self.SPRING_BACK * (1 - self.pull) * 0.3
            self._run()
        else:                                  # let go too early: spring back, no spin
            self.handle, self._h0 = "up", now
            self._run()

    # -- drawing -----------------------------------------------------------------
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        g = self._geo()
        now = time.perf_counter()
        winning = now < self.win_until
        self._draw_body(p, g)
        self._draw_marquee(p, g, now, winning)
        self._draw_reels(p, g, now, winning)
        self._draw_lcd(p, g, now, winning)
        self._draw_handle(p, g)

    def _draw_body(self, p, g):
        b = g["body"]
        grad = QtGui.QLinearGradient(b.topLeft(), b.bottomLeft())
        grad.setColorAt(0, QtGui.QColor("#c0392b"))
        grad.setColorAt(1, QtGui.QColor("#6e1010"))
        p.setPen(QtGui.QPen(QtGui.QColor("#d4af37"), 2))
        p.setBrush(grad)
        p.drawRoundedRect(b, 10, 10)
        # handle slot on the right side of the body
        g2 = g["pivot"]
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#3a0a0a"))
        p.drawRoundedRect(QtCore.QRectF(b.right() - 2, g2.y() - 14, 10, 28), 3, 3)

    def _draw_marquee(self, p, g, now, winning):
        m = g["marquee"]
        p.setPen(QtGui.QPen(QtGui.QColor("#d4af37"), 1.5))
        p.setBrush(QtGui.QColor("#1b0b2e"))
        p.drawRoundedRect(m, 6, 6)
        f = QtGui.QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(max(8.0, m.height() * .36))
        p.setFont(f)
        p.setPen(GOLD if not winning or int(now * 8) % 2 else QtGui.QColor("#ffffff"))
        p.drawText(m, QtCore.Qt.AlignCenter, "LUCKY  LABELS")
        # bulbs along the marquee edge: a chase while spinning, flashing on a win
        n_top = 9
        pts = [QtCore.QPointF(m.left() + 6 + (m.width() - 12) * i / (n_top - 1), m.top() + 3) for i in range(n_top)]
        pts += [QtCore.QPointF(m.right() - 6 - (m.width() - 12) * i / (n_top - 1), m.bottom() - 3)
                for i in range(n_top)]
        step = int(now * (14 if self.spinning else 3))
        for i, q in enumerate(pts):
            if winning:
                on = int(now * 8) % 2 == 0
            else:
                on = (i + step) % 3 == 0
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#fff27a") if on else QtGui.QColor("#5a4a1a"))
            p.drawEllipse(q, 2.2, 2.2)

    def _draw_reels(self, p, g, now, winning):
        w = g["window"]
        p.setPen(QtGui.QPen(QtGui.QColor("#d4af37"), 2))
        p.setBrush(QtGui.QColor("#111111"))
        p.drawRoundedRect(w, 5, 5)
        cell = g["cell"]
        for reel, r in zip(self.reels, g["reels"]):
            p.save()
            p.setClipRect(r)
            p.fillRect(r, QtGui.QColor("#fbfbf6"))
            cy = r.center().y()
            k0 = math.floor(reel.pos)
            fast = reel.state == "spin" or (reel.state == "stopping" and (now - reel.t0) < reel.dur * .45)
            for k in range(k0 - 2, k0 + 3):
                y = cy - (k - reel.pos) * cell              # increasing pos moves symbols down
                rect = QtCore.QRectF(r.x() + cell * .1, y - cell * .4, cell * .8, cell * .8)
                if fast:                                     # motion blur: faded copies
                    for d, a in ((-.22, .25), (.22, .25), (0, .55)):
                        p.setOpacity(a)
                        draw_symbol(p, reel.symbol(k), rect.translated(0, d * cell))
                    p.setOpacity(1)
                else:
                    draw_symbol(p, reel.symbol(k), rect)
            # cylinder shading: darker at the top and bottom edges
            shade = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft())
            shade.setColorAt(0, QtGui.QColor(0, 0, 0, 150))
            shade.setColorAt(.28, QtGui.QColor(0, 0, 0, 0))
            shade.setColorAt(.72, QtGui.QColor(0, 0, 0, 0))
            shade.setColorAt(1, QtGui.QColor(0, 0, 0, 150))
            p.fillRect(r, shade)
            p.restore()
        # pay line
        y = g["reels"][0].center().y()
        if winning and self.win_symbol is not None and int(now * 8) % 2 == 0:
            p.setPen(QtGui.QPen(GOLD, 3))
            p.setBrush(QtCore.Qt.NoBrush)
            r0, r2 = g["reels"][0], g["reels"][2]
            p.drawRoundedRect(QtCore.QRectF(r0.left() - 2, y - cell * .48, r2.right() - r0.left() + 4, cell * .96), 4, 4)
        else:
            p.setPen(QtGui.QPen(QtGui.QColor(220, 30, 30, 140), 1.5))
            p.drawLine(QtCore.QPointF(w.left() + 2, y), QtCore.QPointF(w.right() - 2, y))

    def _draw_lcd(self, p, g, now, winning):
        r = g["lcd"]
        p.setPen(QtGui.QPen(QtGui.QColor("#808080"), 1))
        p.setBrush(QtGui.QColor("#0b1a0b"))
        p.drawRect(r)
        line1 = r.adjusted(3, 2, -3, -r.height() / 2)
        line2 = r.adjusted(3, r.height() / 2, -3, -2)
        p.setPen(GOLD if winning and self.win_symbol is not None else QtGui.QColor("#39ff6a"))
        self._fit_text(p, line1, self.msg, 10.0, True)
        if self.sub:
            self._fit_text(p, line2, self.sub, 9.5, True)
        else:
            p.setPen(QtGui.QColor("#1f9a3f"))
            self._fit_text(p, line2, f"spins {self.spins}  jackpots {self.jackpots}", 8.5, False)

    @staticmethod
    def _fit_text(p, rect, text, size, bold):
        """Monospace LCD text, shrunk until it fits the line."""
        f = QtGui.QFont("Courier New")
        f.setStyleHint(QtGui.QFont.Monospace)
        f.setBold(bold)
        while True:
            f.setPointSizeF(size)
            if QtGui.QFontMetricsF(f).horizontalAdvance(text) <= rect.width() or size <= 6:
                break
            size -= 0.5
        p.setFont(f)
        p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _draw_handle(self, p, g):
        piv, k = g["pivot"], self._knob(g)
        # the bar: thinner where it points at the viewer (around the pivot height)
        thick = 3 + 3 * min(1.0, abs(k.y() - piv.y()) / max(1.0, piv.y() - g["top_y"]))
        grad = QtGui.QLinearGradient(QtCore.QPointF(piv.x() - 3, 0), QtCore.QPointF(piv.x() + 3, 0))
        grad.setColorAt(0, QtGui.QColor("#6d6d6d"))
        grad.setColorAt(.5, QtGui.QColor("#f2f2f2"))
        grad.setColorAt(1, QtGui.QColor("#6d6d6d"))
        p.setPen(QtGui.QPen(QtGui.QBrush(grad), thick, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(piv, k)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#555555"))
        p.drawEllipse(piv, 5, 5)
        rad = QtGui.QRadialGradient(k + QtCore.QPointF(-3, -3), 10)
        rad.setColorAt(0, QtGui.QColor("#ff8a80"))
        rad.setColorAt(.5, QtGui.QColor("#e01818"))
        rad.setColorAt(1, QtGui.QColor("#7a0000"))
        p.setBrush(rad)
        p.setPen(QtGui.QPen(QtGui.QColor("#4a0000"), 1))
        p.drawEllipse(k, 8.5, 8.5)
