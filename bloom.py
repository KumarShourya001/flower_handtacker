"""Flower artwork for the Bloom for You app.

Pure OpenCV/NumPy drawing, deliberately free of Streamlit/MediaPipe: draw_flower
takes a plain image, so the whole thing can be rendered to a still and fiddled
with over a cup of tea, no webcam involved."""
import cv2
import math
import numpy as np


def clamp01(v):
    return max(0.0, min(1.0, v))


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def wobble(seed):
    """Repeatable -1..1 jitter, so every petal keeps its own little quirk."""
    x = math.sin(seed * 12.9898) * 43758.5453
    return 2.0 * (x - math.floor(x)) - 1.0


_shapes = {}


def _unit(t0, t1, n):
    """Unit-size petal profile. There are only a handful of distinct slices and
    they get rebuilt for every petal of every frame, so they are worth keeping."""
    key = (t0, t1, n)
    u = _shapes.get(key)
    if u is None:
        ts = np.linspace(t0, t1, n)
        u = (ts, (np.sin(np.pi * ts) ** 0.85) * 0.5, ts * ts)
        _shapes[key] = u
    return u


def petal_band(length, width, t0, t1, n=8, curl=0.0):
    """Slice of a teardrop petal (base at origin, tip along +x)."""
    ts, unit, sq = _unit(t0, t1, n)
    prof = unit * width
    mid = sq * (curl * length)  # petals lean instead of standing to attention
    xs = ts * length
    top = np.stack([xs, mid - prof], axis=1)
    bottom = np.stack([xs[::-1], mid[::-1] + prof[::-1]], axis=1)
    return np.concatenate([top, bottom])


def petal_spine(length, t0, t1, curl=0.0, n=10):
    ts, _, sq = _unit(t0, t1, n)
    return np.stack([ts * length, sq * (curl * length)], axis=1)


def draw_petal(img, cx, cy, angle, length, width, base_col, tip_col,
               edge_col=None, curl=0.0):
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    def put(pts, col, fill=True, thick=1):
        p = (pts @ rot.T + (cx, cy)).astype(np.int32)
        if fill:
            cv2.fillPoly(img, [p], col, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(img, [p], True, col, thick, cv2.LINE_AA)

    bands = 4
    for k in range(bands):
        t0 = k / bands
        t1 = min(1.0, (k + 1) / bands + 0.04)
        col = lerp_color(base_col, tip_col, k / (bands - 1))
        put(petal_band(length, width, t0, t1, curl=curl), col)
    if edge_col is None:
        edge_col = lerp_color(base_col, (90, 20, 130), 0.45)
    put(petal_band(length, width, 0.0, 1.0, n=18, curl=curl), edge_col, fill=False)

    # a crease down the middle and a thin sheen beside it: reads as a fold of
    # silk rather than a flat sticker. Too small to see on the little petals,
    # and there are a great many of those, so don't bother drawing it there.
    if length < 14:
        return
    crease = lerp_color(base_col, (75, 20, 115), 0.30)
    p = (petal_spine(length * 0.86, 0.06, 1.0, curl) @ rot.T + (cx, cy)).astype(np.int32)
    cv2.polylines(img, [p], False, crease, 1, cv2.LINE_AA)
    sheen = lerp_color(tip_col, (255, 255, 255), 0.28)
    put(petal_band(length * 0.66, width * 0.13, 0.22, 0.78, curl=curl * 1.2), sheen)


def draw_sparkles(img, cx, cy, radius, amount, fc):
    for k in range(14):
        tw = 0.5 + 0.5 * math.sin(fc * 0.12 + k * 1.7)
        if tw * amount < 0.22:
            continue
        a = k * 2.39996 + fc * 0.004
        r = radius * (1.15 + 0.55 * ((k * 37 % 10) / 10.0))
        sx = int(cx + math.cos(a) * r)
        sy = int(cy + math.sin(a) * r * 0.8)
        s = max(2, int((2 + 5 * tw) * amount))
        col = (235, 225, 255) if k % 3 else (255, 235, 250)
        cv2.line(img, (sx - s, sy), (sx + s, sy), col, 1, cv2.LINE_AA)
        cv2.line(img, (sx, sy - s), (sx, sy + s), col, 1, cv2.LINE_AA)
        cv2.line(img, (sx - s // 3, sy - s // 3), (sx + s // 3, sy + s // 3),
                 col, 1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 1, (255, 255, 255), -1, cv2.LINE_AA)


def draw_flower_head(img, cx, cy, base, openness, phase=0.0, fc=0, size=1.0, seed=0.0):
    """Just the rose head (petals + golden centre) at (cx, cy)."""
    rings = [
        (1.00, 8, 0.0,           (150, 60, 210), (205, 130, 250)),
        (0.78, 8, math.pi / 8,   (170, 95, 235), (220, 165, 255)),
        (0.56, 6, 0.0,           (205, 150, 250), (238, 205, 255)),
        (0.36, 5, math.pi / 5,   (228, 196, 255), (250, 238, 255)),
    ]
    for i, (scale, petals, offset, cb, ct) in enumerate(rings):
        ring_open = clamp01(0.30 + openness * (0.70 - i * 0.13))
        pl = base * scale * (0.40 + ring_open * 0.60)
        pw = pl * (0.58 - i * 0.04)
        curl = (0.10 + 0.16 * (1.0 - openness)) * (1 if i % 2 else -1)
        for k in range(petals):
            n = seed + i * 9.1 + k
            ang = offset + phase * (1 + i * 0.15) + k * 2 * math.pi / petals
            ang += 0.07 * wobble(n)
            draw_petal(img, cx, cy, ang,
                       pl * (1 + 0.09 * wobble(n + 3.3)),
                       pw * (1 + 0.13 * wobble(n + 7.7)),
                       cb, ct, curl=curl * (1 + 0.3 * wobble(n + 5.5)))

    # petals tuck into a little shaded well before the golden heart
    cr = max(3, int(base * 0.16))
    cv2.circle(img, (cx, cy), int(cr * 1.5), (168, 100, 216), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(cr * 1.25), (110, 185, 250), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), cr, (60, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(cr * 0.55), (35, 160, 240), -1, cv2.LINE_AA)
    if size > 0.15:
        for k in range(14):
            a = phase + k * 2 * math.pi / 14
            r2 = cr * (0.75 + 0.20 * math.sin(fc * 0.09 + k))
            x2 = int(cx + math.cos(a) * r2)
            y2 = int(cy + math.sin(a) * r2)
            cv2.line(img, (cx, cy), (x2, y2), (40, 160, 245), 1, cv2.LINE_AA)
            cv2.circle(img, (x2, y2), max(1, cr // 7), (120, 235, 255), -1, cv2.LINE_AA)


def _branch_stem(img, p0, p2, w):
    """Short curved stem from p0 (on the main stem) out to p2 (a side bloom)."""
    p0 = np.array(p0, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = np.array([p2[0], p0[1]], dtype=float)  # bows out then rises
    ts = np.linspace(0, 1, 16)[:, None]
    curve = ((1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2).astype(np.int32)
    cv2.polylines(img, [curve], False, (45, 118, 55), w + 2, cv2.LINE_AA)
    cv2.polylines(img, [curve], False, (70, 165, 80), max(1, w), cv2.LINE_AA)


def _dew(img, x, y, r):
    """A little jewel-like water droplet catching the light."""
    cv2.circle(img, (x, y), r, (248, 244, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y + r // 3), max(1, r // 2), (255, 190, 235), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x - r // 3, y - r // 3), max(1, r // 3),
               (255, 255, 255), -1, cv2.LINE_AA)


def _heart(img, x, y, r, col):
    cv2.circle(img, (x - r // 2, y), max(1, r // 2), col, -1, cv2.LINE_AA)
    cv2.circle(img, (x + r // 2, y), max(1, r // 2), col, -1, cv2.LINE_AA)
    pts = np.array([[x - r, y], [x + r, y], [x, y + int(r * 1.25)]])
    cv2.fillPoly(img, [pts], col, lineType=cv2.LINE_AA)


class Drifting:
    """Petals that let go of a wide-open bloom and float down through the frame."""

    def __init__(self, limit=12):
        self.petals = []
        self.limit = limit
        self.n = 0

    def step(self, img, cx, cy, base, amount, fc):
        h = img.shape[0]
        if amount > 0.55 and len(self.petals) < self.limit and fc % 9 == 0:
            self.n += 1
            a = wobble(self.n * 2.7) * math.pi
            self.petals.append([
                cx + math.cos(a) * base * 0.7,      # x
                cy + math.sin(a) * base * 0.5,      # y
                wobble(self.n * 5.1) * 0.9,         # sideways drift
                base * (0.10 + 0.05 * abs(wobble(self.n * 3.3))),  # size
                wobble(self.n * 7.9) * math.pi,     # angle
                0.03 + 0.02 * abs(wobble(self.n * 11.3)),          # spin
                0.0,                                # age 0..1
                0.006 + 0.003 * abs(wobble(self.n * 4.4)),         # ageing rate
            ])

        keep = []
        for p in self.petals:
            p[6] += p[7]
            if p[6] >= 1.0 or p[1] > h + 40:
                continue
            fall = p[6] * p[6]
            p[1] += 1.1 + 4.5 * fall
            p[0] += math.sin(p[6] * 9.0 + p[4]) * 1.6 + p[2]
            p[4] += p[5]
            # they thin out into light rather than darkening: the frame is
            # composited on brightness, so a dim petal would read as a smudge
            pale = clamp01(p[6] * 1.4)
            ln = max(2.0, p[3] * (1.0 - 0.75 * p[6] ** 2))
            draw_petal(img, int(p[0]), int(p[1]), p[4], ln, ln * 0.5,
                       lerp_color((160, 75, 215), (238, 205, 255), pale),
                       lerp_color((220, 160, 252), (252, 238, 255), pale),
                       edge_col=lerp_color((130, 50, 185), (232, 196, 252), pale),
                       curl=0.25)
            keep.append(p)
        self.petals = keep


def draw_flower(img, cx, cy, grow, bloom, phase=0.0, fc=0, root=None, drift=None):
    h, w = img.shape[:2]
    size = clamp01(grow)
    bloom = clamp01(bloom)
    openness = 0.25 + bloom * 0.75
    base = min(w, h) * (0.09 + 0.20 * size)

    # curved, gently swaying stem. When a hand is holding it, the curve is pulled
    # through the palm so the stem really does run between the fingers.
    bend = phase * 220
    p0 = np.array([cx, cy], dtype=float)
    if root is None:
        p1 = np.array([cx + bend, (cy + h) * 0.55], dtype=float)
        p2 = np.array([cx + bend * 0.4, h + 10], dtype=float)
    else:
        p1 = np.array([root[0] + bend * 0.35, root[1]], dtype=float)
        p2 = np.array([root[0] + bend * 0.9 + (root[0] - cx) * 1.6, h + 10], dtype=float)
    ts = np.linspace(0, 1, 24)[:, None]
    curve = ((1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2).astype(np.int32)
    stem_w = max(3, int(4 + size * 9))
    cv2.polylines(img, [curve], False, (45, 118, 55), stem_w + 3, cv2.LINE_AA)
    cv2.polylines(img, [curve], False, (70, 165, 80), stem_w, cv2.LINE_AA)
    cv2.polylines(img, [curve], False, (125, 215, 135), max(1, stem_w // 3), cv2.LINE_AA)

    # companion blooms on short side branches — they emerge and open as the
    # plant grows, turning the single rose into a little spray. Drawn before the
    # main head so the hero flower stays in front where they overlap.
    grown = clamp01((size - 0.28) / 0.55)
    if grown > 0.02:
        for sgn, up, out in ((-1, 0.46, 1.46), (1, 0.28, 1.60)):
            hb = base * (0.36 + 0.09 * grown) * (0.5 + 0.5 * grown)
            hx = int(cx + sgn * base * out)
            hy = int(cy + base * up)
            _branch_stem(img, (int(cx + sgn * stem_w), int(cy + base * 1.75)),
                         (hx, hy), max(2, stem_w // 2))
            for k in range(5):
                a = phase + k * 2 * math.pi / 5
                draw_petal(img, hx, hy, a, hb, hb * 0.28,
                           (48, 128, 58), (88, 182, 98), (35, 95, 45))
            draw_flower_head(img, hx, hy, hb, openness * 0.9, phase=phase,
                             fc=fc + sgn * 23, size=size * 0.8, seed=17.0 + sgn * 6)

    # veined leaves on alternating sides of the stem
    for tpos, sgn in ((0.30, -1), (0.52, 1)):
        idx = int(tpos * (len(curve) - 1))
        lx, ly = int(curve[idx][0]), int(curve[idx][1])
        ang = 0.55 if sgn > 0 else math.pi - 0.55
        ll = base * (0.55 + size * 0.30)
        lw = ll * 0.42
        draw_petal(img, lx, ly, ang, ll, lw,
                   (55, 140, 62), (110, 210, 120), (38, 100, 45), curl=0.18 * sgn)
        ex = int(lx + math.cos(ang) * ll * 0.82)
        ey = int(ly + math.sin(ang) * ll * 0.82)
        cv2.line(img, (lx, ly), (ex, ey), (48, 122, 55), 2, cv2.LINE_AA)

    # sepals peeking out beneath the petals
    for k in range(6):
        a = phase + math.pi / 6 + k * math.pi / 3
        draw_petal(img, cx, cy, a, base * 1.04, base * 0.30,
                   (48, 128, 58), (88, 182, 98), (35, 95, 45))

    draw_flower_head(img, cx, cy, base, openness, phase=phase, fc=fc, size=size)

    # dew drops catching the light on the petals
    if size > 0.4:
        for dx_f, dy_f in ((-0.32, 0.44), (0.5, 0.3)):
            dx = int(cx + base * dx_f)
            dy = int(cy + base * dy_f)
            _dew(img, dx, dy, max(2, int(base * 0.05)))

    # once it is properly open, little hearts keep drifting up out of it
    if bloom > 0.72:
        fade = clamp01((bloom - 0.72) / 0.24)
        for k in range(3):
            t = (fc * 0.011 + k / 3.0) % 1.0
            hx = int(cx + math.sin(t * 5.5 + k * 2.1) * base * 0.24)
            hy = int(cy - base * (1.18 + t * 1.15))
            hr = max(2, int(base * 0.15 * (0.55 + 0.45 * fade) * (1.0 - 0.3 * t)))
            _heart(img, hx, hy, hr,
                   lerp_color((0, 0, 0), (150, 90, 255), fade * (1.0 - t) * 1.4))

    if drift is not None:
        drift.step(img, cx, cy, base, bloom * size, fc)

    draw_sparkles(img, cx, cy, base, 0.3 * size + 0.7 * bloom, fc)


def draw_pill(img, x, y, text):
    """Soft translucent label that stays inside the frame."""
    h, w = img.shape[:2]
    fs = max(0.45, w / 1600.0 * 0.7)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    pad = int(9 * fs / 0.55)
    bw, bh = tw + pad * 2, th + pad * 2
    x0 = min(max(4, x - pad), w - bw - 4)
    y0 = min(max(4, y - bh), h - bh - 4)
    if bw + 8 > w or bh + 8 > h:
        return
    roi = img[y0:y0 + bh, x0:x0 + bw]
    pill = np.zeros_like(roi)
    r = bh // 2
    cv2.rectangle(pill, (r, 0), (bw - r, bh), (255, 255, 255), -1)
    cv2.circle(pill, (r, r), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(pill, (bw - r, r), r, (255, 255, 255), -1, cv2.LINE_AA)
    m = pill[..., :1].astype(np.float32) * (0.78 / 255.0)
    roi[:] = (roi * (1 - m) + np.float32((190, 110, 240)) * m).astype(np.uint8)
    cv2.putText(roi, text, (pad, bh - pad), font, fs, (255, 245, 252), 1, cv2.LINE_AA)


_notes = {}


def _note_art(text, w, h, y, scale):
    """Paint the words once: a dark halo so they read over a bright room, a warm
    glow, then cream ink. Flattened to one colour layer plus a coverage mask, so
    showing it every frame is a single blend."""
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    fs = scale if scale else w / 780.0
    thick = max(1, int(round(fs * 1.5)))
    (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
    pad = int(26 * fs) + 12
    x = (w - tw) // 2
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(w, x + tw + pad), min(h, y + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    ink = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.putText(ink, text, (x - x0, y - y0), font, fs, 255, thick, cv2.LINE_AA)
    ink = ink.astype(np.float32) * (1.0 / 255.0)
    halo = cv2.GaussianBlur(ink, (0, 0), max(2.0, 8 * fs)) * 0.62
    glow = cv2.GaussianBlur(ink, (0, 0), max(1.0, 3 * fs)) * 0.75

    rgb = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    cover = np.zeros_like(ink)
    for col, a in (((38, 8, 26), halo), ((255, 205, 245), glow), ((252, 244, 255), ink)):
        a3 = a[..., None]
        rgb = np.float32(col) * a3 + rgb * (1 - a3)
        cover = a + cover * (1 - a)
    return x0, y0, x1, y1, rgb.astype(np.uint8), cover


def draw_note(img, text, y=None, alpha=1.0, scale=None):
    """A line of handwriting laid softly over the picture."""
    h, w = img.shape[:2]
    alpha = clamp01(alpha)
    if alpha <= 0.01 or not text:
        return
    y = int(h * 0.90) if y is None else y
    key = (text, w, h, y, scale)
    if key not in _notes:
        if len(_notes) > 16:
            _notes.clear()
        _notes[key] = _note_art(text, w, h, y, scale)
    art = _notes[key]
    if art is None:
        return
    x0, y0, x1, y1, rgb, cover = art
    a = cover * alpha
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.blendLinear(roi, rgb, 1.0 - a, a)
