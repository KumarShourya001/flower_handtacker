"""Flower artwork for the Bloom for You app.

Pure OpenCV/NumPy drawing, deliberately free of Streamlit/MediaPipe so the flower
can be rendered to an image and tuned without a webcam (see render_bloom.py)."""
import cv2
import math
import numpy as np


def clamp01(v):
    return max(0.0, min(1.0, v))


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def petal_band(length, width, t0, t1, n=8):
    """Slice of a teardrop petal (base at origin, tip along +x)."""
    ts = np.linspace(t0, t1, n)
    prof = (np.sin(np.pi * ts) ** 0.85) * (width * 0.5)
    xs = ts * length
    top = np.stack([xs, -prof], axis=1)
    bottom = np.stack([xs[::-1], prof[::-1]], axis=1)
    return np.concatenate([top, bottom])


def draw_petal(img, cx, cy, angle, length, width, base_col, tip_col, edge_col=None):
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
        t1 = min(1.0, (k + 1) / bands + 0.03)
        col = lerp_color(base_col, tip_col, k / (bands - 1))
        put(petal_band(length, width, t0, t1), col)
    if edge_col is None:
        edge_col = lerp_color(base_col, (90, 20, 130), 0.45)
    put(petal_band(length, width, 0.0, 1.0, n=18), edge_col, fill=False)
    hi = lerp_color(tip_col, (255, 255, 255), 0.25)
    put(petal_band(length * 0.9, width * 0.15, 0.2, 0.8), hi)


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
        col = (200, 240, 255) if k % 3 else (255, 225, 250)
        cv2.line(img, (sx - s, sy), (sx + s, sy), col, 1, cv2.LINE_AA)
        cv2.line(img, (sx, sy - s), (sx, sy + s), col, 1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 1, (255, 255, 255), -1, cv2.LINE_AA)


def draw_flower_head(img, cx, cy, base, openness, phase=0.0, fc=0, size=1.0):
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
        for k in range(petals):
            ang = offset + phase * (1 + i * 0.15) + k * 2 * math.pi / petals
            draw_petal(img, cx, cy, ang, pl, pw, cb, ct)

    # golden heart of the flower with tiny stamens
    cr = max(3, int(base * 0.16))
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


def draw_flower(img, cx, cy, grow, bloom, phase=0.0, fc=0):
    h, w = img.shape[:2]
    size = clamp01(grow)
    bloom = clamp01(bloom)
    openness = 0.25 + bloom * 0.75
    base = min(w, h) * (0.09 + 0.20 * size)

    # curved, gently swaying stem (quadratic bezier down to the bottom edge)
    bend = phase * 220
    p0 = np.array([cx, cy], dtype=float)
    p1 = np.array([cx + bend, (cy + h) * 0.55], dtype=float)
    p2 = np.array([cx + bend * 0.4, h + 10], dtype=float)
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
        for sgn, up, out in ((-1, 0.30, 1.52), (1, 0.14, 1.66)):
            hb = base * (0.46 + 0.10 * grown) * (0.5 + 0.5 * grown)
            hx = int(cx + sgn * base * out)
            hy = int(cy + base * up)
            _branch_stem(img, (int(cx + sgn * stem_w), int(cy + base * 1.75)),
                         (hx, hy), max(2, stem_w // 2))
            for k in range(5):
                a = phase + k * 2 * math.pi / 5
                draw_petal(img, hx, hy, a, hb, hb * 0.28,
                           (48, 128, 58), (88, 182, 98), (35, 95, 45))
            draw_flower_head(img, hx, hy, hb, openness * 0.9,
                             phase=phase, fc=fc + sgn * 23, size=size * 0.8)

    # veined leaves on alternating sides of the stem
    for tpos, sgn in ((0.30, -1), (0.52, 1)):
        idx = int(tpos * (len(curve) - 1))
        lx, ly = int(curve[idx][0]), int(curve[idx][1])
        ang = 0.55 if sgn > 0 else math.pi - 0.55
        ll = base * (0.55 + size * 0.30)
        lw = ll * 0.42
        draw_petal(img, lx, ly, ang, ll, lw,
                   (55, 140, 62), (110, 210, 120), (38, 100, 45))
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

    # a little heart floats up when the flower is fully bloomed
    if bloom > 0.8:
        fade = (bloom - 0.8) / 0.2
        hb = max(3, int(base * 0.16 * (0.5 + 0.5 * fade)))
        hx = cx
        hy = int(cy - base * 1.30 + math.sin(fc * 0.1) * 6)
        col = (150, 90, 255)
        cv2.circle(img, (hx - hb // 2, hy), hb // 2, col, -1, cv2.LINE_AA)
        cv2.circle(img, (hx + hb // 2, hy), hb // 2, col, -1, cv2.LINE_AA)
        pts = np.array([[hx - hb, hy], [hx + hb, hy], [hx, hy + int(hb * 1.25)]])
        cv2.fillPoly(img, [pts], col, lineType=cv2.LINE_AA)

    draw_sparkles(img, cx, cy, base, 0.3 * size + 0.7 * bloom, fc)


def draw_pill(img, x, y, text):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    pad = 7
    x0, y0, x1, y1 = x, y - th - pad, x + tw, y + pad
    col = (185, 105, 235)
    rad = (y1 - y0) // 2
    cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
    cv2.circle(img, (x0, y0 + rad), rad, col, -1, cv2.LINE_AA)
    cv2.circle(img, (x1, y0 + rad), rad, col, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
