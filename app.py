import cv2
import mediapipe as mp
import math
import numpy as np
import av
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def pinch_value(hand, high):
    thumb = hand[4]
    index = hand[8]
    wrist = hand[0]
    mid = hand[9]
    pinch = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
    size = ((wrist.x - mid.x)**2 + (wrist.y - mid.y)**2)**0.5
    val = pinch / size
    LOW = 0.07
    HIGH = high
    normalized = (val - LOW) / (HIGH - LOW)
    normalized = max(0.0, min(1.0, normalized))
    return normalized


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

    # layered rose petals, deep rose -> near-white (colours are BGR)
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


class FlowerProcessor(VideoProcessorBase):
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE, num_hands=2)
        self.landmarker = HandLandmarker.create_from_options(options)
        self.grow = 0.0
        self.bloom = 0.0
        self.frame_count = 0
        self.glow_amt = 14
        self.smooth = 0.3
        self.high = 1.45

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # dreamy soft-focus pass over the camera feed
        soft = cv2.resize(cv2.resize(img, (w // 4, h // 4)), (w, h))
        img = cv2.addWeighted(img, 0.78, soft, 0.22, 6)

        layer = np.zeros_like(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        self.frame_count += 1

        lm = result.hand_landmarks
        for i, hand in enumerate(lm):
            for point in hand:
                px, py = int(point.x * w), int(point.y * h)
                cv2.circle(img, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 3, (203, 130, 255), -1, cv2.LINE_AA)
            val = pinch_value(hand, self.high)
            label = result.handedness[i][0].category_name
            if label == "Left":
                label = "Right"
                self.grow = self.smooth * val + (1 - self.smooth) * self.grow
            else:
                label = "Left"
                self.bloom = self.smooth * val + (1 - self.smooth) * self.bloom
            text = ("Grow" if label == "Right" else "Bloom") + " it <3"
            wx, wy = int(hand[0].x * w), int(hand[0].y * h)
            draw_pill(img, wx, wy, text)

        cx, cy = w // 2, h // 2
        sway = math.sin(self.frame_count * 0.04) * 0.06
        draw_flower(layer, cx, cy, self.grow, self.bloom,
                    phase=sway, fc=self.frame_count)

        # glow halo computed at half resolution to stay smooth at HD
        small = cv2.resize(layer, (w // 2, h // 2))
        glow = cv2.GaussianBlur(small, (0, 0), max(1, self.glow_amt // 2))
        glow = cv2.resize(glow, (w, h))
        img = cv2.addWeighted(img, 1.0, glow, 0.55, 0)
        # composite the flower over the feed (not additive, so colours stay rich)
        b, g, r = cv2.split(layer)
        mask = cv2.max(cv2.max(b, g), r)
        alpha = cv2.multiply(mask, np.array([4.25]))  # saturates at 255 = opaque
        alpha = alpha.astype(np.float32) * (1.0 / 255.0)
        img = cv2.blendLinear(img, layer, 1.0 - alpha, alpha)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Bloom for You", page_icon="🌸", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Quicksand:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(120deg, #ffd9ec 0%, #fff0f8 30%, #eadcff 65%, #ffe0ef 100%);
        background-size: 300% 300%;
        animation: dreamy 16s ease infinite;
        background-attachment: fixed;
    }
    @keyframes dreamy {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    html, body, [class*="css"], .stMarkdown, p, label, .stSlider {
        font-family: 'Quicksand', sans-serif !important;
        color: #7a3b62 !important;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .bloom-title {
        text-align: center;
        font-size: 3.4rem;
        margin: 0.1rem 0 0 0;
        line-height: 1.25;
    }
    .bloom-title .t-grad {
        font-family: 'Pacifico', cursive;
        background: linear-gradient(90deg, #ff5fa2, #ff8fc0, #c77dff, #ff5fa2);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shine 6s linear infinite;
        padding: 0 0.4rem;
    }
    @keyframes shine { to { background-position: 200% center; } }
    .bloom-sub {
        text-align: center;
        font-size: 1.12rem;
        color: #a25c86 !important;
        margin: 0.2rem 0 1.1rem 0;
        letter-spacing: 0.03em;
    }

    .bloom-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1.5px solid rgba(255, 201, 226, 0.9);
        border-radius: 26px;
        padding: 1.1rem 1.3rem 1rem 1.3rem;
        margin: 0.4rem 0 1.3rem 0;
        box-shadow: 0 10px 34px rgba(255, 140, 190, 0.28);
    }
    .chips { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    .chip {
        flex: 1 1 240px;
        max-width: 380px;
        background: rgba(255, 255, 255, 0.75);
        border: 1.5px solid #ffd3e7;
        border-radius: 20px;
        padding: 0.8rem 1rem;
        text-align: center;
        box-shadow: 0 5px 16px rgba(255, 150, 200, 0.18);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .chip:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(255, 120, 180, 0.3); }
    .chip .big { font-size: 1.9rem; display: block; margin-bottom: 0.15rem; }
    .chip b { color: #ff5fa2; }
    .hint {
        text-align: center;
        margin-top: 0.85rem;
        font-size: 0.95rem;
        color: #a25c86;
    }

    /* the webrtc component (iframe) — big, rounded stage */
    .stApp iframe {
        border-radius: 28px;
        display: block;
        margin: 0 auto;
    }

    .bloom-foot {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 1rem;
        color: #b06a92 !important;
    }
    .bloom-foot .beat { display: inline-block; animation: beat 1.6s ease infinite; }
    @keyframes beat { 0%, 100% { transform: scale(1); } 25% { transform: scale(1.3); } }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0f7 0%, #f7e8ff 100%);
        border-right: 1.5px solid rgba(255, 201, 226, 0.7);
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff8ec2, #ff6fae) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.4rem !important;
        box-shadow: 0 6px 18px rgba(255, 120, 180, 0.45);
        transition: transform 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-2px) scale(1.03); }

    /* floating petals & hearts */
    .fl {
        position: fixed;
        bottom: -60px;
        z-index: 0;
        pointer-events: none;
        opacity: 0;
        animation-name: rise;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
    }
    @keyframes rise {
        0% { transform: translateY(0) rotate(0deg); opacity: 0; }
        12% { opacity: 0.55; }
        85% { opacity: 0.3; }
        100% { transform: translateY(-108vh) rotate(320deg); opacity: 0; }
    }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

_rng = random.Random(7)
_floaters = "".join(
    f'<span class="fl" style="left:{_rng.randint(2, 96)}%;'
    f'font-size:{_rng.randint(14, 30)}px;'
    f'animation-duration:{_rng.uniform(9, 19):.1f}s;'
    f'animation-delay:{_rng.uniform(0, 14):.1f}s;">{e}</span>'
    for e in ["🌸", "💗", "🌷", "✨", "💕", "🌺", "🌸", "💖", "✨", "🌸", "💞", "🩷"]
)
st.markdown(_floaters, unsafe_allow_html=True)

st.markdown(
    '<div class="bloom-title">🌸 <span class="t-grad">Bloom for You</span> 🌸</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="bloom-sub">a little flower that grows with your hands 💕</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="bloom-card">
        <div class="chips">
            <div class="chip">
                <span class="big">🌱</span>
                <b>Right hand</b><br>
                pinch &amp; open to make it <b>grow</b>
            </div>
            <div class="chip">
                <span class="big">🌷</span>
                <b>Left hand</b><br>
                pinch &amp; open to make it <b>bloom</b>
            </div>
        </div>
        <div class="hint">press <b>START</b> below, allow the camera, and show both hands ✨</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Secret tweaks")
    st.caption("only if you want to fiddle 🎀")
    glow = st.slider("glow ✨", 1, 40, 14)
    smooth = st.slider("smoothness 🌊", 0.0, 1.0, 0.3)
    high = st.slider("sensitivity 🤏", 0.5, 3.0, 1.45)

ctx = webrtc_streamer(
    key="flower",
    video_processor_factory=FlowerProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        muted=True,
        style={
            "width": "100%",
            "borderRadius": "26px",
            "border": "3px solid rgba(255,255,255,0.85)",
            "boxShadow": "0 16px 50px rgba(255,105,170,0.35)",
        },
    ),
)

if ctx.video_processor:
    ctx.video_processor.glow_amt = glow
    ctx.video_processor.smooth = smooth
    ctx.video_processor.high = high

st.markdown(
    '<div class="bloom-foot">made with <span class="beat">💖</span> just for you</div>',
    unsafe_allow_html=True,
)
