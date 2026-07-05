import cv2
import mediapipe as mp
import math
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_petal(img, cx, cy, angle, length, width, color, edge=None):
    ex = int(cx + math.cos(angle) * length * 0.55)
    ey = int(cy + math.sin(angle) * length * 0.55)
    deg = math.degrees(angle)
    axes = (max(1, int(length * 0.55)), max(1, int(width * 0.5)))
    if edge is not None:
        cv2.ellipse(img, (ex, ey), axes, deg, 0, 360, edge, -1)
        axes = (max(1, int(axes[0] * 0.86)), max(1, int(axes[1] * 0.82)))
    cv2.ellipse(img, (ex, ey), axes, deg, 0, 360, color, -1)
    hi = lerp_color(color, (255, 255, 255), 0.30)
    crease = (max(1, int(axes[0] * 0.62)), max(1, int(axes[1] * 0.22)))
    cv2.ellipse(img, (ex, ey), crease, deg, 0, 360, hi, -1)


def draw_flower(img, cx, cy, grow, bloom, phase=0.0):
    h = img.shape[0]
    size = max(0.0, min(1.0, grow))
    base = 45 + size * 95
    openness = 0.25 + bloom * 0.75
    stem_w = max(2, int(4 + size * 8))
    cv2.line(img, (cx, cy), (cx, h), (60, 150, 70), stem_w)
    cv2.line(img, (cx - stem_w // 3, cy), (cx - stem_w // 3, h),
             (90, 185, 100), max(1, stem_w // 3))
    leaf_y = cy + int(110 + size * 70)
    leaf_len = int(28 + size * 34)
    leaf_wid = int(13 + size * 12)
    for sgn in (-1, 1):
        lx = cx + sgn * int(30 + size * 34)
        cv2.ellipse(img, (lx, leaf_y), (leaf_len, leaf_wid),
                    sgn * 32, 0, 360, (70, 165, 80), -1)
        cv2.ellipse(img, (lx, leaf_y), (leaf_len, leaf_wid),
                    sgn * 32, 0, 360, (110, 205, 120), 1)
        cv2.ellipse(img, (lx, leaf_y), (int(leaf_len * 0.85), 1),
                    sgn * 32, 0, 360, (110, 205, 120), 1)
    # soft rose petals, layered light -> deep (colours are BGR)
    shadow = (150, 60, 110)
    rings = [
        (1.00, 0.00, (192, 133, 255)),   # blush pink
        (0.80, math.pi / 5, (204, 153, 255)),
        (0.62, 0.00, (220, 182, 255)),   # petal pink
        (0.46, math.pi / 5, (235, 214, 255)),  # near-white pink
    ]
    for i, (scale, offset, color) in enumerate(rings):
        petals = 5 if i >= 2 else 7
        ring_open = 0.35 + openness * (0.65 - i * 0.12)
        pl = base * scale * (0.45 + ring_open * 0.55)
        pw = pl * (0.66 - i * 0.05)
        for k in range(petals):
            angle = offset + phase + k * (2 * math.pi / petals)
            draw_petal(img, cx, cy, angle, pl, pw, color, edge=shadow)
    # golden heart of the flower
    cr = int(8 + size * 16)
    cv2.circle(img, (cx, cy), cr, (60, 205, 255), -1)
    cv2.circle(img, (cx, cy), int(cr * 0.62), (40, 165, 240), -1)
    for k in range(3):
        a = phase + k * (2 * math.pi / 3)
        sx = int(cx + math.cos(a) * cr * 0.4)
        sy = int(cy + math.sin(a) * cr * 0.4)
        cv2.ellipse(img, (sx, sy), (max(1, int(cr * 0.5)), max(1, int(cr * 0.22))),
                    math.degrees(a), 0, 360, (30, 140, 230), -1)


class FlowerProcessor(VideoProcessorBase):
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE, num_hands=2)
        self.landmarker = HandLandmarker.create_from_options(options)
        self.grow = 0.0
        self.bloom = 0.0
        self.frame_count = 0
        self.glow_amt = 15
        self.smooth = 0.3
        self.high = 1.45

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        layer = np.zeros_like(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        self.frame_count += 1

        lm = result.hand_landmarks
        for i, hand in enumerate(lm):
            for point in hand:
                px, py = int(point.x * w), int(point.y * h)
                cv2.circle(img, (px, py), 3, (203, 130, 255), -1)
            val = pinch_value(hand, self.high)
            label = result.handedness[i][0].category_name
            if label == "Left":
                label = "Right"
                self.grow = self.smooth * val + (1 - self.smooth) * self.grow
            else:
                label = "Left"
                self.bloom = self.smooth * val + (1 - self.smooth) * self.bloom
            text = ("Grow" if label == "Right" else "Bloom") + " it <3"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            wx, wy = int(hand[0].x * w), int(hand[0].y * h)
            cv2.rectangle(img, (wx - 6, wy - th - 9), (wx + tw + 6, wy + 6), (170, 90, 220), -1)
            cv2.putText(img, text, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cx, cy = w // 2, h // 2
        sway = math.sin(self.frame_count * 0.04) * 0.08
        draw_flower(layer, cx, cy, self.grow, self.bloom, phase=sway)
        glow = cv2.GaussianBlur(layer, (0, 0), max(1, self.glow_amt))
        img = cv2.addWeighted(img, 1.0, glow, 0.8, 0)
        img = cv2.addWeighted(img, 1.0, layer, 1.0, 0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Bloom for You", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Quicksand:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #ffe3f1 0%, #fff2f8 45%, #f4e8ff 100%);
        background-attachment: fixed;
    }
    html, body, [class*="css"], .stMarkdown, p, label, .stSlider {
        font-family: 'Quicksand', sans-serif !important;
        color: #7a3b62 !important;
    }
    .bloom-title {
        font-family: 'Pacifico', cursive;
        font-size: 3rem;
        text-align: center;
        color: #ff6fae;
        text-shadow: 0 2px 12px rgba(255,150,200,0.45);
        margin: 0.2rem 0 0.1rem 0;
    }
    .bloom-sub {
        text-align: center;
        font-size: 1.05rem;
        color: #a25c86 !important;
        margin-bottom: 1.2rem;
    }
    .bloom-card {
        background: rgba(255,255,255,0.55);
        border: 1px solid #ffc9e2;
        border-radius: 22px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0 1.2rem 0;
        box-shadow: 0 6px 22px rgba(255,150,200,0.25);
        text-align: center;
    }
    .bloom-foot {
        text-align: center;
        margin-top: 1.4rem;
        font-size: 0.95rem;
        color: #b06a92 !important;
    }
    /* pink slider accents */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: #ff6fae !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0f7 0%, #f7e8ff 100%);
    }
    /* start/stop webrtc buttons */
    .stButton>button {
        background: #ff8ec2 !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(255,140,190,0.4);
    }
    .stButton>button:hover { background: #ff6fae !important; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="bloom-title">🌸 Bloom for You 🌸</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="bloom-sub">a little flower that grows with your hands 💕</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="bloom-card">
        🌱 <b>Right hand</b> pinch &amp; open → makes it <b>grow</b><br>
        🌷 <b>Left hand</b> pinch &amp; open → makes it <b>bloom</b><br>
        <span style="font-size:0.9rem;">press <b>Start</b>, allow the camera, and show both hands ✨</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Secret tweaks")
    st.caption("only if you want to fiddle 🎀")
    glow = st.slider("glow ✨", 1, 50, 15)
    smooth = st.slider("smoothness 🌊", 0.0, 1.0, 0.3)
    high = st.slider("sensitivity 🤏", 0.5, 3.0, 1.45)

ctx = webrtc_streamer(key="flower", video_processor_factory=FlowerProcessor)

if ctx.video_processor:
    ctx.video_processor.glow_amt = glow
    ctx.video_processor.smooth = smooth
    ctx.video_processor.high = high

st.markdown(
    '<div class="bloom-foot">made with 💖 just for you</div>',
    unsafe_allow_html=True,
)
