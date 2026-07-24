import os
import cv2
import mediapipe as mp
import math
import numpy as np
import av
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes
from bloom import draw_flower, draw_pill

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
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

def overlay_png(bg, png, cx, cy, width, angle_deg):
    oh, ow = png.shape[:2]
    scale = width / ow
    img = cv2.resize(png, (max(1, int(ow*scale)), max(1, int(oh*scale))),
                     interpolation=cv2.INTER_AREA)
    nh, nw = img.shape[:2]
    M = cv2.getRotationMatrix2D((nw/2, nh/2), -angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    bw, bh = int(nh*sin + nw*cos), int(nh*cos + nw*sin)
    M[0,2] += bw/2 - nw/2
    M[1,2] += bh/2 - nh/2
    img = cv2.warpAffine(img, M, (bw, bh), flags=cv2.INTER_LINEAR,
                         borderValue=(0,0,0,0))
    x0, y0 = int(cx - bw/2), int(cy - bh/2)
    for c in range(3):
        y1, x1 = min(y0+bh, bg.shape[0]), min(x0+bw, bg.shape[1])
        ys, xs = max(0,-y0), max(0,-x0)
        y0c, x0c = max(0,y0), max(0,x0)
        a = img[ys:ys+(y1-y0c), xs:xs+(x1-x0c), 3:4] / 255.0
        bg[y0c:y1, x0c:x1, c] = (a[...,0]*img[ys:ys+(y1-y0c), xs:xs+(x1-x0c), c]
                                 + (1-a[...,0])*bg[y0c:y1, x0c:x1, c])
    return bg


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
        self.fx = None
        self.fy = None
        self.follow = 0.25
        self.lift = 1.3
        self.crown = 0.12
        self.face_every = 3
        self.crown_at = None
        face_opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE, num_faces=1)
        try:
            self.face = FaceLandmarker.create_from_options(face_opts)
        except Exception:
            self.face = None
        self.tiara = cv2.imread('tiara.png', cv2.IMREAD_UNCHANGED)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        soft = cv2.resize(cv2.resize(img, (w // 4, h // 4)), (w, h))
        img = cv2.addWeighted(img, 0.78, soft, 0.22, 6)

        layer = np.zeros_like(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        self.frame_count += 1

        lm = result.hand_landmarks
        hand_pos = None
        for i, hand in enumerate(lm):
            val = pinch_value(hand, self.high)
            label = result.handedness[i][0].category_name
            if label == "Left":
                label = "Right"
                self.grow = self.smooth * val + (1 - self.smooth) * self.grow
            else:
                label = "Left"
                self.bloom = self.smooth * val + (1 - self.smooth) * self.bloom
                hand_pos = (hand[9].x * w, hand[9].y * h)
            text = ("Grow" if label == "Right" else "Bloom") + " it <3"
            wx, wy = int(hand[0].x * w), int(hand[0].y * h)
            draw_pill(img, wx, wy, text)

        if hand_pos is not None:
            if self.fx is None:
                self.fx, self.fy = hand_pos
            else:
                self.fx += (hand_pos[0] - self.fx) * self.follow
                self.fy += (hand_pos[1] - self.fy) * self.follow

        if self.fx is not None:
            size = max(0.0, min(1.0, self.grow))
            lift = min(w, h) * (0.09 + 0.20 * size) * self.lift
            cx, cy = int(self.fx), int(self.fy - lift)
        else:
            cx, cy = w // 2, h // 2
        sway = math.sin(self.frame_count * 0.04) * 0.06
        draw_flower(layer, cx, cy, self.grow, self.bloom,
                    phase=sway, fc=self.frame_count)

        small = cv2.resize(layer, (w // 2, h // 2))
        glow = cv2.GaussianBlur(small, (0, 0), max(1, self.glow_amt // 2))
        glow = cv2.resize(glow, (w, h))
        img = cv2.addWeighted(img, 1.0, glow, 0.55, 0)
        b, g, r = cv2.split(layer)
        mask = cv2.max(cv2.max(b, g), r)
        alpha = cv2.multiply(mask, np.array([4.25]))
        alpha = alpha.astype(np.float32) * (1.0 / 255.0)
        img = cv2.blendLinear(img, layer, 1.0 - alpha, alpha)

        if self.tiara is not None and self.face is not None:
            if self.frame_count % self.face_every == 0:
                self.crown_at = None
                for face in self.face.detect(mp_image).face_landmarks:
                    chin = face[152]
                    forehead = face[10]
                    L, R = face[234], face[454]
                    ux = (forehead.x - chin.x) * w
                    uy = (forehead.y - chin.y) * h
                    ax = int(forehead.x * w + ux * self.crown)
                    ay = int(forehead.y * h + uy * self.crown)
                    width = math.hypot((R.x - L.x) * w, (R.y - L.y) * h) * 1.2
                    angle = math.degrees(math.atan2((R.y - L.y) * h, (R.x - L.x) * w))
                    self.crown_at = (ax, ay, int(width), angle)
            if self.crown_at is not None:
                overlay_png(img, self.tiara, *self.crown_at)

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
    .bloom-hb {
        text-align: center;
        font-family: 'Pacifico', cursive;
        font-size: 2rem;
        margin: 0.3rem 0 0.1rem 0;
        background: linear-gradient(90deg, #ff5fa2, #ffb347, #c77dff, #ff5fa2);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shine 5s linear infinite, hbpop 2.4s ease-in-out infinite;
        letter-spacing: 0.02em;
    }
    @keyframes hbpop { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.06); } }
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

    @media (max-width: 640px) {
        .block-container { padding-top: 0.6rem; }
        .bloom-title { font-size: 2.1rem; }
        .bloom-hb { font-size: 1.4rem; }
        .bloom-sub { font-size: 0.95rem; }
        .bloom-card { padding: 0.8rem 0.9rem; border-radius: 20px; }
        .chips { gap: 0.6rem; }
        .chip { flex: 1 1 100%; padding: 0.6rem 0.8rem; }
        .chip .big { font-size: 1.5rem; }
        .hint { font-size: 0.85rem; }
        .stApp iframe { border-radius: 18px; }
    }
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
    '<div class="bloom-hb">🎂 Happy Birthday 🎉</div>',
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
                hold the stem, pinch to make it <b>bloom</b>
            </div>
            <div class="chip">
                <span class="big">👑</span>
                <b>Your head</b><br>
                a tiara appears <b>all by itself</b>
            </div>
        </div>
        <div class="hint">press <b>START</b> below, allow the camera, then show both hands ✨</div>
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
    follow = st.slider("follow speed 🤲", 0.05, 0.6, 0.25)
    lift = st.slider("stem length 🌿", 0.5, 2.5, 1.3)
    crown = st.slider("tiara height 👑", -0.3, 0.8, 0.12)

def _cred(name):
    """Read a credential from the environment or Streamlit secrets."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_ice_servers():
    """ICE servers for the WebRTC connection.

    Uses Twilio's TURN relay when credentials are set (required on hosted
    deployments behind NAT, e.g. Hugging Face Spaces); otherwise falls back to a
    public STUN server, which is enough for local runs.
    """
    stun_only = [{"urls": ["stun:stun.l.google.com:19302"]}]
    account_sid = _cred("TWILIO_ACCOUNT_SID")
    auth_token = _cred("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        return stun_only
    try:
        from twilio.rest import Client
        token = Client(account_sid, auth_token).tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Couldn't fetch Twilio TURN servers; using STUN only ({e}).")
        return stun_only


ice_servers = get_ice_servers()


def _ice_urls(servers):
    urls = []
    for s in servers:
        u = s.get("urls") or s.get("url")
        urls.extend(u if isinstance(u, list) else [u])
    return urls


with st.sidebar.expander("🔧 connection debug"):
    _urls = _ice_urls(ice_servers)
    _turn = any("turn:" in (u or "") for u in _urls)
    st.write(f"SID set: **{bool(_cred('TWILIO_ACCOUNT_SID'))}**")
    st.write(f"Token set: **{bool(_cred('TWILIO_AUTH_TOKEN'))}**")
    st.write(f"TURN relay active: **{_turn}**")
    st.caption("URLs (secrets hidden):")
    st.code("\n".join(u for u in _urls if u) or "(none)")

ctx = webrtc_streamer(
    key="flower",
    video_processor_factory=FlowerProcessor,
    server_rtc_configuration={"iceServers": ice_servers},
    frontend_rtc_configuration={"iceServers": ice_servers},
    media_stream_constraints={
        "video": {
            "facingMode": "user",
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        muted=True,
        playsInline=True,
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
    ctx.video_processor.follow = follow
    ctx.video_processor.lift = lift
    ctx.video_processor.crown = crown

st.markdown(
    '<div class="bloom-foot">made with <span class="beat">💖</span> just for you</div>',
    unsafe_allow_html=True,
)
