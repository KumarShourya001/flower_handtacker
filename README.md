---
title: Bloom for You
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# 🌸 Bloom for You

### 🌐 Live demo → **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/KrShourya/hand-flower)**

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/KrShourya/hand-flower)

A hand-controlled flower you hold in your own hand — it grows and blooms as you
move, and crowns you with a tiara while you do. Built with Streamlit, MediaPipe
hand + face tracking and streamlit-webrtc.

- 🌱 **Right hand** pinch & open → the flower **grows**
- 🌷 **Left hand** holds the stem; pinch & open → the flower **blooms**
- 👑 **Your head** gets a tiara, scaled and tilted to match automatically

Press **Start**, allow the camera, and show both hands ✨

The stem leans whichever way your palm is pointing and side blooms open as it
grows. Once it is wide open, petals let go and drift down the frame, and a line
of handwriting appears underneath.

### Making it yours

The words live at the top of [`app.py`](app.py) — `TO` for the name in the
birthday line, `CARD` for the note under the video, and `NOTES` for the lines
that appear over the picture at full bloom. Change those and it is your gift,
not a demo.

Open the sidebar for live tweaks: glow, smoothness and pinch sensitivity, plus
how tightly the flower follows your palm, the stem length, and the tiara height.

The drawing lives in [`bloom.py`](bloom.py) and touches nothing but OpenCV and
NumPy, so you can render the flower to a still image and fiddle with it without a
webcam anywhere in sight.

### Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

The two `.task` model files are in the repo via Git LFS, so clone with LFS
installed or they will arrive as pointer stubs.

> Note: the webcam runs through WebRTC in your browser, so allow camera access
> when prompted. On the free Space, the first load can take a little while as the
> model warms up.
>
> If the camera won't connect, set `BLOOM_DEBUG=1` to show the ICE/TURN panel in
> the sidebar. It stays hidden otherwise so the page is just the gift.
