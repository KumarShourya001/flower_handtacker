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

Open the sidebar for live tweaks: glow, smoothness and pinch sensitivity, plus
how tightly the flower follows your palm, the stem length, and the tiara height.

> Note: the webcam runs through WebRTC in your browser, so allow camera access
> when prompted. On the free Space, the first load can take a little while as the
> model warms up.
