import os
import json
import requests
import streamlit as st

# """
# NBA AI Commentary Front-End (Streamlit)
# --------------------------------------
# A minimal yet polished web UI that allows users to:
# 1. Watch a demo NBA video with optional AI-generated commentary.
# 2. Pause at any time and ask the AI questions about the current play.

# The backend is expected to expose a POST /ask endpoint that accepts
#     {"question": str, "timestamp": float}
# and returns the contract described in the High-Level Design.
# """

# ----------------------------  Page Config  -----------------------------
st.set_page_config(
    page_title="NBA AI Commentator",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------  Constants  -------------------------------
VIDEO_PATH = os.path.join("data", "raw", "nba_domo.mov")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
ASK_ENDPOINT = f"{BACKEND_URL}/ask"

# ----------------------------  Styling  ---------------------------------
CUSTOM_CSS = """
<style>
/* Overall page tweaks */
body {
    background: radial-gradient(circle at 25% 25%, #1c1c1c 0%, #000000 100%);
    color: #efefef;
    font-family: "Inter", sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffb400;
}

/* Buttons & inputs */
.stButton>button {
    background-color: #ffb400;
    color: #000;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #ffc940;
    color: #000;
}

/* Slider thumb */
.stSlider > div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ffb400;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------  Sidebar  ---------------------------------
st.sidebar.title("⚙️ Settings")

ai_commentary = st.sidebar.toggle("Enable AI Commentary Track", value=False)

audio_status = "on" if ai_commentary else "off"
st.sidebar.caption(f"AI commentary track is {audio_status}")

# ----------------------------  Main UI  ---------------------------------
st.title("🏀 NBA AI Commentator Demo")

# Video Player
st.subheader("Game Video")

if not os.path.exists(VIDEO_PATH):
    st.error("Demo video not found. Expected at data/raw/nba_domo.mov")
    st.stop()

# Convert video to base64 for embedding
import base64
with open(VIDEO_PATH, "rb") as video_file:
    video_bytes = video_file.read()
    video_b64 = base64.b64encode(video_bytes).decode()

# Custom HTML5 Video Player with muted audio and real-time timestamp
video_player_html = f"""
<div style="width: 100%; max-width: 800px; margin: 0 auto;">
    <video id="nbaVideo" width="100%" controls muted style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <div style="margin-top: 15px; padding: 15px; background: rgba(255,180,0,0.15); border-radius: 8px; text-align: center; border: 2px solid rgba(255,180,0,0.3);">
        <span style="color: #ffb400; font-weight: 700; font-size: 18px;">▶ Video Time: </span>
        <span id="currentTime" style="color: #fff; font-weight: 700; font-size: 22px;">0.0s</span>
        <br>
        <span style="color: #aaa; font-size: 13px; margin-top: 5px; display: inline-block;">
            👆 Use this time when asking questions below
        </span>
    </div>
</div>

<script>
    const video = document.getElementById('nbaVideo');
    const timeDisplay = document.getElementById('currentTime');
    
    video.addEventListener('timeupdate', function() {{
        const currentSeconds = video.currentTime.toFixed(1);
        timeDisplay.textContent = currentSeconds + 's';
    }});
    
    // Ensure video is muted (AI commentary will be added later)
    video.volume = 0;
    video.muted = true;
</script>
"""

st.components.v1.html(video_player_html, height=600)

# Timestamp input (manual entry based on video time shown above)
video_duration = 300  # ~5-minute demo; adjust if necessary

st.markdown("### Question Timestamp")
st.caption("⬆ Watch the **Video Time** display above, then enter that time here")
current_time = st.number_input(
    label="Enter timestamp (seconds) for your question:",
    min_value=0.0,
    max_value=float(video_duration),
    value=0.0,
    step=0.5,
    format="%0.1f",
    key="timestamp_input",
)

# Question input
st.markdown("### Ask the AI")
question = st.text_input(
    "Type your question about the current play:",
    placeholder="e.g. Which team is on offense right now?",
)

# Submit button
ask_clicked = st.button("Ask AI 🤖", disabled=len(question.strip()) == 0)

# ----------------------------  Backend Call  ----------------------------
if ask_clicked:
    payload = {"question": question, "timestamp": current_time}

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(ASK_ENDPOINT, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"Failed to reach backend: {exc}")
            st.stop()

    # Display answer
    st.success(data.get("answer", "No answer returned."))

    # Additional metadata
    with st.expander("🔎 Details from AI"):
        st.write("**Referenced Time Range:**", data.get("time_range", "N/A"))
        st.write("**Segment Summary:**", data.get("used_segment_summary", "N/A"))
        if script_excerpt := data.get("used_script_excerpt"):
            st.write("**Script Excerpt:**", script_excerpt)

# ----------------------------  Footer  ----------------------------------
st.markdown("---")
st.caption("© 2025 NBA AI Commentator Demo – Built with Streamlit")
