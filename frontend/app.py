import os
import json
import requests
import streamlit as st
import time

LIVE_STATUS_ENDPOINT = os.environ.get("LIVE_STATUS_ENDPOINT", os.environ.get("BACKEND_URL", "http://localhost:8001") + "/live/status")
TTS_FILE_ENDPOINT = os.environ.get("TTS_FILE_ENDPOINT", os.environ.get("BACKEND_URL", "http://localhost:8001") + "/tts/file")

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
auto_refresh = st.sidebar.toggle("Auto Refresh Live Feed", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (s)", 0.5, 5.0, 1.5, 0.5)

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
        query_type = data.get("query_type", "N/A")
        st.write("**Query Type:**", "🔍 Web Search" if query_type == "search" else "📹 Video Analysis")
        st.write("**Referenced Time Range:**", data.get("time_range", "N/A"))
        st.write("**Segment Summary:**", data.get("used_segment_summary", "N/A"))
        
        # 显示引用来源（Perplexity）
        if citations := data.get("citations"):
            st.write("**Sources:**")
            for i, url in enumerate(citations[:5], 1):
                st.markdown(f"{i}. [{url}]({url})")
        
        if script_excerpt := data.get("used_script_excerpt"):
            st.write("**Script Excerpt:**", script_excerpt)

# ----------------------------  Footer  ----------------------------------
st.markdown("### Live Commentary (Realtime)")
live_container = st.empty()

def fetch_live():
    try:
        r = requests.get(LIVE_STATUS_ENDPOINT, params={"limit": 20}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "items": []}

def render_live(data):
    if not data or not data.get("items"):
        live_container.info("No live commentary yet. Start the runner with --live-log.")
        return
    items = data["items"]
    md_lines = []
    for it in items:
        ts = it.get("ts")
        txt = it.get("text", "")
        score = it.get("score", "?")
        audio_files = it.get("audio") or []
        header = f"**t={ts:.1f}s | Score {score}**" if ts is not None else f"**Score {score}**"
        md_lines.append(header + "\n" + txt)
        # 简单内联音频（仅显示第一个，避免堆积）
        if ai_commentary and audio_files:
            name = audio_files[-1]
            audio_url = f"{TTS_FILE_ENDPOINT}?name={name}"
            md_lines.append(f"<audio controls {'autoplay' if ai_commentary else ''} src='{audio_url}'></audio>")
        md_lines.append("---")
    live_container.markdown("\n".join(md_lines), unsafe_allow_html=True)

if st.button("Refresh Live Commentary"):
    render_live(fetch_live())

if auto_refresh:
    # 简单轮询机制
    last_time = time.time()
    render_live(fetch_live())
    while auto_refresh and time.time() - last_time < 60:  # 60s loop safeguard (Streamlit rerun model)
        time.sleep(refresh_interval)
        render_live(fetch_live())
        # 触发 Streamlit rerun
        st.experimental_rerun()

# ----------------------------  Transcript View  ------------------
st.markdown("### Live Transcript")
with st.expander("Show Full Transcript", expanded=True):
    transcript_limit = st.slider("Max entries", 20, 300, 120, 10, key="transcript_limit")
    show_score = st.checkbox("Show score line", value=True, key="tx_show_score")
    compact = st.checkbox("Compact (only commentary line)", value=False, key="tx_compact")
    data_tx = fetch_live()
    if data_tx.get("items"):
        # items 已按 wall_time 降序 -> 反转让最旧在上，阅读更自然
        ordered = list(reversed(data_tx["items"]))[:transcript_limit]
        lines = []
        for it in ordered:
            raw_text = it.get("text", "")
            if not raw_text:
                continue
            parts = raw_text.splitlines()
            line1 = parts[0] if parts else ""
            line2 = parts[1] if len(parts) > 1 else ""
            ts = it.get("ts")
            # 提取 mm:ss 信息（已经在第一行中）
            if compact:
                core = line2 or line1
            else:
                if show_score:
                    core = f"{line1}\n{line2}" if line2 else line1
                else:
                    core = line2 or line1
            lines.append(core)
        st.text("\n\n".join(lines))
    else:
        st.info("No transcript yet. Ensure runner started with --live-log.")

st.markdown("---")
st.caption("© 2025 NBA AI Commentator Demo – Built with Streamlit")
