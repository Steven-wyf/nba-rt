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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------  Constants  -------------------------------
VIDEO_PATH = os.path.join("data", "raw", "nba_domo.mov")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
ASK_ENDPOINT = f"{BACKEND_URL}/ask"

# ----------------------------  Commentary Script Data  ------------------
# Pre-recorded commentary script extracted from runner output (0-123s)
COMMENTARY_SCRIPT = [
    {"time": 1, "text": "The teams are in a half-court set, with the ball moving around the perimeter to find the best shot. Players are spreading out for spacing, looking for driving lanes."},
    #{"time": 5, "text": "The teams maintain their half-court setup, with players cutting to create space for an open shot. Ball movement continues as they search for a high-percentage look."},
    #{"time": 8, "text": "The teams continue to work in their half-court offense, establishing spacing as players cut to open areas. The ball is being swung around as they look for an opportunity to attack the defense."},
    {"time": 12, "text": "The teams maintain their half-court setup, with players moving without the ball to create passing lanes. The offense is looking to exploit defensive gaps for an open shot."},
    #{"time": 17, "text": "The teams continue in a half-court set, with sharp ball movement seeking an open look. Players are cutting and spacing effectively to create driving lanes."},
    {"time": 24, "text": "The teams maintain their half-court setup, looking for gaps in the defense while players rotate for better positioning. The tempo is steady, with each team probing for an opening."},
    #{"time": 32, "text": "The teams continue to probe the defense, swinging the ball for open looks while maintaining solid spacing. Players are positioned well, ready for a potential drive or kick-out."},
    {"time": 39, "text": "The teams maintain their half-court setup, with players shifting for open passing lanes and potential drives. The away team is looking to exploit mismatches as they circulate the ball."},
    #{"time": 49, "text": "The teams continue in a half-court setup, with players actively cutting and screening for open looks. The ball is being rotated around the arc, searching for a high-percentage shot."},
    {"time": 55, "text": "The teams maintain a half-court set, with players utilizing off-ball screens to create separation. The ball is being passed around as they look for an open shot."},
    #{"time": 65, "text": "Both teams are still in a half-court set, with players actively cutting and looking for open lanes to exploit. The away team is trying to create mismatches with their spacing."},
    {"time": 71, "text": "The teams maintain a half-court set, with players making off-ball movements and looking for openings. The ball is circulating, attempting to find a good shooting opportunity."},
    #{"time": 77, "text": "The teams continue in a half-court setup, with players carefully spacing to create passing lanes. The ball is being swung around, probing for an open shot."},
    {"time": 79, "text": "The teams maintain a disciplined half-court setup, with players using screens to create separation. The ball is being passed around, looking for an opening in the defense."},
    {"time": 89, "text": "The teams continue in a structured half-court set, with players positioning for potential drives and kick-out passes. Off-ball movement is evident as they search for open opportunities."},
    {"time": 97, "text": "The teams maintain a patient half-court offense, with players spacing out to create driving lanes. The ball continues to move around the arc searching for an open shot."},
    {"time": 103, "text": "The teams continue to work the ball around, looking to exploit defensive gaps with quick passes. Players are maintaining good spacing, readying for a potential drive or shot."},
    {"time": 108, "text": "The away team is probing the defense, utilizing ball movement to create open lanes for cuts. Players are positioning themselves for potential catch-and-shoot opportunities."},
    {"time": 113, "text": "The away team continues to work the ball around the arc, looking for an opening. Players maintain good spacing, ready to cut or shoot."},
    {"time": 122, "text": "The away team is still circulating the ball, probing for defensive weaknesses. Players maintain good spacing as they look to create a clear shot opportunity."},
]

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
auto_voice = st.sidebar.toggle("Auto Voice Commentary", value=True)
auto_refresh = st.sidebar.toggle("Auto Refresh Live Feed", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (s)", 0.5, 5.0, 1.5, 0.5)

audio_status = "on" if ai_commentary else "off"
voice_status = "on" if auto_voice else "off"
st.sidebar.caption(f"AI commentary track is {audio_status}")
st.sidebar.caption(f"Auto voice commentary is {voice_status}")

# ----------------------------  Main UI  ---------------------------------
st.title("🏀 NBA AI Commentator Demo")

if not os.path.exists(VIDEO_PATH):
    st.error("Demo video not found. Expected at data/raw/nba_domo.mov")
    st.stop()

# Convert video to base64 for embedding
import base64
with open(VIDEO_PATH, "rb") as video_file:
    video_bytes = video_file.read()
    video_b64 = base64.b64encode(video_bytes).decode()

# Prepare commentary script as JSON for JavaScript
import json
commentary_json = json.dumps(COMMENTARY_SCRIPT)

# Custom HTML5 Video Player with Live Commentary Display
video_player_html = f"""
<div style="display: flex; gap: 20px; width: 100%;">
    <!-- Left: Video Player -->
    <div style="flex: 1; min-width: 0;">
        <h3 style="color: #ffb400; margin-bottom: 10px;">Game Video</h3>
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
    
    <!-- Right: Live Commentary -->
    <div style="flex: 0 0 400px; background: rgba(28,28,28,0.9); border-radius: 8px; padding: 20px; border: 2px solid rgba(255,180,0,0.3); box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <h3 style="color: #ffb400; margin-top: 0; margin-bottom: 15px; border-bottom: 2px solid rgba(255,180,0,0.3); padding-bottom: 10px;">
            🎙️ Live Commentary
        </h3>
        <div id="commentaryDisplay" style="color: #efefef; font-size: 16px; line-height: 1.6; min-height: 400px; max-height: 500px; overflow-y: auto;">
            <p style="color: #aaa; font-style: italic;">Commentary will appear as you play the video...</p>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: rgba(255,180,0,0.1); border-radius: 4px; border-left: 4px solid #ffb400;">
            <div style="font-size: 14px; color: #aaa;">Current Time:</div>
            <div id="commentaryTime" style="font-size: 20px; color: #ffb400; font-weight: 700;">00:00</div>
        </div>
        <div style="margin-top: 10px; padding: 8px; background: rgba(255,180,0,0.08); border-radius: 4px; text-align: center;">
            <span id="voiceStatus" style="font-size: 12px; color: #ffb400;">
                🔊 Voice: {'ON' if auto_voice else 'OFF'}
            </span>
        </div>
    </div>
</div>

<script>
    const video = document.getElementById('nbaVideo');
    const timeDisplay = document.getElementById('currentTime');
    const commentaryDisplay = document.getElementById('commentaryDisplay');
    const commentaryTime = document.getElementById('commentaryTime');
    
    // Commentary script data
    const commentaryScript = {commentary_json};
    const autoVoiceEnabled = {'true' if auto_voice else 'false'};
    
    let currentCommentaryIndex = -1;
    
    // Text-to-Speech function
    function speakCommentary(text) {{
        if (!autoVoiceEnabled || !window.speechSynthesis) {{
            return;
        }}
        
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        // Create new speech utterance
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;  // Normal speed
        utterance.pitch = 1.0; // Normal pitch
        utterance.volume = 1.0; // Full volume
        
        // Use English voice
        const voices = window.speechSynthesis.getVoices();
        const enVoice = voices.find(voice => voice.lang.startsWith('en'));
        if (enVoice) {{
            utterance.voice = enVoice;
        }}
        
        // Speak
        window.speechSynthesis.speak(utterance);
    }}
    
    // Format seconds to MM:SS
    function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins.toString().padStart(2, '0') + ':' + secs.toString().padStart(2, '0');
    }}
    
    // Update commentary based on current video time
    function updateCommentary(currentTime) {{
        // Find the most recent commentary for this timestamp
        let newIndex = -1;
        for (let i = commentaryScript.length - 1; i >= 0; i--) {{
            if (currentTime >= commentaryScript[i].time) {{
                newIndex = i;
                break;
            }}
        }}
        
        // Only update if commentary changed
        if (newIndex !== currentCommentaryIndex && newIndex >= 0) {{
            currentCommentaryIndex = newIndex;
            const commentary = commentaryScript[newIndex];
            
            // Speak the commentary text
            speakCommentary(commentary.text);
            
            // Animate the update
            commentaryDisplay.style.opacity = '0.5';
            setTimeout(() => {{
                commentaryDisplay.innerHTML = `
                    <div style="padding: 15px; background: rgba(255,180,0,0.05); border-radius: 6px; border-left: 4px solid #ffb400;">
                        <div style="color: #ffb400; font-weight: 600; margin-bottom: 10px; font-size: 14px;">
                            ⏱️ ${{formatTime(commentary.time)}}
                        </div>
                        <div style="font-size: 16px; line-height: 1.8;">
                            ${{commentary.text}}
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">
                        <div style="font-size: 13px; color: #888; margin-bottom: 8px;">Recent Commentary History:</div>
                `;
                
                // Show previous commentaries
                for (let i = Math.max(0, newIndex - 2); i < newIndex; i++) {{
                    const prev = commentaryScript[i];
                    commentaryDisplay.innerHTML += `
                        <div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.03); border-radius: 4px; font-size: 13px; color: #999;">
                            <span style="color: #ffb400; font-weight: 600;">⏱️ ${{formatTime(prev.time)}}</span><br>
                            <span style="color: #bbb;">${{prev.text.substring(0, 80)}}...</span>
                        </div>
                    `;
                }}
                
                commentaryDisplay.innerHTML += '</div>';
                commentaryDisplay.style.opacity = '1';
                
                // Scroll to top to show current commentary
                commentaryDisplay.scrollTop = 0;
            }}, 150);
        }} else if (newIndex === -1) {{
            if (currentCommentaryIndex !== -1) {{
                currentCommentaryIndex = -1;
                commentaryDisplay.innerHTML = '<p style="color: #aaa; font-style: italic;">Commentary will appear as you play the video...</p>';
            }}
        }}
    }}
    
    // Update on video timeupdate
    video.addEventListener('timeupdate', function() {{
        const currentSeconds = video.currentTime;
        timeDisplay.textContent = currentSeconds.toFixed(1) + 's';
        commentaryTime.textContent = formatTime(currentSeconds);
        updateCommentary(currentSeconds);
    }});
    
    // Stop voice when video is paused
    video.addEventListener('pause', function() {{
        if (window.speechSynthesis) {{
            window.speechSynthesis.cancel();
        }}
    }});
    
    // Ensure video is muted
    video.volume = 0;
    video.muted = true;
    
    // Load voices (some browsers need this)
    if (window.speechSynthesis) {{
        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = function() {{
            window.speechSynthesis.getVoices();
        }};
    }}
    
    // Initial commentary check
    updateCommentary(0);
</script>
"""

st.components.v1.html(video_player_html, height=700)

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
