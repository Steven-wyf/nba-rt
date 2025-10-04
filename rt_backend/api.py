"""
FastAPI Backend for NBA AI Commentary
--------------------------------------
Provides POST /ask endpoint that:
1. Receives question + timestamp from frontend
2. Extracts video frames around timestamp
3. Sends frames + question to OpenAI Vision API
4. Returns structured answer
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

# Unified imports (resolved): use relative paths, include search helpers
from .frame_extractor import extract_frames_at_timestamp, format_frames_for_openai
from .vlm import init_openai_client, query_vision_model
from .prompts import VIDEO_QA_SYSTEM_PROMPT, VIDEO_QA_USER_TEMPLATE, USER_PROMPT_TEMPLATE
from .router import is_search_question
from .web_search import search_perplexity

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="NBA AI Commentary API", version="1.0.0")

# CORS middleware (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# Get project root directory (parent of rt_backend)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "nba_domo.mov")

VIDEO_PATH = os.getenv("VIDEO_PATH", DEFAULT_VIDEO_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    print("✓ NBA AI Commentary API started")
    print(f"✓ Video path: {VIDEO_PATH}")
    if OPENAI_API_KEY:
        print("✓ OpenAI API key configured")
        try:  # pragma: no cover
            init_openai_client(OPENAI_API_KEY)
        except Exception as e:
            print(f"⚠ Vision client init failed: {e}")
    else:
        print("⚠ Warning: OPENAI_API_KEY not set")

    if PERPLEXITY_API_KEY:
        print("✓ Perplexity API key configured")
    else:
        print("ℹ️  Perplexity not configured (optional)")


# Request/Response models
class AskRequest(BaseModel):
    question: str
    timestamp: float


class AskResponse(BaseModel):
    answer: str
    time_range: str
    used_segment_summary: str
    used_script_excerpt: Optional[str] = None
    query_type: Optional[str] = None  # "video" or "search"
    citations: Optional[List[str]] = None


class TTSListItem(BaseModel):
    filename: str
    mtime: float
    size: int


class TTSListResponse(BaseModel):
    count: int
    items: List[TTSListItem]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "NBA AI Commentary API",
        "version": "1.0.0"
    }


@app.get("/tts/list", response_model=TTSListResponse)
async def list_tts(limit: int = Query(10, ge=1, le=100)):
    """List recent synthesized TTS commentary audio files.

    Returns newest first up to 'limit'. Frontend can poll this to fetch latest audio.
    """
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache", "tts")
    if not os.path.isdir(cache_dir):
        return TTSListResponse(count=0, items=[])
    entries = []
    for name in os.listdir(cache_dir):
        if not name.lower().endswith((".mp3", ".wav", ".ogg")):
            continue
        fp = os.path.join(cache_dir, name)
        try:
            st = os.stat(fp)
            entries.append((st.st_mtime, name, st.st_size))
        except OSError:
            continue
    entries.sort(key=lambda x: x[0], reverse=True)
    items = [TTSListItem(filename=n, mtime=mt, size=sz) for mt, n, sz in entries[:limit]]
    return TTSListResponse(count=len(items), items=items)


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer user question about video at specified timestamp.
    
    Args:
        request: Contains question and timestamp
    
    Returns:
        Structured answer with time range and context
    """
    try:
        # Validate inputs
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if request.timestamp < 0:
            raise HTTPException(status_code=400, detail="Timestamp must be non-negative")
        
        # 判断问题类型：搜索 or 视频分析
        if is_search_question(request.question):
            context = f"User is watching NBA game at {request.timestamp:.1f}s"
            result = search_perplexity(request.question, context)
            return AskResponse(
                answer=result["content"],
                time_range=f"{request.timestamp:.1f}",
                used_segment_summary="Retrieved from web search",
                used_script_excerpt=None,
                query_type="search",
                citations=result.get("citations", [])
            )

        # 视频分析逻辑
        frames = extract_frames_at_timestamp(
            video_path=VIDEO_PATH,
            timestamp=request.timestamp,
            num_frames_before=2,
            num_frames_after=2,
            frame_interval=1.0
        )
        if not frames:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract frames at timestamp {request.timestamp}s"
            )
        time_start = frames[0][0]
        time_end = frames[-1][0]
        time_range_str = f"{time_start:.1f}-{time_end:.1f}"
        image_contents = format_frames_for_openai(frames)
        user_prompt = VIDEO_QA_USER_TEMPLATE.format(
            question=request.question,
            time_start=time_start,
            time_end=time_end
        )
        answer = query_vision_model(
            system_prompt=VIDEO_QA_SYSTEM_PROMPT,
            user_text=user_prompt,
            image_contents=image_contents
        )
        return AskResponse(
            answer=answer,
            time_range=time_range_str,
            used_segment_summary=f"Analyzed {len(frames)} frames from {time_range_str}s",
            used_script_excerpt=None,
            query_type="video",
            citations=None
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

