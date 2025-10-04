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

# Import local modules
from frame_extractor import extract_frames_at_timestamp, format_frames_for_openai
from vlm import query_vision_model
from prompts import VIDEO_QA_SYSTEM_PROMPT, VIDEO_QA_USER_TEMPLATE
from router import is_search_question
from web_search import search_perplexity
from logger import logger, setup_llmobs

# Load environment variables
load_dotenv()

# Initialize Datadog LLM Observability
setup_llmobs()

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
    logger.info("NBA AI Commentary API started", 
                video_path=VIDEO_PATH,
                openai_configured=bool(OPENAI_API_KEY),
                perplexity_configured=bool(PERPLEXITY_API_KEY))
    
    print("✓ NBA AI Commentary API started")
    print(f"✓ Video path: {VIDEO_PATH}")
    if OPENAI_API_KEY:
        print("✓ OpenAI API key configured")
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


class LiveItem(BaseModel):
    ts: float
    wall_time: float
    text: str
    score: Optional[str] = None
    audio: Optional[List[str]] = None


class LiveStatusResponse(BaseModel):
    count: int
    items: List[LiveItem]


@app.get("/")
async def root():
    """Health check endpoint."""
    logger.info("Health check requested")
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


@app.get("/tts/file")
async def tts_file(name: str):
    """Serve a single TTS audio file by filename (for <audio src>)."""
    from fastapi.responses import FileResponse
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache", "tts")
    fp = os.path.join(cache_dir, name)
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(fp)


@app.get("/live/status", response_model=LiveStatusResponse)
async def live_status(limit: int = Query(20, ge=1, le=200)):
    """Return latest live commentary entries written by runner --live-log.

    The runner appends NDJSON lines with keys: ts, wall_time, text, score, audio.
    """
    log_path = os.getenv("LIVE_LOG_PATH", os.path.join(PROJECT_ROOT, "data", "cache", "live_commentary.ndjson"))
    if not os.path.exists(log_path):
        return LiveStatusResponse(count=0, items=[])
    items: List[LiveItem] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        import json as _json
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = _json.loads(ln)
                items.append(LiveItem(**obj))
            except Exception:
                continue
        # 按 wall_time 降序
        items.sort(key=lambda x: x.wall_time, reverse=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"read live log failed: {e}")
    return LiveStatusResponse(count=len(items), items=items)


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
            logger.warning("Empty question received", timestamp=request.timestamp)
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if request.timestamp < 0:
            logger.warning("Invalid timestamp", timestamp=request.timestamp)
            raise HTTPException(status_code=400, detail="Timestamp must be non-negative")
        
        # Log incoming request
        logger.info("Question received",
                   question=request.question,
                   timestamp=request.timestamp,
                   question_length=len(request.question))
        
        # 判断问题类型：搜索 or 视频分析
        if is_search_question(request.question):
            logger.info("Routing to web search", 
                       question=request.question,
                       timestamp=request.timestamp)
            
            context = f"User is watching NBA game at {request.timestamp:.1f}s"
            result = search_perplexity(request.question, context)
            
            logger.info("Web search completed",
                       question=request.question,
                       citations_count=len(result.get("citations", [])),
                       answer_length=len(result["content"]))
            
            return AskResponse(
                answer=result["content"],
                time_range=f"{request.timestamp:.1f}",
                used_segment_summary="Retrieved from web search",
                used_script_excerpt=None,
                query_type="search",
                citations=result.get("citations", [])
            )

        # 视频分析逻辑
        logger.info("Routing to video analysis",
                   question=request.question,
                   timestamp=request.timestamp)
        
        frames = extract_frames_at_timestamp(
            video_path=VIDEO_PATH,
            timestamp=request.timestamp,
            num_frames_before=2,
            num_frames_after=2,
            frame_interval=1.0
        )
        
        if not frames:
            logger.error("Failed to extract frames",
                        timestamp=request.timestamp,
                        video_path=VIDEO_PATH)
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract frames at timestamp {request.timestamp}s"
            )
        
        time_start = frames[0][0]
        time_end = frames[-1][0]
        time_range_str = f"{time_start:.1f}-{time_end:.1f}"
        
        logger.info("Frames extracted successfully",
                   frames_count=len(frames),
                   time_range=time_range_str)
        
        image_contents = format_frames_for_openai(frames)
        user_prompt = VIDEO_QA_USER_TEMPLATE.format(
            question=request.question,
            time_start=time_start,
            time_end=time_end
        )
        
        logger.info("Calling vision model",
                   frames_count=len(frames),
                   time_range=time_range_str)
        
        answer = query_vision_model(
            system_prompt=VIDEO_QA_SYSTEM_PROMPT,
            user_text=user_prompt,
            image_contents=image_contents
        )
        
        logger.info("Video analysis completed",
                   question=request.question,
                   time_range=time_range_str,
                   answer_length=len(answer))
        
        return AskResponse(
            answer=answer,
            time_range=time_range_str,
            used_segment_summary=f"Analyzed {len(frames)} frames from {time_range_str}s",
            used_script_excerpt=None,
            query_type="video",
            citations=None
        )
    
    except FileNotFoundError as e:
        logger.error("File not found",
                    error=str(e),
                    question=request.question,
                    timestamp=request.timestamp)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error in /ask endpoint",
                    error=str(e),
                    error_type=type(e).__name__,
                    question=request.question,
                    timestamp=request.timestamp)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

