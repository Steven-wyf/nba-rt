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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from frame_extractor import extract_frames_at_timestamp, format_frames_for_openai
from vlm import query_vision_model
from prompts import VIDEO_QA_SYSTEM_PROMPT, VIDEO_QA_USER_TEMPLATE

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

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    print(f"✓ NBA AI Commentary API started")
    print(f"✓ Video path: {VIDEO_PATH}")
    if OPENAI_API_KEY:
        print(f"✓ OpenAI API key configured")
    else:
        print(f"⚠ Warning: OPENAI_API_KEY not set")


# Request/Response models
class AskRequest(BaseModel):
    question: str
    timestamp: float


class AskResponse(BaseModel):
    answer: str
    time_range: str
    used_segment_summary: str
    used_script_excerpt: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "NBA AI Commentary API",
        "version": "1.0.0"
    }


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
        
        # Extract frames around timestamp
        frames = extract_frames_at_timestamp(
            video_path=VIDEO_PATH,
            timestamp=request.timestamp,
            num_frames_before=2,
            num_frames_after=2,
            frame_interval=1.0  # 1 second intervals
        )
        
        if not frames:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract frames at timestamp {request.timestamp}s"
            )
        
        # Calculate time range
        time_start = frames[0][0]
        time_end = frames[-1][0]
        time_range_str = f"{time_start:.1f}-{time_end:.1f}"
        
        # Format frames for OpenAI
        image_contents = format_frames_for_openai(frames)
        
        # Construct user prompt
        user_prompt = VIDEO_QA_USER_TEMPLATE.format(
            question=request.question,
            time_start=time_start,
            time_end=time_end
        )
        
        # Query OpenAI Vision model
        answer = query_vision_model(
            system_prompt=VIDEO_QA_SYSTEM_PROMPT,
            user_text=user_prompt,
            image_contents=image_contents
        )
        
        # Construct response
        return AskResponse(
            answer=answer,
            time_range=time_range_str,
            used_segment_summary=f"Analyzed {len(frames)} frames from {time_range_str}s",
            used_script_excerpt=None  # Will be populated when script generation is implemented
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

