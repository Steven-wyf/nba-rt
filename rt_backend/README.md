# NBA AI Commentary Backend

## Quick Start

### 1. Setup Environment

```bash
# Create .env file in project root
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend Server

```bash
cd rt_backend
python api.py
```

Or with uvicorn directly:
```bash
uvicorn rt_backend.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which team is on offense?", "timestamp": 45.0}'
```

## Architecture

```
api.py              → FastAPI server, /ask endpoint
frame_extractor.py  → Extract video frames at timestamp ± context
vlm.py              → OpenAI Vision API wrapper
prompts.py          → System prompts for GPT-4V
```

## API Endpoints

### POST /ask

**Request:**
```json
{
  "question": "What is happening right now?",
  "timestamp": 45.5
}
```

**Response:**
```json
{
  "answer": "The home team is pushing the ball up court...",
  "time_range": "43.0-47.0",
  "used_segment_summary": "Analyzed 5 frames from 43.0-47.0s",
  "used_script_excerpt": null
}
```

## Configuration

Environment variables (in `.env`):
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `VIDEO_PATH`: Path to video file (default: `data/raw/nba_domo.mov`)
- `BACKEND_URL`: Backend URL for frontend (default: `http://localhost:8000`)

