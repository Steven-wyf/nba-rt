"""
Video Frame Extraction Module
------------------------------
Extracts frames from video at specified timestamp ± context window.
"""

import cv2
import base64
from typing import List, Tuple
import os


def extract_frames_at_timestamp(
    video_path: str,
    timestamp: float,
    num_frames_before: int = 2,
    num_frames_after: int = 2,
    frame_interval: float = 1.0
) -> List[Tuple[float, str]]:
    """
    Extract frames around a given timestamp from video.
    
    Args:
        video_path: Path to video file
        timestamp: Target timestamp in seconds
        num_frames_before: Number of frames to extract before timestamp
        num_frames_after: Number of frames to extract after timestamp
        frame_interval: Time interval between extracted frames in seconds
    
    Returns:
        List of (timestamp, base64_image) tuples
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate target timestamps
    target_times = []
    for i in range(-num_frames_before, num_frames_after + 1):
        t = timestamp + (i * frame_interval)
        if 0 <= t <= duration:
            target_times.append(t)
    
    frames = []
    
    for target_time in target_times:
        # Seek to target time
        frame_number = int(target_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append((target_time, frame_b64))
    
    cap.release()
    return frames


def format_frames_for_openai(frames: List[Tuple[float, str]]) -> List[dict]:
    """
    Format extracted frames for OpenAI Vision API.
    
    Args:
        frames: List of (timestamp, base64_image) tuples
    
    Returns:
        List of image content dictionaries for OpenAI API
    """
    content = []
    for ts, frame_b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "high"
            }
        })
    return content

