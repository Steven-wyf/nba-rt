# 轻量VLM调用（多帧合一）

import os
from typing import List, Dict, Any
from openai import OpenAI

client = None


def init_openai_client(api_key: str = None):
    """Initialize OpenAI client with API key."""
    global client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)


def query_vision_model(
    system_prompt: str,
    user_text: str,
    image_contents: List[Dict[str, Any]],
    model: str = "gpt-4o",
    max_tokens: int = 500
) -> str:
    """
    Query OpenAI Vision model with multiple frames.
    
    Args:
        system_prompt: System message for model behavior
        user_text: User question/prompt text
        image_contents: List of image content dicts (formatted for OpenAI)
        model: Model name (default: gpt-4o for vision)
        max_tokens: Maximum response tokens
    
    Returns:
        Model response text
    """
    global client
    if client is None:
        init_openai_client()
    
    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                *image_contents
            ]
        }
    ]
    
    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3  # Lower temperature for factual analysis
    )
    
    return response.choices[0].message.content
