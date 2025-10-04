"""web_search.py - Perplexity API 调用"""

import os
import requests
from typing import Dict, List


def search_perplexity(question: str, context: str = None) -> Dict:
    """
    调用 Perplexity API 进行网络搜索
    
    Args:
        question: 用户问题
        context: 可选上下文
    
    Returns:
        {"content": 答案文本, "citations": 引用来源列表}
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not set")
    
    # 系统提示
    system_msg = (
        "You are a knowledgeable NBA assistant. "
        "Answer in 2-3 sentences maximum. "
        "Focus on factual, up-to-date information."
    )
    
    # 构建用户消息
    user_msg = f"Context: {context}\n\nQuestion: {question}" if context else question
    
    # API 请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
        "return_citations": True,
        "top_p": 0.9,
    }
    
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
        timeout=45
    )
    resp.raise_for_status()
    
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    
    # 提取引用
    citations: List[str] = []
    if "citations" in data:
        citations = data["citations"]
    elif "choices" in data and data["choices"] and "citations" in data["choices"][0]:
        citations = data["choices"][0]["citations"]
    
    return {"content": content, "citations": citations}
