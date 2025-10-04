"""router.py - 判断问题类型（视频分析 vs 网络搜索）"""

# 视频相关关键词
VIDEO_KEYWORDS = [
    "now", "currently", "just", "right now", "happening", "this moment",
    "score", "offense", "defense", "possession", "ball", "court",
    "shot", "pass", "rebound", "dribble", "foul",
    "see", "visible", "shown", "screen", "footage",
    "who is", "what is", "where is", "which team"
]

# 搜索相关关键词
SEARCH_KEYWORDS = [
    "player", "career", "stats", "statistics", "average", "born",
    "team history", "championship", "roster", "coach",
    "rule", "regulation", "explain", "definition",
    "season", "year", "history", "record","search",
    "why", "how does", "what are the", "tell me about"
]


def is_search_question(question: str) -> bool:
    """判断是否需要网络搜索"""
    q = question.lower()
    
    # 计算关键词匹配
    video_score = sum(1 for kw in VIDEO_KEYWORDS if kw in q)
    search_score = sum(1 for kw in SEARCH_KEYWORDS if kw in q)
    
    # 搜索关键词明显更多时走搜索
    if search_score >= 2:
        return True
    
    # 默认走视频分析
    return False