"""prompts.py

Unified prompt utilities.

Contains two parts:
1. Real‑time commentary prompt builder (used by runner) – produces exactly two lines.
2. Lightweight video QA prompt templates (evidence‑based, uncertainty aware) that came from the other branch.

This file merges both sides of a previous merge conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


BASE_STYLE = (
    "You are a professional live NBA commentator. Requirements: "
    "1) Output MUST be concise real‑time ENGLISH. Avoid vague words like maybe / probably / seems. "
    "2) Each response has EXACTLY two lines. First line: 'Time mm:ss | Score: A-B' (home first). "
    "3) If score not confirmed yet use 'Score: unknown-unknown'. Reuse last known score if unchanged. "
    "4) Second line: 1-2 sentences focusing ONLY on the freshest action with concrete verbs: drive, kick-out pass, interior feed, post up, hook shot, layup, dunk, three-pointer, mid-range jumper, fast break, rebound, turnover, block. "
    "5) If no clear new event: briefly describe current half-court setup / spacing / tempo (do not repeat previous text verbatim). "
    "6) Do NOT hallucinate player names; use 'home team' and 'away team'. "
    "7) Keep high information density; no filler or commentary about the AI."
)


@dataclass(slots=True)
class PromptContext:
    summary: str = ""       # Rolling short summary
    last_output: str = ""   # Previous full output
    score: str = "未知-未知"  # Tracked score A-B (may be '未知-未知')


class PromptBuilder:
    def __init__(self, base_style: str = BASE_STYLE):
        self.base_style = base_style

    def build(self, vlm_description: str, ctx: PromptContext, extra_style: Optional[str] = None, current_time: Optional[str] = None) -> tuple[str, str]:
        system = self.base_style
        if extra_style:
            system += "\n附加风格: " + extra_style.strip()

        user_parts = []
        if ctx.summary:
            user_parts.append(f"[Rolling summary] {ctx.summary}")
        if ctx.last_output:
            user_parts.append(f"[Previous output] {ctx.last_output[:120]}")
        if current_time:
            user_parts.append(f"[Current time] {current_time}")
        user_parts.append(f"[Recent multi-frame visual description] {vlm_description}")
        user_parts.append(f"[Tracked score] {ctx.score}")
        user_parts.append(
            "Produce exactly two lines: First line 'Time mm:ss | Score: A-B'. Second line new action commentary (1-2 sentences) in English."
        )
        user = "\n".join(user_parts)
        return system, user


#############################
# Video QA (evidence-based) #
#############################

VIDEO_QA_SYSTEM_PROMPT = """You are a professional, objective basketball analyst reviewing game footage.

Your task is to answer questions about NBA game plays based ONLY on the visual evidence provided in the video frames.

Guidelines:
1. Evidence-based: Only describe what you can clearly see in the frames.
2. Admit uncertainty: If something is unclear or not visible, explicitly state "unclear from the footage" or similar.
3. Time reference: Always mention the approximate time range you're analyzing (e.g., "around 45–47 seconds").
4. Concise: 1–3 sentences, information‑dense.
5. No speculation: Do not infer player names or tactics not visible.
6. Key observations: possession, visible score/quarter, key actions (shot, pass, rebound, turnover, block, foul if obvious)."""

VIDEO_QA_USER_TEMPLATE = (
	"Question: {question}\n\nContext: These frames are from approximately {time_start:.1f}s to {time_end:.1f}s of the game.\n\n"
	"Answer strictly based on what is visible." )

# Backward compatibility alias (api.py expects USER_PROMPT_TEMPLATE)
USER_PROMPT_TEMPLATE = VIDEO_QA_USER_TEMPLATE

__all__ = [
	"PromptBuilder",
	"PromptContext",
	"VIDEO_QA_SYSTEM_PROMPT",
    "VIDEO_QA_USER_TEMPLATE",
    "USER_PROMPT_TEMPLATE",
]
