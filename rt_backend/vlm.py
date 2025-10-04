"""vlm.py

封装多帧视觉理解：输入多张帧 (numpy arrays) -> 输出一段紧凑中文语义描述。

第一阶段：
 - 不调用真正的多模态 API，先做占位：简单取若干帧做均值哈希+尺寸统计 -> 伪描述
 - 方便后续替换为 OpenAI GPT-4o / o3-mini / vision 等多帧描述接口

后续接入（参考计划）:
 - OpenAI Responses API: attachments = [image[]]
 - 控制 token 成本：最多 4~8 帧，必要时降采样
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import hashlib
import cv2  # type: ignore
import numpy as np


@dataclass(slots=True)
class VLMConfig:
	max_frames: int = 6
	resize_short: int = 224  # 未来可用于下采样


class VLMClient:
	def __init__(self, cfg: VLMConfig | None = None):
		self.cfg = cfg or VLMConfig()

	def describe(self, frames: List[np.ndarray]) -> str:
		if not frames:
			return "(无帧)"
		use = frames[-self.cfg.max_frames :]
		# 简化：对每帧做亮度/边缘特征，尝试形成粗略动作线索（纯启发式，占位用）
		sigs = []
		action_hints = []
		for f in use:
			h, w = f.shape[:2]
			mean_val = float(f.mean())
			m = hashlib.md5(f[::8, ::8, :1].tobytes()).hexdigest()[:6]
			# 边缘强度估计
			gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 80, 160)
			edge_ratio = edges.mean()/255.0
			# 中央区域亮度 (可能与篮筐/内线动作相关)
			ch0, ch1 = int(h*0.35), int(h*0.65)
			cw0, cw1 = int(w*0.35), int(w*0.65)
			center_mean = gray[ch0:ch1, cw0:cw1].mean() if ch1>ch0 and cw1>cw0 else mean_val
			# 简单启发：高边缘+中心亮度变化 -> 可能是 drive/contact; 边缘低+中心稳定 -> 设置或慢节奏
			if edge_ratio > 0.18 and center_mean > mean_val * 0.95:
				action_hints.append("drive or inside attack")
			elif edge_ratio > 0.22:
				action_hints.append("fast perimeter motion")
			elif edge_ratio < 0.10:
				action_hints.append("half-court set / spacing")
			else:
				action_hints.append("ball circulation")
			sigs.append(f"[{h}x{w} mv={mean_val:5.1f} edge={edge_ratio:0.2f} h={m}]")
		# 汇总动作提示（去重保留顺序）
		ordered = []
		for a in action_hints:
			if a not in ordered:
				ordered.append(a)
		actions = "; ".join(ordered[:3])
		return "frames=" + ", ".join(sigs) + (f" | coarse_actions: {actions}" if actions else "")


def init_openai_client(api_key: str | None = None):  # compatibility shim for api.py
	# In merged architecture, vision querying lazily initializes inside query_vision_model.
	# We keep this no-op so the FastAPI startup hook doesn't fail.
	return None

__all__ = ["VLMClient", "VLMConfig", "init_openai_client", "query_vision_model"]

# Optional real vision API helper (from merged branch) -----------------------
try:  # Lazy import; only if user wants to call real vision model externally
    import os as _os
    from openai import OpenAI as _OpenAI  # type: ignore
    _vision_client = None

    def _vision_init(api_key: str | None = None):  # pragma: no cover
        global _vision_client
        if api_key is None:
            api_key = _os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for vision model")
        _vision_client = _OpenAI(api_key=api_key)

    def query_vision_model(system_prompt: str, user_text: str, image_contents: list, model: str = "gpt-4o", max_tokens: int = 500) -> str:  # pragma: no cover
        global _vision_client
        if _vision_client is None:
            _vision_init()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}, *image_contents]},
        ]
        resp = _vision_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content
except Exception:  # pragma: no cover
    pass
