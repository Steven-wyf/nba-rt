"""tempo.py

负责实时帧 -> 文本解说 的节奏控制：
1. 聚合一定时间窗口的帧 (多帧上下文)
2. 控制调用大模型的最小间隔，避免频繁抖动
3. 基于最近输出结果做“去抖”：若视觉语义变化很小，则延迟输出

核心概念：
 - FrameEvent: 传入的帧对象（引用 runner.Frame）
 - Window: 保留最近 N 秒或 N 帧
 - Tick: 外部每来一帧调用 push(frame)，内部判断是否需要触发一次 VLM 推理

当前实现策略（简化版）：
 - 配置 target_interval_s: 两次调用最短间隔
 - 配置 max_window_frames: VLM 上下文使用的最大帧数（最新的几帧）
 - push 时将帧加入队列，若距离上次触发 >= target_interval_s 则返回一组帧用于推理
 - 后续可以扩展：场景变化检测 / 事件检测 / 时间加权等
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Any
import time

# 仅类型提示导入（运行时不强依赖）
try:  # pragma: no cover
	from .runner import Frame  # type: ignore
except Exception:  # noqa: PIE786
	Frame = Any  # 回退为 Any，避免类型检查报错


@dataclass(slots=True)
class WindowConfig:
	target_interval_s: float = 2.0          # 两次模型调用最小时间间隔
	max_window_frames: int = 8              # 模型输入的最大帧数
	min_frames_to_fire: int = 2             # 触发推理最少帧


class TempoController:
	def __init__(self, cfg: WindowConfig):
		self.cfg = cfg
		self.frames: Deque[Frame] = deque()  # type: ignore[valid-type]
		self._last_fire_wall: float = 0.0

	def push(self, frame: Frame) -> Optional[List[Frame]]:  # type: ignore[valid-type]
		"""推入一帧；若到达触发条件，返回给上层用于推理的帧列表。"""
		self.frames.append(frame)
		# 仅保留最新 max_window_frames
		while len(self.frames) > self.cfg.max_window_frames:
			self.frames.popleft()

		now = time.time()
		if (now - self._last_fire_wall) < self.cfg.target_interval_s:
			return None
		if len(self.frames) < self.cfg.min_frames_to_fire:
			return None

		self._last_fire_wall = now
		return list(self.frames)


__all__ = ["TempoController", "WindowConfig"]

