"""runner.py

主进程入口（第一阶段：仅实现本地 mp4 抽帧管线）。

后续会在此文件逐步串联：
1. 抓帧 -> 形成多帧窗口 (tempo.py 未来接管节奏)
2. VLM 视觉理解 (vlm.py)
3. 语义增量生成 & Prompt 控制 (prompts.py)
4. TTS 低延迟合成 (tts.py)
5. 前端流式输出 / 播放

当前版本只实现：
 - 读取 data/raw/nba_domo.mov （模拟直播，可换 mp4/mov）
 - 按目标输出帧率（采样）抽取帧
 - 生成统一的帧数据结构（含时间戳 / numpy / 可选 JPEG 压缩）
 - 可选落盘到 data/cache/frames 便于调试

设计要点：
 - “模拟直播” => 串行逐帧 + sleep(实时节奏可选)；默认不 sleep，方便开发
 - 抽帧策略：如果源 fps 很高，只按 target_fps 采样；若 target_fps >= 源 fps 则不跳帧
 - 为后续多模块协作，提供一个 Frame dataclass，后续可以扩展：检测结果 / OCR / 事件标签
 - 代码需在无 OpenCV 时给出友好错误提示
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Iterable, Optional
import threading
from collections import deque
import re

try:
	import pytesseract  # type: ignore
except Exception:  # pragma: no cover
	pytesseract = None  # type: ignore

from .tempo import TempoController, WindowConfig
from .vlm import VLMClient, VLMConfig
from .prompts import PromptBuilder, PromptContext
from .tts import TTSClient, TTSConfig

try:  # Optional openai import; user may not configure yet
	import openai  # type: ignore
except Exception:  # pragma: no cover
	openai = None  # type: ignore

try:
	import cv2  # type: ignore
	import numpy as np
except ImportError as e:  # pragma: no cover
	print("[runner] 需要依赖 opencv-python 与 numpy，请先在 requirements.txt 中加入并安装。", file=sys.stderr)
	raise


# ------------------------------ Data Structures ------------------------------ #

@dataclasses.dataclass(slots=True)
class Frame:
	index: int                # 采样后序号（非原始序号）
	orig_index: int           # 视频中真实帧序号
	timestamp: float          # 秒
	image: "np.ndarray"       # H x W x C (BGR)
	jpeg: Optional[bytes] = None  # 可选的 JPEG 压缩缓存（延迟生成）

	def to_jpeg(self, quality: int = 85) -> bytes:
		if self.jpeg is None:
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
			ok, buf = cv2.imencode('.jpg', self.image, encode_param)
			if not ok:
				raise RuntimeError("JPEG 编码失败")
			self.jpeg = buf.tobytes()
		return self.jpeg


# ------------------------------ Core Extraction ------------------------------ #

def ensure_decodable(video_path: Path) -> Path:
	"""检测视频是否可被 OpenCV 解码；若失败且存在 ffmpeg，则转码为临时 mp4 返回路径。

	转码策略：
	  - 视频流: libx264 veryfast crf 28 (低码率快速预览)
	  - 去音频 (-an) 减少延迟
	"""
	test_cap = cv2.VideoCapture(str(video_path))
	ok = test_cap.isOpened()
	if ok:
		# 试读一帧
		ret, _ = test_cap.read()
		ok = ret
	test_cap.release()
	if ok:
		return video_path
	# 不可解码 -> 转码
	if shutil.which("ffmpeg") is None:
		print(f"[runner] 视频无法直接解码且未检测到 ffmpeg: {video_path}", file=sys.stderr)
		return video_path  # 继续走原逻辑，后面会抛错
	cache_dir = Path("data/cache/transcode")
	cache_dir.mkdir(parents=True, exist_ok=True)
	out_path = cache_dir / f"{video_path.stem}_proxy.mp4"
	cmd = [
		"ffmpeg", "-y", "-i", str(video_path),
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
		"-pix_fmt", "yuv420p", "-an", str(out_path)
	]
	try:
		subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		print(f"[runner] 已自动转码为代理文件: {out_path}")
		return out_path
	except Exception as e:
		print(f"[runner] 转码失败，继续使用原文件: {e}", file=sys.stderr)
		return video_path


def extract_frames(
	video_path: str | os.PathLike,
	target_fps: float = 1.0,
	max_frames: Optional[int] = None,
	dump_dir: Optional[str | os.PathLike] = None,
	dump_jpeg_quality: int = 85,
	realtime: bool = False,
	start_time_offset: float = 0.0,
) -> Generator[Frame, None, None]:
	"""按 target_fps 采样抽取帧。

	参数:
		video_path: mp4 路径
		target_fps: 目标采样 FPS（<=0 视为使用源 FPS）
		max_frames: 限制产出帧数（调试）
		dump_dir: 若提供，则将采样帧保存为 jpg
		dump_jpeg_quality: 保存质量
		realtime: 若 True，按时间戳 sleep 模拟直播节奏
		start_time_offset: 从某个秒数偏移开始（跳过开头）
	"""
	path = Path(video_path)
	if not path.exists():
		raise FileNotFoundError(f"视频不存在: {path}")

	# MOV 兼容：若无法直接打开尝试转码代理
	path = ensure_decodable(path)
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise RuntimeError(f"无法打开视频: {path}")

	src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	if src_fps <= 0:
		# 回退：尝试基于时间估算 or 直接不采样
		print("[runner] 源 FPS 读取失败，默认使用 30", file=sys.stderr)
		src_fps = 30.0

	if target_fps <= 0 or target_fps >= src_fps:
		frame_interval = 1  # 不跳帧
		eff_fps = src_fps
	else:
		frame_interval = max(int(round(src_fps / target_fps)), 1)
		eff_fps = src_fps / frame_interval

	if start_time_offset > 0:
		# 计算需要跳过的帧数
		skip = int(start_time_offset * src_fps)
		cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

	dump_path: Optional[Path] = None
	if dump_dir:
		dump_path = Path(dump_dir)
		dump_path.mkdir(parents=True, exist_ok=True)

	print(
		f"[runner] 开始抽帧: src_fps={src_fps:.3f}, target_fps={target_fps}, eff_fps={eff_fps:.3f}, interval={frame_interval}, total_frames={total_frames}"
	)

	origin_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
	sampled_index = 0
	t0_wall = time.time()
	first_timestamp: Optional[float] = None

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		origin_index += 1

		# 只保留 interval 上的帧
		if (origin_index - 1) % frame_interval != 0:
			continue

		timestamp = (origin_index - 1) / src_fps
		if first_timestamp is None:
			first_timestamp = timestamp

		# 模拟直播节奏
		if realtime and first_timestamp is not None:
			elapsed_video = timestamp - first_timestamp
			elapsed_wall = time.time() - t0_wall
			if elapsed_video > elapsed_wall:
				time.sleep(elapsed_video - elapsed_wall)

		fr = Frame(
			index=sampled_index,
			orig_index=origin_index - 1,
			timestamp=timestamp,
			image=frame,
		)

		if dump_path is not None:
			# 延迟编码 -> 直接用 cv2.imwrite 以避免额外内存拷贝
			out_file = dump_path / f"frame_{sampled_index:05d}.jpg"
			cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), dump_jpeg_quality])

		yield fr

		sampled_index += 1
		if max_frames is not None and sampled_index >= max_frames:
			break

	cap.release()
	print(f"[runner] 抽帧结束: 输出 {sampled_index} 帧")


# ------------------------------ CLI / Demo ------------------------------ #

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser("nba-rt runner (frame extractor phase)")
	p.add_argument("--video", default="data/raw/nba_domo.mov", help="视频路径 (mp4/mov 均可)")
	p.add_argument("--fps", type=float, default=1.0, help="目标采样 FPS (<=0 使用源 FPS)")
	p.add_argument("--max-frames", type=int, default=10, help="限制输出帧数，默认10；传负数表示不限制")
	p.add_argument("--duration", type=float, default=0.0, help="解说持续秒数 (0 表示不限制)")
	p.add_argument("--model", default="gpt-4o-mini", help="OpenAI 模型名称 (默认 gpt-4o-mini)")
	p.add_argument("--dump", action="store_true", help="是否将采样帧保存到 data/cache/frames")
	p.add_argument("--realtime", action="store_true", help="是否按真实时间节奏 sleep")
	p.add_argument("--start-offset", type=float, default=0.0, help="起始秒偏移")
	p.add_argument("--show", action="store_true", help="调试：窗口显示帧 (需要 GUI 环境)")
	p.add_argument("--debug-openai", action="store_true", help="调试 OpenAI: 打印可能的非 ASCII 环境变量并尝试清理")
	p.add_argument("--probe-models", type=str, default="", help="逗号分隔模型名列表，只做可用性探测后退出 (例: gpt-5,gpt-5-mini,gpt-4o-mini)")
	p.add_argument("--interval", type=float, default=2.0, help="解说触发的时间窗口秒 (默认 2.0，可调小提升密度，如 1.0 / 0.75)")
	p.add_argument("--window-frames", type=int, default=8, help="窗口内最大帧数 (默认8) 用于描述聚合，调大更全面，调小更快")
	p.add_argument("--stream", action="store_true", help="启用流式输出（边生成边打印第二行内容）")
	p.add_argument("--adaptive", action="store_true", help="启用自适应：根据模型平均延迟动态调整 interval")
	p.add_argument("--min-interval", type=float, default=0.6, help="自适应/手动允许的最小 interval 下限 (默认0.6)")
	p.add_argument("--max-interval", type=float, default=2.5, help="自适应允许的最大 interval 上限 (默认2.5)")
	p.add_argument("--event-trigger", action="store_true", help="启用事件触发：边缘变化大时即刻提前解说")
	p.add_argument("--event-gap", type=float, default=0.6, help="事件触发最小间隔秒 (默认0.6)")
	p.add_argument("--event-threshold", type=float, default=0.08, help="事件触发边缘变化阈值 (默认0.08)")
	p.add_argument("--strict-model", action="store_true", help="严格使用指定模型，404 不自动回退")
	p.add_argument("--tts", action="store_true", help="开启 TTS 朗读第二行解说 (需要 OPENAI_API_KEY 支持语音模型)")
	p.add_argument("--no-tts-autoplay", action="store_true", help="仅生成音频文件不本地播放")
	p.add_argument("--tts-early-chars", type=int, default=0, help="流式模式下第二行累计达到该字符数立即启动早期 TTS (0 关闭)")
	p.add_argument("--tts-debug", action="store_true", help="打印 TTS 播放相关调试信息 (检测 /dev/snd 等)")
	p.add_argument("--live-log", type=str, default="", help="将每次 commentary 追加写入 NDJSON 文件 (用于前端轮询)")
	return p.parse_args(list(argv) if argv is not None else None)


def _openai_commentary(system: str, user: str, model: str = "gpt-4o-mini", *, debug: bool = False, stream: bool = False, strict: bool = False, on_delta=None) -> str:
	"""仅使用新版 OpenAI SDK 接口 (>=1.0) 进行调用；不回退旧 ChatCompletion。

	策略:
	- 若缺少 openai 包 或 未设置 OPENAI_API_KEY -> 抛出 RuntimeError
	- 仅调用 OpenAI().chat.completions.create
	- 失败直接抛出 RuntimeError（包含诊断 & UTF-8 提示），由上层决定是否终止
	"""
	if openai is None:
		raise RuntimeError("openai 未安装，需 pip install openai >=1.0")
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("缺少环境变量 OPENAI_API_KEY")

	# 始终进行非 ASCII 头部安全清理（避免 ascii 编码异常）
	def _non_ascii(s: str) -> bool:
		return any(ord(c) > 127 for c in s)
	suspects: dict[str, str] = {}
	for k, v in list(os.environ.items()):
		if k.startswith("OPENAI") and isinstance(v, str) and _non_ascii(v):
			if k != "OPENAI_API_KEY":  # 保留真正的 key
				suspects[k] = v
	if suspects:
		print("[openai-sanitize] 移除含非 ASCII 的环境变量以避免 httpx header 错误:")
		for k, v in suspects.items():
			preview = v[:40] + ("..." if len(v) > 40 else "")
			print(f"  {k} = {preview}")
		for k in suspects.keys():
			os.environ.pop(k, None)

	# 确保 UTF-8 相关环境，减少编码问题
	for _k, _v in ("PYTHONIOENCODING", "utf-8"), ("LANG", "C.UTF-8"), ("LC_ALL", "C.UTF-8"):
		os.environ.setdefault(_k, _v)

	# 简单别名映射（仅限 gpt-4-mini -> gpt-4o-mini）。用户之前要求 gpt-5 不做映射，这里保持范围最小。
	alias_map = {
		"gpt-4-mini": "gpt-4o-mini",
		"gpt4-mini": "gpt-4o-mini",
		"gpt4o-mini": "gpt-4o-mini",
	}
	orig_model = model
	if not strict and model in alias_map:
		print(f"[model-alias] {model} -> {alias_map[model]}")
		model = alias_map[model]

	def _invoke(target_model: str) -> str:
		from openai import OpenAI  # type: ignore
		client = OpenAI(api_key=api_key)
		# 说明：
		# gpt-5 系列：不支持自定义 temperature（只能默认），并要求使用 max_completion_tokens。
		# 其它（如 gpt-4o / 4o-mini）：使用 max_tokens，允许 temperature。
		# 我们做自适应：
		#   1. 构造一组候选 token 参数字段顺序
		#   2. 首次请求不带不兼容参数（gpt-5* 不带 temperature）
		#   3. 若 400 且包含 unsupported parameter/value，通过解析报错移除该参数重试

		is_gpt5 = "gpt-5" in target_model
		primary_order = ["max_completion_tokens", "max_tokens"] if is_gpt5 else ["max_tokens", "max_completion_tokens"]
		base_messages = [
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		]

		# 初始可用参数集合
		removable_params = {"temperature", "max_tokens", "max_completion_tokens"}
		attempt_errors: list[str] = []

		for param_name in primary_order:
			# 构造 kwargs
			kwargs: dict = {
				"model": target_model,
				"messages": base_messages,
			}
			if not is_gpt5:
				kwargs["temperature"] = 0.7  # 仅非 gpt-5 系列尝试设置
			kwargs[param_name] = 120

			# 最多做多轮移除重试（例如先去掉 temperature，再换 token 字段）
			for _ in range(3):
				try:
					if debug:
						print(f"[openai-debug] 尝试参数: {kwargs}")
					if stream:
						# 流式模式：逐 delta 打印，仅组装最终文本返回
						accum = []
						resp_iter = client.chat.completions.create(stream=True, **kwargs)  # type: ignore[arg-type]
						for event in resp_iter:  # type: ignore
							try:
								choice = event.choices[0]
								delta = getattr(choice.delta, "content", None)
								if delta:
									accum.append(delta)
									print(delta, end="", flush=True)
									if on_delta is not None:
										try:
											on_delta(delta)
										except Exception as _cb_err:
											if debug:
												print(f"[stream-callback-error] {_cb_err}")
							except Exception:
								continue
						# 结束换行
						print()
						content_full = "".join(accum).strip()
						if not content_full:
							return "(空响应 stream)"
						return content_full
					else:
						resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
					if debug:
						# 打印原始响应（尽量安全截断）
						try:
							print(f"[openai-raw] usage={getattr(resp,'usage',None)}")
							print(f"[openai-raw] choices={len(resp.choices)} model={getattr(resp,'model',None)} id={getattr(resp,'id',None)}")
						except Exception as _dbg:
							print(f"[openai-debug] 无法打印 usage/choices: {_dbg}")
					choice = resp.choices[0]
					content = getattr(choice.message, "content", None)
					if content is None or not str(content).strip():
						# 尝试挖掘其它字段
						raw_choice = repr(choice)
						if debug:
							print(f"[openai-debug] 空内容，raw choice: {raw_choice[:400]}{'...' if len(raw_choice)>400 else ''}")
						return f"(空响应 raw_choice_snippet={raw_choice[:120]})"
					return str(content).strip()
				except Exception as e_inner:  # pragma: no cover
					err_text = str(e_inner)
					if debug:
						print(f"[openai-error] {err_text}")
					attempt_errors.append(err_text)
					low = err_text.lower()
					# 检测明确 unsupported 提示
					unsupported_param = None
					if "unsupported" in low:
						# 简单基于关键字匹配
						for rp in list(removable_params):
							if rp in low:
								unsupported_param = rp
								break
					if unsupported_param is not None and unsupported_param in kwargs:
						# 移除并下一轮重试
						if debug:
							print(f"[openai-debug] 移除不被支持参数: {unsupported_param}")
						kwargs.pop(unsupported_param, None)
						removable_params.discard(unsupported_param)
						continue
					# 如果是当前 token 字段不支持且还有下一个 param_name，会跳出外层循环尝试下一个 token 字段
					if param_name in low and "unsupported" in low:
						break  # 跳出内层 -> 尝试下一个 token 字段
					# 其它错误直接抛出（非可自动修复类型）
					raise
		# 所有候选字段失败
		joined = " | ".join(attempt_errors)
		raise RuntimeError(f"模型调用失败(参数自适应后仍不支持): {joined}")

	# 第一次尝试
	try:
		return _invoke(model)
	except Exception as e:
		msg = str(e)
		# 如果是 model_not_found 且非 strict 且存在已知 alias 回退链
		if ("model_not_found" in msg.lower() or "does not exist" in msg.lower()) and not strict:
			fallback_chain = []
			if orig_model == "gpt-4-mini":
				fallback_chain = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
			for fb in fallback_chain:
				if fb == model:
					continue
				print(f"[model-fallback] {orig_model} -> {fb}")
				try:
					return _invoke(fb)
				except Exception as e2:  # 继续下一候选
					print(f"[model-fallback-fail] {fb}: {e2}")
		# 普通错误路径
		if "ascii" in msg.lower() and "codec" in msg.lower():
			msg += " | 编码提示: export PYTHONIOENCODING=utf-8; export LANG=C.UTF-8; export LC_ALL=C.UTF-8"
		raise RuntimeError(f"OpenAI 新接口调用失败: {msg}") from e


def _probe_models(models: list[str], debug: bool = False) -> None:
	"""探测多个模型是否可用：逐个发送最小 prompt。

	策略:
	- 对每个模型构造极简 user 消息
	- 使用与主逻辑相同的自适应参数调整
	- 输出: [OK] / [FAIL] + 失败原因（截断）
	"""
	print(f"[probe] 开始探测 {len(models)} 个模型: {models}")
	for m in models:
		try:
			resp = _openai_commentary(
				system="你是一个探测助手。",
				user="简短回答: ok",
				model=m,
				debug=debug,
			)
			print(f"[probe][OK] {m}: {resp[:60]}{'...' if len(resp)>60 else ''}")
		except Exception as e:
			msg = str(e)
			print(f"[probe][FAIL] {m}: {msg[:160]}{'...' if len(msg)>160 else ''}")
	print("[probe] 完成")


def run_realtime_commentary(args: argparse.Namespace) -> None:
	dump_dir = "data/cache/frames" if args.dump else None
	# clamp 初始 interval 在 [min_interval, max_interval]
	init_interval = max(min(args.interval, args.max_interval), args.min_interval)
	if init_interval != args.interval:
		print(f"[interval] 调整初始 interval {args.interval} -> {init_interval}")
	args.interval = init_interval
	tempo = TempoController(WindowConfig(target_interval_s=args.interval, max_window_frames=args.window_frames))
	vlm = VLMClient(VLMConfig(max_frames=6))
	tts_client: Optional[TTSClient] = None
	# 收集当前 commentary 期间生成的音频文件路径
	audio_batch: list[str] = []
	if args.tts:
		cfg = TTSConfig()
		if args.no_tts_autoplay:
			cfg.autoplay = False
		def _on_audio(path: Path):  # noqa: ANN001
			# 仅记录文件名，前端通过 /tts/file?name= 访问
			try:
				audio_batch.append(Path(path).name)
			except Exception:
				pass
		tts_client = TTSClient(cfg, on_audio=_on_audio)
	prompt_builder = PromptBuilder()
	ctx = PromptContext(summary="", last_output="")

	lock = threading.Lock()
	recent_frames = deque(maxlen=args.window_frames)
	last_fire_wall = 0.0  # 记录最近一次任何形式（interval 或事件）的解说时间
	latency_mavg = 0.0
	alpha = 0.3  # 平滑系数

	def edge_ratio_fast(img) -> float:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 80, 160)
		return float(edges.mean()/255.0)

	# ------------------ 简单比分 OCR ------------------ #
	# 假设比分位于左上角区域：截取相对宽高的一个小矩形并进行 OCR。
	# 后续可以根据实际素材调整区域或做模板匹配 / 颜色过滤。

	def ocr_score(image) -> Optional[str]:  # image: np.ndarray (BGR)
		if pytesseract is None:
			return None
		h, w = image.shape[:2]
		# 取左上角 20% 宽 * 15% 高区域
		crop = image[0:int(0.15*h), 0:int(0.20*w)].copy()
		# 预处理：转灰度 -> 二值 -> 放大
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		# 自适应阈值
		bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
							 cv2.THRESH_BINARY_INV, 25, 15)
		# 反转确保黑字白底（多数 OCR 对深色字白底较稳定）
		inv = 255 - bw
		# 放大
		inv = cv2.resize(inv, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
		config = "--psm 6 -c tessedit_char_whitelist=0123456789:-"  # 行文本模式
		try:
			text = pytesseract.image_to_string(inv, config=config)  # type: ignore
		except Exception:
			return None
		text = text.strip().replace("\n", " ")
		if not text:
			return None
		# 匹配比分模式 常见形式 like 89-76 或 89:76
		m = re.search(r"(\d{1,3})\s*[-:]\s*(\d{1,3})", text)
		if not m:
			return None
		return f"{m.group(1)}-{m.group(2)}"

	def format_time(ts: float) -> str:
		m = int(ts // 60)
		s = int(ts % 60)
		return f"{m:02d}:{s:02d}"

	def process_frames(frames_batch: list[Frame]):
		nonlocal ctx
		desc = vlm.describe([f.image for f in frames_batch])
		current_ts = frames_batch[-1].timestamp
		time_str = format_time(current_ts)
		# OCR 尝试更新比分（仅使用最新一帧）
		ocr_sc = ocr_score(frames_batch[-1].image)
		if ocr_sc:
			with lock:
				ctx.score = ocr_sc
		system, user = prompt_builder.build(desc, ctx, current_time=time_str)
		# 统计调用延迟
		start_call = time.time()
		stream_accum_second = []
		def _maybe_early_tts():
			if not (args.stream and args.tts and args.tts_early_chars > 0 and tts_client is not None):
				return
			cur = "".join(stream_accum_second)
			if len(cur) >= args.tts_early_chars:
				# 只触发一次：清空标记避免重复
				# 为避免与最终完整重复，这里不分片，只要长度满足就直接合成当前片段
				tts_client.enqueue(cur.strip())
				# 设置为极大值防止二次进入
				args.tts_early_chars = 10**9

		if args.stream and args.tts and args.tts_early_chars > 0:
			# 临时包装 _openai_commentary 的流式打印：我们无法直接注入内部，所以暂时不改内部实现。
			# 现阶段内部直接 print，无法逐行区分。折中：不改内部；早期 TTS 仅在最终结果前暂不实现 token 级触发（需对 _openai_commentary 做更细 Hook 才能真正实时）。
			pass
		# 流式早期 TTS：跟踪第一行换行后的第二行字符
		second_line_started = False
		second_line_buf = []
		early_fired = False
		def on_delta(delta: str):
			nonlocal second_line_started, early_fired
			if not args.stream or not args.tts or args.tts_early_chars <= 0 or tts_client is None:
				return
			if '\n' in delta and not second_line_started:
				# 可能包含换行及后续内容，拆分
				parts = delta.split('\n', 1)
				if len(parts) == 2:
					second_line_started = True
					second_line_buf.append(parts[1])
			elif second_line_started:
				second_line_buf.append(delta)
			# 判断触发
			if second_line_started and not early_fired:
				cur = ''.join(second_line_buf).strip()
				if len(cur) >= args.tts_early_chars:
					tts_client.enqueue(cur)
					early_fired = True
		commentary = _openai_commentary(system, user, model=args.model, debug=getattr(args, "debug_openai", False), stream=args.stream, strict=getattr(args, "strict_model", False), on_delta=on_delta)
		latency = time.time() - start_call
		# 更新移动平均延迟 & 自适应 interval
		if latency > 0:
			nonlocal_latency = latency  # for closure clarity
			# 使用外部变量 latency_mavg
			# (Python 3.8+ 需要 nonlocal 声明)
		pass
		with lock:
			ctx.last_output = commentary
			ctx.summary = (ctx.summary + " " + commentary)[:160]
			# 尝试从第一行提取比分模式 '比分: A-B'
			first_line = commentary.splitlines()[0] if commentary else ""
			m2 = re.search(r"Score:\s*(\d+)\s*-\s*(\d+)", first_line)
			if m2:
				ctx.score = f"{m2.group(1)}-{m2.group(2)}"
		if not args.stream:
			print(f"[commentary] {commentary}")
		# 提取第二行做 TTS（若早期已触发，仍再合成一次可能导致重复；后续可做去重）
		if tts_client is not None:
			lines = commentary.splitlines()
			if len(lines) >= 2:
				second = lines[1].strip()
				if len(second) > 4:
					tts_client.enqueue(second)
		print(f"[latency] model_call={latency*1000:.1f} ms window={len(frames_batch)} frames interval={tempo.cfg.target_interval_s:.2f}s")
		# 自适应逻辑（在打印后进行）
		if args.adaptive:
			if 'latency_mavg' in globals():  # avoid lint warning
				pass
			# 使用外层变量
			nonlocal latency_mavg
			if latency_mavg == 0.0:
				latency_mavg = latency
			else:
				latency_mavg = latency_mavg * (1 - alpha) + latency * alpha
			desired = max(latency_mavg * 1.05, args.min_interval)
			# 若明显偏离才调整，避免过度抖动
			current = tempo.cfg.target_interval_s
			if desired < current * 0.85 or desired > current * 1.15:
				new_interval = max(min(desired, args.max_interval), args.min_interval)
				if abs(new_interval - current) > 1e-3:
					tempo.cfg.target_interval_s = new_interval
					print(f"[adaptive] adjust interval {current:.2f} -> {new_interval:.2f} (latency_avg={latency_mavg*1000:.1f}ms)")
		# 若开启 live-log, 追加写入 NDJSON
		if args.live_log:
			try:
				import json as _json
				log_entry = {
					"ts": current_ts,
					"wall_time": time.time(),
					"text": commentary,
					"score": ctx.score,
					"audio": list(audio_batch) if audio_batch else [],
				}
				with open(args.live_log, "a", encoding="utf-8") as lf:
					lf.write(_json.dumps(log_entry, ensure_ascii=False) + "\n")
			except Exception as _e:  # noqa: BLE001
				print(f"[live-log][warn] 写入失败: {_e}")
			finally:
				audio_batch.clear()
		# 更新上次触发时间
		nonlocal last_fire_wall
		last_fire_wall = time.time()

	first_frame_ts: Optional[float] = None
	last_edge_ratio: Optional[float] = None
	for fr in extract_frames(
		video_path=args.video,
		target_fps=args.fps,
		max_frames=None if args.max_frames < 0 else args.max_frames,
		dump_dir=dump_dir,
		realtime=args.realtime,
		start_time_offset=args.start_offset,
	):
		print(f"[frame] idx={fr.index} ts={fr.timestamp:.2f}s")
		recent_frames.append(fr)
		if first_frame_ts is None:
			first_frame_ts = fr.timestamp
		elif args.duration > 0 and (fr.timestamp - first_frame_ts) >= args.duration:
			print(f"[runner] 达到设定时长 {args.duration}s，提前结束。")
			break
		batch = tempo.push(fr)
		if batch:
			process_frames(batch)
		elif args.event_trigger:
			# 事件触发检测：边缘比率变化
			er = edge_ratio_fast(fr.image)
			fire = False
			if last_edge_ratio is not None and abs(er - last_edge_ratio) >= args.event_threshold:
				# 时间间隔满足 event_gap
				if (time.time() - last_fire_wall) >= args.event_gap:
					fire = True
			if fire:
				frames_for_event = list(recent_frames)
				if len(frames_for_event) >= 2:
					print(f"[event] edge_change {last_edge_ratio:.3f}->{er:.3f} triggering early commentary")
					process_frames(frames_for_event)
			last_edge_ratio = er
		else:
			last_edge_ratio = edge_ratio_fast(fr.image)
		if args.show:
			cv2.imshow("frame", fr.image)
			if cv2.waitKey(1) & 0xFF == 27:  # ESC
				break
	if args.show:
		cv2.destroyAllWindows()
	print("[runner] 结束 (实时解说模式)")
	if tts_client is not None:
		try:
			tts_client.close()
		except Exception:
			pass


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover (CLI)
	args = parse_args(argv)
	# 若指定探测模式，先执行探测直接退出
	if args.probe_models:
		probe_list = [m.strip() for m in args.probe_models.split(',') if m.strip()]
		if not probe_list:
			print("[probe] 未解析出有效模型名称")
			return
		try:
			_probe_models(probe_list, debug=getattr(args, "debug_openai", False))
		except Exception as e:  # pragma: no cover
			print(f"[probe] 发生异常: {e}")
		return
	run_realtime_commentary(args)


if __name__ == "__main__":  # pragma: no cover
	main()

