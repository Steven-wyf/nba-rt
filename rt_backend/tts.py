"""tts.py

低延迟短句 TTS：
 - 入口: TTSClient.enqueue(text) 异步合成
 - 默认使用 OpenAI 新版语音接口 (audio.speech.create)；若失败打印降级日志
 - 自动写入 data/cache/tts/*.mp3 便于前端或后续回放
 - 可选自动播放：优先 ffplay -> aplay；失败仅落盘

设计目标：
 - 非阻塞：主线程只做 enqueue；后台线程串行合成，避免并发抢占 API
 - 短句 (< ~160 chars) 低延迟；长句自动切分句号/逗号
 - 可扩展：后续可接入本地 TTS / 其它云厂商
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import threading
import time
import os
import shutil
import sys
from typing import Optional, List

try:  # 用户可能还没装 openai
	import openai  # type: ignore
except Exception:  # pragma: no cover
	openai = None  # type: ignore


@dataclass(slots=True)
class TTSConfig:
	model: str = "gpt-4o-mini-tts"          # 语音模型（账号可用性决定）
	voice: str = "alloy"                    # 语音名称
	format: str = "mp3"                     # 仅用于落盘文件扩展；部分 SDK 不再接受 format 参数
	autoplay: bool = True                    # 合成后尝试本地播放
	max_chunk_chars: int = 180               # 超长文本分块上限
	play_cmd_timeout: float = 15.0           # 播放超时秒
	max_queue: int = 12                      # 队列上限，>0 启用丢弃最旧


class TTSClient:
	def __init__(self, cfg: Optional[TTSConfig] = None):
		self.cfg = cfg or TTSConfig()
		self._q: "Queue[str]" = Queue()
		self._stop = threading.Event()
		self._worker = threading.Thread(target=self._run, daemon=True)
		self.cache_dir = Path("data/cache/tts")
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		self._worker.start()

	# --------------------------- Public API --------------------------- #
	def enqueue(self, text: str) -> None:
		text = (text or "").strip()
		if not text:
			return
		# 若设置队列上限，提前丢弃旧元素
		if self.cfg.max_queue > 0:
			while self._q.qsize() >= self.cfg.max_queue:
				try:
					_ = self._q.get_nowait()
					print("[tts][drop-old] queue overflow, discard oldest chunk")
				except Empty:
					break
		# 拆句（粗略）
		if len(text) > self.cfg.max_chunk_chars:
			chunks = self._split_text(text, self.cfg.max_chunk_chars)
			for ck in chunks:
				self._q.put(ck)
		else:
			self._q.put(text)

	def close(self) -> None:
		self._stop.set()
		self._q.put("__TTS_STOP__")
		self._worker.join(timeout=2.0)

	# --------------------------- Worker Loop -------------------------- #
	def _run(self) -> None:  # pragma: no cover (运行期行为)
		while not self._stop.is_set():
			try:
				item = self._q.get(timeout=0.5)
			except Empty:
				continue
			if item == "__TTS_STOP__":
				break
			try:
				audio_path = self._synthesize(item)
				if self.cfg.autoplay and audio_path is not None:
					self._play(audio_path)
			except Exception as e:  # noqa: PIE786
				print(f"[tts][error] {e}", file=sys.stderr)
			finally:
				self._q.task_done()

	# --------------------------- Synthesis ---------------------------- #
	def _synthesize(self, text: str) -> Optional[Path]:
		if openai is None:
			print("[tts] openai 未安装，跳过语音合成。")
			return None
		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			print("[tts] 缺少 OPENAI_API_KEY，跳过。")
			return None
		try:
			from openai import OpenAI  # type: ignore
			client = OpenAI(api_key=api_key)
			start = time.time()
			# 新 SDK 语法：不再使用 format 参数；默认返回 mp3 bytes 或可读流
			# 优先直接调用 create；若失败尝试 streaming 上下文。
			audio_bytes: Optional[bytes] = None
			try:
				resp = client.audio.speech.create(  # type: ignore[attr-defined]
					model=self.cfg.model,
					voice=self.cfg.voice,
					input=text,
				)
			except TypeError as te:
				# 可能需要使用 streaming API
				if "unexpected keyword" in str(te).lower():
					resp = None
				else:
					raise
			if 'resp' in locals() and resp is not None:
				if hasattr(resp, 'read'):
					try:
						audio_bytes = resp.read()  # type: ignore
					except Exception:
						pass
				if audio_bytes is None:
					if isinstance(resp, bytes):
						audio_bytes = resp
					else:
						# 常见字段探测
						for cand in ('content', 'data', 'audio'):
							if hasattr(resp, cand):
								v = getattr(resp, cand)
								if isinstance(v, bytes):
									audio_bytes = v
									break
								if isinstance(v, str):  # 可能是 b64
									# 尝试 base64 解码
									import base64
									try:
										audio_bytes = base64.b64decode(v)
										break
									except Exception:
										continue
			if audio_bytes is None:
				# 尝试 streaming 模式
				try:
					stream_ctx = client.audio.speech.with_streaming_response.create(  # type: ignore[attr-defined]
						model=self.cfg.model,
						voice=self.cfg.voice,
						input=text,
					)
					from contextlib import closing
					with closing(stream_ctx) as sr:
						# 直接读到内存
						try:
							audio_bytes = sr.read()  # type: ignore[attr-defined]
						except Exception:
							# 如果有 stream_to_file 接口，可以先写临时文件再读
							tmp_path = self.cache_dir / f"_tmp_stream_{int(time.time()*1000)}.bin"
							try:
								sr.stream_to_file(tmp_path)  # type: ignore[attr-defined]
								audio_bytes = tmp_path.read_bytes()
								tmp_path.unlink(missing_ok=True)
							except Exception:
								pass
				except Exception as e_stream:
					print(f"[tts][debug] streaming fallback failed: {e_stream}")
			if not audio_bytes:
				print("[tts] 空音频响应或模型暂不可用，跳过。")
				return None
			ext = self.cfg.format.lower() if self.cfg.format else 'mp3'
			if ext not in ('mp3', 'wav', 'ogg'):  # 简单白名单
				ext = 'mp3'
			fn = self.cache_dir / f"tts_{int(time.time()*1000)}.{ext}"
			fn.write_bytes(audio_bytes)
			dur = (time.time() - start) * 1000
			print(f"[tts] synthesized {len(audio_bytes)} bytes in {dur:.1f} ms -> {fn}")
			return fn
		except Exception as e:
			print(f"[tts][fallback] 合成失败: {e}")
			return None

	# ---------------------------- Playback ---------------------------- #
	def _play(self, path: Path) -> None:
		# 优先 ffplay
		if shutil.which("ffplay"):
			self._spawn(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
			return
		# 次选 aplay (wav)；若 mp3 且只有 aplay 则放弃
		if path.suffix.lower() == ".wav" and shutil.which("aplay"):
			self._spawn(["aplay", str(path)])
			return
		# 无播放器
		print(f"[tts] (no player) saved: {path}")

	def _spawn(self, cmd: List[str]) -> None:
		try:
			import subprocess
			subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except Exception as e:  # pragma: no cover
			print(f"[tts] 播放失败 {cmd}: {e}")

	# -------------------------- Text Split ---------------------------- #
	def _split_text(self, text: str, limit: int) -> List[str]:
		seps = ["。", "?", "!", ".", "，", ","]
		parts: List[str] = []
		buf = ""
		for ch in text:
			buf += ch
			if ch in seps and len(buf) >= limit * 0.5:
				parts.append(buf.strip())
				buf = ""
			elif len(buf) >= limit:
				parts.append(buf.strip())
				buf = ""
		if buf.strip():
			parts.append(buf.strip())
		return parts


__all__ = ["TTSClient", "TTSConfig"]

