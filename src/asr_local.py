from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
import wave
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Deque, Dict, List, Optional


log = logging.getLogger(__name__)


@dataclass
class LocalASRConfig:
    """Configuration for the local ASR worker."""

    model_path: str
    binary_path: str
    language: str = "ru"
    sample_rate: int = 16000
    context_seconds: float = 30.0
    max_queue: int = 32


@dataclass
class LocalTranscriptSegment:
    text: str
    start_ts: float
    end_ts: float


class LocalTranscriber:
    """Async wrapper around whisper.cpp (or compatible) with a rolling transcript buffer."""

    def __init__(
        self,
        cfg: LocalASRConfig,
        *,
        memory: Optional["MemoryStore"] = None,
        speaker_lookup: Optional[Callable[[int], str]] = None,
    ) -> None:
        self.cfg = cfg
        self._queue: asyncio.Queue[tuple[int, float, float, bytes]] = asyncio.Queue(maxsize=cfg.max_queue)
        self._events: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._buffers: Dict[int, Deque[LocalTranscriptSegment]] = defaultdict(deque)
        self._worker_task: Optional[asyncio.Task] = None
        self._available = self._detect_availability()
        self._shutdown = False
        self._memory = memory
        self._speaker_lookup = speaker_lookup
        if self._available:
            log.info("Local ASR available via whisper.cpp (binary=%s, model=%s)", cfg.binary_path, cfg.model_path)
        else:
            log.warning("Local ASR not available; configure whisper.cpp binary/model for offline transcription")

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="local-asr-worker")

    async def close(self) -> None:
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(Exception):
                await self._worker_task
            self._worker_task = None

    async def push_audio(self, user_id: int, pcm16: bytes, ts: float, duration: float) -> None:
        """Enqueue raw PCM16 mono audio for transcription."""
        if self._shutdown:
            return
        try:
            await self._queue.put((int(user_id), float(ts), float(duration), pcm16))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Failed to enqueue audio for local ASR")

    async def inject_transcript(
        self,
        user_id: int,
        text: str,
        ts: Optional[float] = None,
        duration: float = 0.0,
    ) -> None:
        """Testing helper: inject a ready-made transcript without audio processing."""
        stamp = ts if ts is not None else time.time()
        self._append_segment(user_id, text, stamp, stamp + duration)
        await self._log_to_memory(user_id, stamp, text)
        await self._events.put(
            {
                "user_id": int(user_id),
                "text": text,
                "ts": stamp,
                "final": True,
                "duration": duration,
            }
        )

    async def events(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            evt = await self._events.get()
            yield evt

    def set_speaker_lookup(self, fn: Callable[[int], str]) -> None:
        self._speaker_lookup = fn

    @property
    def available(self) -> bool:
        return self._available

    def recent_text(self, user_id: int, seconds: Optional[float] = None, now: Optional[float] = None) -> str:
        """Return the concatenated recent transcript for a user."""
        segments = self.recent_segments(seconds=seconds, now=now, user_id=user_id)
        return " ".join(seg["text"] for seg in segments if seg.get("text"))

    def recent_segments(
        self,
        seconds: Optional[float] = None,
        now: Optional[float] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ref = now if now is not None else time.time()
        cutoff = ref - float(seconds) if seconds is not None else None
        entries: List[Dict[str, Any]] = []
        users = [int(user_id)] if user_id is not None else list(self._buffers.keys())
        for uid in users:
            for seg in self._buffers.get(int(uid), []):
                if cutoff is not None and seg.end_ts < cutoff:
                    continue
                entries.append(
                    {
                        "user_id": int(uid),
                        "speaker": self._resolve_speaker_name(int(uid)),
                        "text": seg.text,
                        "timestamp": seg.start_ts,
                    }
                )
        entries.sort(key=lambda item: item.get("timestamp", 0))
        return entries[-200:]

    def _detect_availability(self) -> bool:
        model_path = Path(self.cfg.model_path) if self.cfg.model_path else None
        binary_path = Path(self.cfg.binary_path) if self.cfg.binary_path else None
        return bool(model_path and model_path.exists() and binary_path and binary_path.exists())

    async def _worker(self) -> None:
        while True:
            try:
                user_id, ts, duration, pcm = await self._queue.get()
                text = await self._transcribe(pcm)
                if not text:
                    continue
                end_ts = ts + duration
                self._append_segment(user_id, text, ts, end_ts)
                await self._log_to_memory(user_id, ts, text)
                await self._events.put(
                    {
                        "user_id": user_id,
                        "text": text,
                        "ts": ts,
                        "final": True,
                        "duration": duration,
                    }
                )
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Local ASR worker failed")

    async def _transcribe(self, pcm16: bytes) -> str:
        if not pcm16:
            return ""
        if not self._available:
            return ""
        try:
            return await self._run_whisper_cpp(pcm16)
        except FileNotFoundError:
            log.warning("whisper.cpp binary or model missing; disabling local ASR")
            self._available = False
        except Exception:
            log.exception("whisper.cpp invocation failed; disabling local ASR")
            self._available = False
        return ""

    async def _run_whisper_cpp(self, pcm16: bytes) -> str:
        assert self.cfg.sample_rate > 0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            path = Path(wav_file.name)
        try:
            self._write_wav(path, pcm16)
            cmd = [
                self.cfg.binary_path,
                "--model",
                self.cfg.model_path,
                "--language",
                self.cfg.language,
                "--output-json",
                "--print-progress",
                "false",
                str(path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.warning("whisper.cpp exited with %s: %s", proc.returncode, stderr.decode("utf-8", "ignore"))
                return ""
            if not stdout:
                return ""
            # whisper.cpp --output-json emits JSON per segment; concatenate text fields
            text_parts: list[str] = []
            for line in stdout.splitlines():
                try:
                    record = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(record, dict):
                    seg_text = record.get("text")
                    if isinstance(seg_text, str):
                        text_parts.append(seg_text.strip())
            return " ".join(text_parts)
        finally:
            with contextlib.suppress(Exception):
                os.remove(path)

    def _append_segment(self, user_id: int, text: str, start_ts: float, end_ts: float) -> None:
        buffer = self._buffers[int(user_id)]
        buffer.append(LocalTranscriptSegment(text=text.strip(), start_ts=start_ts, end_ts=end_ts))
        self._prune_buffer(buffer)

    def _prune_buffer(self, buffer: Deque[LocalTranscriptSegment]) -> None:
        if not buffer:
            return
        cutoff = time.time() - float(self.cfg.context_seconds)
        while buffer and buffer[0].end_ts < cutoff:
            buffer.popleft()

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        with contextlib.ExitStack() as stack:
            wf = stack.enter_context(wave.open(path.as_posix(), "wb"))
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm)

    def _resolve_speaker_name(self, user_id: int) -> str:
        if self._speaker_lookup:
            try:
                return self._speaker_lookup(int(user_id))
            except Exception:
                pass
        return str(user_id)

    async def _log_to_memory(self, user_id: int, ts: float, text: str) -> None:
        if not self._memory or not text.strip():
            return
        from .memory import MemoryEntry  # local import to avoid cycle

        entry = MemoryEntry(
            timestamp=float(ts),
            user_id=int(user_id),
            speaker=self._resolve_speaker_name(user_id),
            text=text.strip(),
        )
        try:
            await self._memory.log_event(entry)
        except Exception:
            log.exception("Failed to log local ASR event to memory")
