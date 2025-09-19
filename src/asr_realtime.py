from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional
import logging

import aiohttp
import contextlib


class WakeWordDetector:
    """Lightweight text-based keyword spotter for Russian wake phrases."""

    def __init__(
        self,
        phrases: list[str],
        fallback_phrases: list[str],
        confidence_threshold: float,
        enabled: bool = True,
        cooldown_seconds: float = 2.5,
        max_chars: int = 160,
    ) -> None:
        self.enabled = enabled
        self._primary_map = {self._normalize(p): p for p in phrases if p}
        self._fallback_map = {self._normalize(p): p for p in fallback_phrases if p}
        self._confidence_threshold = confidence_threshold
        self._cooldown = cooldown_seconds
        self._max_chars = max_chars
        self._buffer = ""
        self._last_detection_ts = 0.0
        self._recent_phrase: Optional[str] = None

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> Optional["WakeWordDetector"]:
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return None
        wake = cfg.get("wake_phrase")
        fallback = cfg.get("fallback_phrases") or []
        if isinstance(fallback, str):
            fallback = [fallback]
        phrases: list[str] = []
        if isinstance(wake, str) and wake.strip():
            phrases.append(wake.strip())
        threshold = float(cfg.get("confidence_threshold", 0.5) or 0.0)
        cooldown = float(cfg.get("cooldown", 2.5) or 2.5)
        return WakeWordDetector(phrases, [str(f).strip() for f in fallback if str(f).strip()], threshold, enabled=True, cooldown_seconds=cooldown)

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower()
        return "".join(ch for ch in lowered if ch.isalnum())

    def reset(self) -> None:
        self._buffer = ""
        self._recent_phrase = None
        self._last_detection_ts = 0.0

    def process(
        self,
        fragment: str,
        confidence: Optional[float],
        *,
        is_final: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            if is_final:
                self._prune_buffer()
            return None
        piece = fragment.strip()
        if piece:
            if self._buffer:
                self._buffer = f"{self._buffer} {piece}"
            else:
                self._buffer = piece
            if len(self._buffer) > self._max_chars:
                self._buffer = self._buffer[-self._max_chars :]

        if not self._buffer:
            if is_final:
                self._prune_buffer()
            return None

        normalized = self._normalize(self._buffer)
        detection = self._match(normalized, confidence)
        if detection:
            self._mark_detected(detection["wake_phrase_norm"])
            detection["wake_confidence"] = confidence
        if is_final:
            self._prune_buffer()
        return detection

    def _prune_buffer(self) -> None:
        if self._buffer:
            self._buffer = self._buffer[-(self._max_chars // 2) :]

    def _mark_detected(self, phrase_norm: Optional[str]) -> None:
        self._recent_phrase = phrase_norm
        self._last_detection_ts = time.time()

    def _match(self, normalized: str, confidence: Optional[float]) -> Optional[Dict[str, Any]]:
        now = time.time()
        for norm, original in self._primary_map.items():
            if norm and norm in normalized:
                is_new = self._is_new_detection(norm, now)
                return {
                    "wake_detected": True,
                    "wake_phrase": original,
                    "wake_phrase_norm": norm,
                    "wake_source": "primary",
                    "wake_is_new": is_new,
                }
        for norm, original in self._fallback_map.items():
            if not norm or norm not in normalized:
                continue
            if confidence is not None and confidence < self._confidence_threshold:
                continue
            is_new = self._is_new_detection(norm, now)
            return {
                "wake_detected": True,
                "wake_phrase": original,
                "wake_phrase_norm": norm,
                "wake_source": "fallback",
                "wake_is_new": is_new,
            }
        return None

    def _is_new_detection(self, phrase_norm: str, now: float) -> bool:
        if not self._recent_phrase:
            return True
        if self._recent_phrase != phrase_norm:
            return True
        return (now - self._last_detection_ts) > self._cooldown


class RealtimeASR:
    """
    Minimal client for OpenAI Realtime ASR over WebSocket, with HTTP Whisper fallback.

    Usage:
      async with RealtimeASR(...) as asr:
          await asr.connect()
          await asr.send_audio(pcm16_mono_16k)
          async for evt in asr.events():
              ...  # {user_id, text, final, ts}
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        whisper_model: str = "whisper-1",
        user_id: int | None = None,
        *,
        window_seconds: float = 10.0,
        stride_seconds: float = 3.0,
        kws_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.whisper_model = whisper_model
        self.user_id = user_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._q: asyncio.Queue[Dict] = asyncio.Queue()
        self._rx_task: Optional[asyncio.Task] = None
        self._connected = False
        self._log = logging.getLogger(__name__)
        self._accum_text: str = ""
        self._sample_rate = 16000
        self._window_seconds = max(window_seconds, stride_seconds)
        self._stride_seconds = max(0.5, stride_seconds)
        self._window_samples = int(self._sample_rate * self._window_seconds)
        self._stride_samples = max(1, int(self._sample_rate * self._stride_seconds))
        self._buffer = bytearray()
        self._ready_samples = 0
        self._pending_force = False
        self._has_sent_window = False
        self._flush_event = asyncio.Event()
        self._flush_task: Optional[asyncio.Task] = None
        kws_conf = kws_config or {}
        if "fallback_phrases" not in kws_conf and kws_conf.get("fallback_phrase"):
            kws_conf = dict(kws_conf)
            fp = kws_conf.pop("fallback_phrase")
            kws_conf["fallback_phrases"] = [fp]
        self._wake_detector = WakeWordDetector.from_config({
            "enabled": kws_conf.get("enabled", False),
            "wake_phrase": kws_conf.get("wake_phrase"),
            "fallback_phrases": kws_conf.get("fallback_phrases", []),
            "confidence_threshold": kws_conf.get("confidence_threshold", 0.5),
            "cooldown": kws_conf.get("cooldown", 2.5),
        })
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._buffer.clear()
        self._ready_samples = 0
        self._pending_force = False
        self._has_sent_window = False
        self._flush_event.clear()
        if self._wake_detector:
            self._wake_detector.reset()

    async def __aenter__(self) -> "RealtimeASR":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected:
            return
        self._session = aiohttp.ClientSession()
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            self._ws = await self._session.ws_connect(url, headers=headers, heartbeat=15)
            self._connected = True
            self._reset_stream_state()
            self._rx_task = asyncio.create_task(self._reader())
            self._flush_task = asyncio.create_task(self._flush_loop())
        except Exception:
            # Fallback to HTTP mode (no true streaming)
            self._connected = False
            self._log.warning("Realtime WS connect failed; will use HTTP fallback for ASR")

    async def close(self) -> None:
        if self._rx_task:
            self._rx_task.cancel()
            with contextlib.suppress(Exception):
                await self._rx_task
        if self._flush_task:
            self._flush_task.cancel()
            with contextlib.suppress(Exception):
                await self._flush_task
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._ws = None
        self._session = None
        self._connected = False
        self._flush_task = None
        self._reset_stream_state()

    async def _reader(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                # Heuristic parsing of realtime events
                et = data.get("type") or data.get("event")
                ts = time.time()
                self._log.debug("Realtime event: %s", et)
                if et in ("response.created",):
                    self._accum_text = ""
                if et in ("response.output_text.delta", "transcript.delta", "response.delta", "response.text.delta"):
                    raw_text = data.get("delta") or data.get("text") or data.get("output_text") or ""
                    text = str(raw_text)
                    if text:
                        self._accum_text += text
                    conf_val = data.get("confidence")
                    try:
                        conf = float(conf_val) if conf_val is not None else None
                    except (TypeError, ValueError):
                        conf = None
                    evt: Dict[str, Any] = {
                        "user_id": self.user_id or 0,
                        "text": text,
                        "final": False,
                        "confidence": conf,
                        "ts": ts,
                    }
                    if self._wake_detector and text:
                        detection = self._wake_detector.process(text, conf, is_final=False)
                        if detection:
                            evt.update(detection)
                    await self._q.put(evt)
                elif et in ("response.text.done", "response.completed", "transcript.completed", "response.output_text.done"):
                    raw_text = data.get("text") or data.get("output_text") or self._accum_text or ""
                    text = str(raw_text)
                    self._accum_text = ""
                    conf_val = data.get("confidence")
                    try:
                        conf = float(conf_val) if conf_val is not None else None
                    except (TypeError, ValueError):
                        conf = None
                    evt = {
                        "user_id": self.user_id or 0,
                        "text": text,
                        "final": True,
                        "confidence": conf,
                        "ts": ts,
                    }
                    if self._wake_detector and text:
                        detection = self._wake_detector.process(text, conf, is_final=True)
                        if detection:
                            evt.update(detection)
                    await self._q.put(evt)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                # Ignore
                pass

    async def events(self) -> AsyncGenerator[Dict, None]:
        while True:
            evt = await self._q.get()
            yield evt

    async def send_audio(self, pcm16_mono_16k: bytes, *, force_flush: bool = False) -> None:
        if not pcm16_mono_16k:
            if force_flush:
                await self.flush()
            return
        if not self._connected or self._ws is None:
            await self._send_http_fallback(pcm16_mono_16k)
            return

        self._buffer.extend(pcm16_mono_16k)
        max_bytes = self._window_samples * 2
        if len(self._buffer) > max_bytes:
            self._buffer = self._buffer[-max_bytes:]
        self._ready_samples += len(pcm16_mono_16k) // 2
        if force_flush:
            self._pending_force = True
        self._flush_event.set()

    async def flush(self, force: bool = True) -> None:
        if not self._connected or self._ws is None:
            return
        if force:
            self._pending_force = True
        self._flush_event.set()

    async def _send_http_fallback(self, pcm16_mono_16k: bytes) -> None:
        text = await self._transcribe_http(pcm16_mono_16k)
        evt: Dict[str, Any] = {
            "user_id": self.user_id or 0,
            "text": text,
            "final": True,
            "confidence": None,
            "ts": time.time(),
        }
        if self._wake_detector and text:
            detection = self._wake_detector.process(text, None, is_final=True)
            if detection:
                evt.update(detection)
        await self._q.put(evt)

    async def _flush_loop(self) -> None:
        try:
            while True:
                await self._flush_event.wait()
                self._flush_event.clear()
                if not self._connected or self._ws is None:
                    continue
                while self._buffer:
                    should_flush = False
                    if self._pending_force:
                        should_flush = True
                    elif not self._has_sent_window:
                        should_flush = True
                    elif self._ready_samples >= self._stride_samples:
                        should_flush = True
                    if not should_flush:
                        break
                    window_bytes = min(len(self._buffer), self._window_samples * 2)
                    if window_bytes <= 0:
                        break
                    audio = bytes(self._buffer[-window_bytes:])
                    await self._transmit_window(audio)
                    if self._ready_samples >= self._stride_samples:
                        self._ready_samples -= self._stride_samples
                    else:
                        self._ready_samples = 0
                    self._pending_force = False
                    self._has_sent_window = True
        except asyncio.CancelledError:
            pass

    async def _transmit_window(self, pcm16_mono_16k: bytes) -> None:
        if not pcm16_mono_16k:
            return
        if not self._connected or self._ws is None:
            await self._send_http_fallback(pcm16_mono_16k)
            return
        b64 = base64.b64encode(pcm16_mono_16k).decode("ascii")
        try:
            await self._ws.send_json({"type": "input_audio_buffer.clear"})
            await self._ws.send_json({"type": "input_audio_buffer.append", "audio": b64})
            await self._ws.send_json({"type": "input_audio_buffer.commit"})
            await self._ws.send_json({
                "type": "response.create",
                "response": {"modalities": ["text"], "instructions": "Transcribe the audio input."},
            })
        except Exception:
            self._log.warning("Realtime stream send failed; falling back to HTTP ASR", exc_info=True)
            self._connected = False
            self._reset_stream_state()
            await self._send_http_fallback(pcm16_mono_16k)

    async def _transcribe_http(self, pcm16_mono_16k: bytes) -> str:
        url = "https://api.openai.com/v1/audio/transcriptions"
        assert self._session is not None
        data = aiohttp.FormData()
        data.add_field("model", self.whisper_model)
        # Package as WAV mono 16k headerless raw is not accepted; create minimal WAV on the fly
        wav_bytes = _pcm16_to_wav_bytes(pcm16_mono_16k, 16000, 1)
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self._session.post(url, headers=headers, data=data) as resp:
            resp.raise_for_status()
            jd = await resp.json()
            return jd.get("text") or ""


def _pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
