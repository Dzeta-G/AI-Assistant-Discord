from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import discord
import yaml

from .asr_realtime import RealtimeASR
from .asr_local import LocalASRConfig, LocalTranscriber
from .ark_agent import ARKAgent
from .memory import MemoryStore, is_recall_query
from .router import Router
from .tts_realtime import TTSClient
from .vad import VoiceActivityDetector, VADConfig, resample_mono_pcm16, rms_dbfs

log = logging.getLogger(__name__)


@dataclass
class UserStream:
    user_id: int
    vad: VoiceActivityDetector
    asr: RealtimeASR
    consumer_task: Optional[asyncio.Task] = None
    last_voice_ts: float = 0.0
    last_flush_ts: float = 0.0


@dataclass
class AudioChunk:
    timestamp: float
    pcm: bytes
    duration: float
    sent: bool = False


class VoicePipeline:
    def __init__(
        self,
        voice: discord.VoiceClient,
        router: Router | None,
        ark: ARKAgent,
        tts: Optional[TTSClient],
        openai_key: str,
        rt_model: str,
        whisper_model: str,
        memory: Optional[MemoryStore] = None,
    ) -> None:
        self.voice = voice
        base = Path(__file__).resolve().parent
        campaign_cfg: Dict[str, Any] = {}
        try:
            with open(base / "config" / "campaign.yml", "r", encoding="utf-8") as fh:
                campaign_cfg = yaml.safe_load(fh) or {}
        except Exception:
            log.debug("Failed loading campaign config for voice pipeline", exc_info=True)
        self._campaign_cfg: Dict[str, Any] = campaign_cfg
        self._asr_cfg: Dict[str, Any] = dict((campaign_cfg.get("asr") or {}))
        # Be resilient if router wasn't supplied (shouldn't happen, but logs show None)
        if router is None:
            log.warning("VoicePipeline initialized without router; falling back to config router")
            try:
                self.router = Router.from_files(str(base / "config" / "campaign.yml"), str(base / "config" / "mapping.yml"))
            except Exception:
                log.exception("Failed to load fallback router configuration")
                self.router = None  # final guard; downstream will skip
        else:
            self.router = router
        if ark is None:
            log.warning("VoicePipeline initialized without ARK agent; falling back to config agent")
            try:
                system_prompt = (
                    campaign_cfg.get("ark", {}).get("system_prompt")
                    or campaign_cfg.get("system_prompt", "")
                )
                self.ark = ARKAgent(api_key=os.environ.get("OPENAI_API_KEY", ""), system_prompt=system_prompt)
            except Exception:
                log.exception("Failed to build fallback ARK agent")
                self.ark = None
        else:
            self.ark = ark
        if tts is None:
            log.warning("VoicePipeline initialized without TTS client; falling back to config TTS")
            try:
                model = os.environ.get("OPENAI_TTS_MODEL") or campaign_cfg.get("tts", {}).get("model")
                voice_name = os.environ.get("OPENAI_TTS_VOICE") or campaign_cfg.get("tts", {}).get("voice") or "alloy"
                if model:
                    self.tts = TTSClient(api_key=os.environ.get("OPENAI_API_KEY", ""), model=model, voice=voice_name)
                else:
                    self.tts = None
            except Exception:
                log.exception("Failed to build fallback TTS client")
                self.tts = None
        else:
            self.tts = tts
        chunk_seconds: float | None = None
        chunk_env = os.environ.get("VOICE_CHUNK_SECONDS")
        if chunk_env:
            try:
                chunk_seconds = float(chunk_env)
            except ValueError:
                log.warning("Invalid VOICE_CHUNK_SECONDS env value: %s", chunk_env)
        if chunk_seconds is None:
            cfg_value = campaign_cfg.get("asr", {}).get("chunk_seconds")
            if cfg_value is not None:
                try:
                    chunk_seconds = float(cfg_value)
                except (TypeError, ValueError):
                    log.warning("Invalid chunk_seconds in campaign config: %s", cfg_value)
        if not chunk_seconds or chunk_seconds < 1.0:
            chunk_seconds = 12.0
        self._chunk_seconds = float(chunk_seconds)
        self._window_seconds = self._resolve_float(os.environ.get("ASR_WINDOW_SECONDS"), self._asr_cfg.get("window_seconds"), 10.0)
        self._stride_seconds = self._resolve_float(os.environ.get("ASR_WINDOW_STRIDE"), self._asr_cfg.get("window_stride"), 3.0)
        if self._stride_seconds <= 0:
            self._stride_seconds = 3.0
        if self._window_seconds < self._stride_seconds:
            self._window_seconds = max(self._stride_seconds, 6.0)
        self._local_context_seconds = self._resolve_float(
            os.environ.get("ASR_LOCAL_CONTEXT_WINDOW"),
            self._asr_cfg.get("local_context_window") or self._asr_cfg.get("context_window"),
            90.0,
        )
        self._api_context_seconds = self._resolve_float(
            os.environ.get("ASR_CONTEXT_WINDOW"),
            self._asr_cfg.get("api_context_window") or self._asr_cfg.get("context_window"),
            30.0,
        )
        self._active_window_duration = self._resolve_float(
            os.environ.get("ASR_ACTIVE_WINDOW"), self._asr_cfg.get("active_window_duration"), 12.0
        )
        self.openai_key = openai_key
        self.rt_model = rt_model
        self.whisper_model = whisper_model
        self.memory = memory
        local_model_hint = self._resolve_path_hint(os.environ.get("LOCAL_ASR_MODEL") or self._asr_cfg.get("local_model"))
        binary_hint = self._resolve_path_hint(
            os.environ.get("LOCAL_ASR_BIN")
            or os.environ.get("WHISPER_CPP_BIN")
            or str(Path.cwd() / "whisper.cpp" / "main")
        )
        self.local_asr = LocalTranscriber(
            LocalASRConfig(
                model_path=local_model_hint or "",
                binary_path=binary_hint or "",
                context_seconds=self._local_context_seconds,
            ),
            memory=memory,
            speaker_lookup=self._resolve_speaker_name,
        )
        log.info(
            "VoicePipeline created (router=%s, ark=%s, tts=%s, chunk=%.2fs, window=%.1fs, stride=%.1fs, active=%.1fs)",
            bool(self.router),
            bool(self.ark),
            bool(self.tts),
            self._chunk_seconds,
            self._window_seconds,
            self._stride_seconds,
            self._active_window_duration,
        )
        self._streams: Dict[int, UserStream] = {}
        self._audio_buffers: Dict[int, Deque[AudioChunk]] = {}
        self._activation: Dict[int, float] = {}
        self._player_lock = asyncio.Lock()
        self._running = False
        self._stop_timer: Optional[asyncio.Task] = None
        self._local_consumer_task: Optional[asyncio.Task] = None
        # Energy fallback threshold (dBFS). If VAD yields no segments but
        # loudness is above this threshold, we push whole chunk to ASR.
        try:
            self.energy_dbfs_threshold = float(os.environ.get("VAD_ENERGY_DBFS", "-45"))
        except Exception:
            self.energy_dbfs_threshold = -45.0

    @staticmethod
    def _resolve_float(env_value: Optional[str], cfg_value: Any, default: float) -> float:
        for candidate in (env_value, cfg_value):
            if candidate is None:
                continue
            if isinstance(candidate, str) and not candidate.strip():
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _resolve_path_hint(hint: Optional[str]) -> str:
        if hint is None:
            return ""
        raw = str(hint).strip()
        if not raw:
            return ""
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate.as_posix()
        cwd_candidate = Path.cwd() / raw
        if cwd_candidate.exists():
            return cwd_candidate.as_posix()
        return raw

    def _resolve_speaker_name(self, user_id: int) -> str:
        try:
            if self.voice and getattr(self.voice, "guild", None):
                member = self.voice.guild.get_member(int(user_id))
                if member:
                    return member.display_name
        except Exception:
            pass
        if self.router:
            character = self.router.character_for(user_id)
            if character:
                return str(character)
        return str(user_id)

    def _collect_short_term_context(self) -> List[Dict[str, Any]]:
        if not self.local_asr:
            return []
        segments = self.local_asr.recent_segments(seconds=self._api_context_seconds)
        context: List[Dict[str, Any]] = []
        for seg in segments:
            context.append(
                {
                    "speaker": seg.get("speaker") or self._resolve_speaker_name(seg.get("user_id", 0)),
                    "text": seg.get("text", ""),
                    "ts": seg.get("timestamp", 0.0),
                    "meta": "short_term",
                }
            )
        return context[-50:]

    def _should_query_memory(self, text: str) -> bool:
        return is_recall_query(text)

    async def _collect_memory_context(self, query: str, speaker: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.memory or not query:
            return []
        try:
            results = await self.memory.search(query, limit=5, user_id=None)
        except Exception:
            log.exception("Memory search failed")
            return []
        context: List[Dict[str, Any]] = []
        for entry in results:
            when = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            context.append(
                {
                    "speaker": entry.speaker,
                    "text": f"[memory {when}] {entry.text}",
                    "ts": entry.timestamp,
                    "meta": "memory",
                }
            )
        return context

    @staticmethod
    def _dedupe_context(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str]] = set()
        deduped: List[Dict[str, Any]] = []
        for item in entries:
            key = (str(item.get("speaker")), str(item.get("text")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[-10:]

    @staticmethod
    def _normalize_transcript_text(text: Optional[str]) -> str:
        if not text:
            return ""
        if isinstance(text, str) and text.strip().startswith("{"):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ("transcription", "text"):
                        value = data.get(key)
                        if isinstance(value, str):
                            return value.strip()
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return str(text).strip()

    async def start(self) -> None:
        self._running = True
        # Start rolling recordings using sinks if available
        if not hasattr(discord, "sinks"):
            log.error("This discord package lacks voice receive sinks. Install py-cord>=2.5.")
            return
        if self.local_asr:
            self.local_asr.start()
            self._local_consumer_task = asyncio.create_task(self._consume_local_transcripts())
        self._start_recording_cycle()

    def stop(self) -> None:
        self._running = False
        with contextlib.suppress(Exception):
            self.voice.stop_recording()
        if self._stop_timer:
            self._stop_timer.cancel()
            self._stop_timer = None
        if self._local_consumer_task:
            self._local_consumer_task.cancel()
            self._local_consumer_task = None
        # Cancel consumers
        for us in list(self._streams.values()):
            if us.consumer_task:
                us.consumer_task.cancel()
            # Close ASR session
            try:
                asyncio.create_task(us.asr.close())
            except Exception:
                pass
        self._streams.clear()
        self._audio_buffers.clear()
        self._activation.clear()
        if self.local_asr:
            try:
                asyncio.create_task(self.local_asr.close())
            except Exception:
                pass

    def _start_recording_cycle(self) -> None:
        if not self._running:
            return
        seconds = self._chunk_seconds
        sink = discord.sinks.WaveSink()
        log.info("Starting recording cycle (%ss)...", seconds)
        # Start recording; the extra arg 'seconds' will be forwarded to callback but not used by py-cord itself
        self.voice.start_recording(sink, self._on_recording_finished)
        # Schedule a timed stop to flush chunk periodically
        self._stop_timer = asyncio.create_task(self._stop_after(seconds))

    async def _stop_after(self, seconds: int) -> None:
        try:
            await asyncio.sleep(seconds)
            if self._running and getattr(self.voice, "recording", True):
                log.debug("Stopping recording to flush chunk")
                self.voice.stop_recording()
        except asyncio.CancelledError:
            pass

    async def _on_recording_finished(self, sink: Any, *args: Any) -> None:
        # Called on the bot loop (py-cord schedules as coroutine)
        try:
            # Immediately schedule next cycle to keep continuous capture
            self._start_recording_cycle()
        except Exception as e:
            log.exception("Failed to restart recording: %s", e)
        # Process sink data
        try:
            # Give sink a brief moment to finalize WAVE headers
            await asyncio.sleep(0.2)
            audio_map: Dict[int, Any] = getattr(sink, "audio_data", {})
            log.info("Recording chunk finished; users captured: %s", list(audio_map.keys()))
            for user, adata in audio_map.items():
                user_id = int(getattr(adata, "user", user))
                raw_bytes: Optional[bytes] = None
                # Py-cord exposes .file (BytesIO) OR .file (path str) depending on sink impl, or .data (bytes)
                fobj = getattr(adata, "file", None)
                if isinstance(fobj, (bytes, bytearray)):
                    raw_bytes = bytes(fobj)
                elif hasattr(fobj, "read") and hasattr(fobj, "getvalue"):
                    # BytesIO-like
                    try:
                        raw_bytes = fobj.getvalue()
                    except Exception:
                        try:
                            fobj.seek(0)
                            raw_bytes = fobj.read()
                        except Exception:
                            raw_bytes = None
                elif isinstance(fobj, (str, Path)):
                    p = Path(fobj)
                    if p.exists():
                        raw_bytes = p.read_bytes()
                if raw_bytes is None:
                    raw_bytes = getattr(adata, "data", None)

                if not raw_bytes:
                    log.debug("No audio bytes for user %s in this chunk", user_id)
                    continue
                log.debug("User %s raw wav bytes: %d", user_id, len(raw_bytes))
                # Extract PCM from WAVE
                pcm, sr, ch = _wav_to_pcm(raw_bytes)
                if not pcm:
                    log.debug("Empty PCM for user %s; skipping", user_id)
                    continue
                pcm16k = resample_mono_pcm16(pcm, sr, 16000)
                if not pcm16k:
                    log.debug("Resampled chunk empty for user %s; skipping", user_id)
                    continue
                # Push into per-user VAD and ASR
                asyncio.create_task(self._push_user_audio(user_id, pcm16k))
        except Exception:
            log.exception("Error processing recording sink")

    async def _ensure_user_stream(self, user_id: int) -> UserStream:
        us = self._streams.get(user_id)
        if us:
            return us
        vad = VoiceActivityDetector(VADConfig())
        fallback_phrase = getattr(self.router, "asr_kws_wake_phrase", None) if self.router else None
        if not fallback_phrase:
            fallback_phrase = self._asr_cfg.get("kws_wake_phrase")
        fallback_list = [str(fallback_phrase)] if fallback_phrase else []
        wake_phrase = getattr(self.router, "wake_word", None) if self.router else None
        if not wake_phrase:
            wake_phrase = (self._campaign_cfg.get("ark", {}) or {}).get("wake_phrase")
        kws_enabled = getattr(self.router, "asr_kws_enabled", False) if self.router else bool(self._asr_cfg.get("kws_enabled"))
        conf_threshold = getattr(self.router, "asr_confidence_threshold", None) if self.router else None
        if conf_threshold is None:
            conf_threshold = self._resolve_float(None, self._asr_cfg.get("confidence_threshold"), 0.5)
        cooldown = self._resolve_float(os.environ.get("ASR_KWS_COOLDOWN"), self._asr_cfg.get("kws_cooldown"), 2.5)
        kws_config = {
            "enabled": kws_enabled,
            "wake_phrase": wake_phrase,
            "fallback_phrases": [f.strip() for f in fallback_list if str(f).strip()],
            "confidence_threshold": conf_threshold,
            "cooldown": cooldown,
        }
        asr = RealtimeASR(
            api_key=self.openai_key,
            model=self.rt_model,
            whisper_model=self.whisper_model,
            user_id=user_id,
            window_seconds=self._window_seconds,
            stride_seconds=self._stride_seconds,
            kws_config=kws_config,
        )
        await asr.connect()
        us = UserStream(user_id=user_id, vad=vad, asr=asr)
        us.consumer_task = asyncio.create_task(self._consume_asr_events(us))
        self._streams[user_id] = us
        return us

    async def _push_user_audio(self, user_id: int, pcm16k: bytes) -> None:
        us = await self._ensure_user_stream(user_id)
        now = time.time()
        duration = len(pcm16k) / (2 * 16000)
        buffer = self._audio_buffers.setdefault(int(user_id), deque())
        chunk = AudioChunk(timestamp=now, pcm=pcm16k, duration=duration)
        buffer.append(chunk)
        self._prune_audio_buffer(int(user_id))

        local_ready = bool(self.local_asr and getattr(self.local_asr, "available", False))
        if self.local_asr:
            try:
                await self.local_asr.push_audio(user_id, pcm16k, now, duration)
            except Exception:
                log.exception("Failed to enqueue audio for local ASR (user=%s)", user_id)

        speech_segments = us.vad.push(pcm16k)
        loud = rms_dbfs(pcm16k)
        force_flush = bool(speech_segments)
        if not local_ready or self._is_active(user_id):
            await self._send_chunk_remote(us, chunk, force_flush=force_flush)
        if speech_segments or loud > self.energy_dbfs_threshold:
            us.last_voice_ts = now
            return

        # Silence: if we have been quiet for a moment, flush residual window
        if us.last_voice_ts and (now - us.last_voice_ts) > 0.8 and (now - us.last_flush_ts) > 0.5:
            if not local_ready or self._is_active(user_id):
                try:
                    await us.asr.flush()
                except Exception:
                    log.exception("ASR flush failed for user %s", user_id)
            us.last_flush_ts = now

    async def _consume_asr_events(self, us: UserStream) -> None:
        try:
            async for evt in us.asr.events():
                evt["text"] = self._normalize_transcript_text(evt.get("text"))
                evt["source"] = "remote"
                log.debug("ASR event: %s", evt)
                if not self.router:
                    log.warning("No router available; dropping ASR event")
                    continue
                routed = self.router.route(evt)
                log.debug("Routed: %s", routed)
                await self._handle_routed_event(evt, routed)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("ASR consumer failed for user %s", us.user_id)

    async def _consume_local_transcripts(self) -> None:
        if not self.local_asr:
            return
        try:
            async for evt in self.local_asr.events():
                evt.setdefault("user_id", 0)
                evt["text"] = self._normalize_transcript_text(evt.get("text"))
                evt["source"] = "local"
                if not self.router:
                    continue
                routed = self.router.route(evt)
                await self._handle_routed_event(evt, routed)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Local ASR consumer failed")

    async def _handle_routed_event(self, evt: Dict[str, Any], routed: Dict[str, Any]) -> None:
        user_id = int(routed.get("speaker", evt.get("user_id", 0)))
        if routed.get("activate_window") or routed.get("wake_detected"):
            await self._activate_user(user_id)
        if evt.get("source") == "local":
            # Local transcripts prime activation but do not trigger replies directly.
            return
        if not routed.get("needs_reply"):
            return
        await self._dispatch_reply(routed)

    async def _dispatch_reply(self, routed: Dict[str, Any]) -> None:
        speaker = routed.get("speaker")
        text = self._normalize_transcript_text(routed.get("text"))
        if not self.router:
            return
        character = self.router.character_for(speaker)
        if not self.ark:
            log.warning("No ARK agent available; cannot generate reply")
            await self._send_text_fallback(text)
            return
        context_events = self.router.context()
        short_term = self._collect_short_term_context()
        memory_context: List[Dict[str, Any]] = []
        if self._should_query_memory(text):
            memory_context = await self._collect_memory_context(text, speaker)
        additional = self._dedupe_context(short_term + memory_context)
        reply = await self.ark.reply(context_events, text, character, additional_context=additional)
        log.info("ARK reply: %s", reply)
        if self.tts and self.voice and self.voice.is_connected():
            log.debug("Using TTS for reply to user %s", speaker)
            try:
                path = await self.tts.speak_to_tempfile(reply)
                await self._play_file(path)
                return
            except Exception:
                log.exception("TTS failed; falling back to text message")
        await self._send_text_fallback(reply)

    async def _activate_user(self, user_id: int) -> None:
        now = time.time()
        expires = now + float(self._active_window_duration)
        previous = self._activation.get(int(user_id))
        self._activation[int(user_id)] = expires
        if not previous or previous < now:
            log.info("Activated ARK window for user %s (%.1fs)", user_id, self._active_window_duration)
        await self._replay_buffer_to_remote(user_id)
        try:
            us = await self._ensure_user_stream(user_id)
            await us.asr.flush()
        except Exception:
            log.debug("Flush post-activation failed for user %s", user_id)

    async def _replay_buffer_to_remote(self, user_id: int) -> None:
        buffer = self._audio_buffers.get(int(user_id))
        if not buffer:
            return
        us = await self._ensure_user_stream(user_id)
        pending = [chunk for chunk in buffer if not chunk.sent]
        if not pending:
            return
        for idx, chunk in enumerate(pending):
            await self._send_chunk_remote(us, chunk, force_flush=(idx == len(pending) - 1))

    async def _send_chunk_remote(self, us: UserStream, chunk: AudioChunk, force_flush: bool) -> None:
        if chunk.sent:
            return
        try:
            await us.asr.send_audio(chunk.pcm, force_flush=force_flush)
            chunk.sent = True
        except Exception:
            log.exception("Failed sending audio to ASR for user %s", us.user_id)

    def _prune_audio_buffer(self, user_id: int) -> None:
        buffer = self._audio_buffers.setdefault(int(user_id), deque())
        if not buffer:
            return
        cutoff = time.time() - float(self._local_context_seconds)
        while buffer and (buffer[0].timestamp + buffer[0].duration) < cutoff:
            buffer.popleft()

    def _is_active(self, user_id: int) -> bool:
        expires = self._activation.get(int(user_id))
        if not expires:
            return False
        now = time.time()
        if expires <= now:
            self._activation.pop(int(user_id), None)
            return False
        return True

    async def _send_text_fallback(self, text: str) -> None:
        # Try to post to a text channel the bot can see (voice channel's guild system channel)
        try:
            if self.voice.channel and getattr(self.voice.channel, "guild", None):
                guild = self.voice.channel.guild
                channel = guild.system_channel or next((c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None)
                if channel:
                    await channel.send(text)
        except Exception:
            log.debug("Failed to send text fallback")

    async def _play_file(self, path: Path) -> None:
        async with self._player_lock:
            # Wait until nothing is playing
            while self.voice.is_playing() or self.voice.is_paused():
                await asyncio.sleep(0.1)
            # Use FFmpegPCMAudio to play arbitrary formats
            src = discord.FFmpegPCMAudio(str(path))
            self.voice.play(src)
            # Wait for playback to finish
            while self.voice.is_playing() or self.voice.is_paused():
                await asyncio.sleep(0.1)
        with contextlib.suppress(Exception):
            Path(path).unlink(missing_ok=True)


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Extract PCM16 from a wav-like payload.
    Fallback heuristics are applied if the header is malformed but the payload
    looks like PCM16LE at 48kHz (common for Discord sinks).
    """
    try:
        with contextlib.ExitStack() as stack:
            bio = stack.enter_context(io.BytesIO(wav_bytes))
            wf = stack.enter_context(wave.open(bio, "rb"))
            ch = wf.getnchannels()
            sr = wf.getframerate()
            n = wf.getnframes()
            pcm = wf.readframes(n)
            if not pcm and len(wav_bytes) > 100:
                raise ValueError("empty frames from wave; fallback")
            # If stereo, take mono average
            if ch == 2:
                import numpy as np

                arr = np.frombuffer(pcm, dtype=np.int16)
                left = arr[0::2]
                right = arr[1::2]
                mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
                pcm = mono.tobytes()
                ch = 1
            return pcm, sr, ch
    except Exception:
        # Heuristic fallback: assume little-endian PCM16 stereo 48kHz with 44-byte header
        try:
            if len(wav_bytes) > 100 and wav_bytes[:4] == b"RIFF":
                payload = wav_bytes[44:]
                # Average stereo to mono
                import numpy as np

                arr = np.frombuffer(payload, dtype=np.int16)
                if arr.size == 0:
                    return b"", 48000, 1
                if arr.size % 2 == 0:
                    left = arr[0::2]
                    right = arr[1::2]
                    mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
                else:
                    mono = arr
                return mono.tobytes(), 48000, 1
        except Exception:
            pass
        return b"", 48000, 1
