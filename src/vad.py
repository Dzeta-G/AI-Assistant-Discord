from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple

import numpy as np
import webrtcvad


def pcm16_to_np(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16)


def np_to_pcm16(arr: np.ndarray) -> bytes:
    arr16 = np.clip(arr, -32768, 32767).astype(np.int16)
    return arr16.tobytes()


def resample_mono_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    if not pcm:
        return b""
    if from_rate == to_rate:
        return pcm
    data = pcm16_to_np(pcm)
    if data.size == 0:
        return b""
    # If stereo interleaved is mistakenly supplied, take mono by averaging
    if data.ndim == 1 and (len(data) % 2 == 0) and from_rate in (44100, 48000) and to_rate in (16000, 8000):
        # Heuristic: guess stereo at common Discord SRs; average channels
        left = data[0::2]
        right = data[1::2]
        data = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    ratio = to_rate / float(from_rate)
    # Simple linear interpolation
    x_old = np.arange(len(data))
    if x_old.size == 0:
        return b""
    x_new = np.linspace(0, len(data) - 1, int(len(data) * ratio))
    resampled = np.interp(x_new, x_old, data.astype(np.float32)).astype(np.int16)
    return resampled.tobytes()


def rms_dbfs(pcm: bytes) -> float:
    """Approximate loudness in dBFS for PCM16 mono.
    Returns -inf for empty or silent buffers.
    """
    if not pcm:
        return float("-inf")
    data = pcm16_to_np(pcm).astype(np.float32)
    if data.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms <= 0.0001:
        return float("-inf")
    # Full-scale for int16 is 32768
    dbfs = 20.0 * float(np.log10(rms / 32768.0))
    return dbfs


def frame_generator(pcm: bytes, sample_rate: int, frame_ms: int = 20) -> Iterable[bytes]:
    bytes_per_sample = 2  # 16-bit
    frame_size = int(sample_rate * (frame_ms / 1000.0)) * bytes_per_sample
    for i in range(0, len(pcm), frame_size):
        frame = pcm[i : i + frame_size]
        if len(frame) == frame_size:
            yield frame


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2  # 0-3
    # How many frames of padding to determine start/stop
    padding_ms: int = 300


class VoiceActivityDetector:
    def __init__(self, cfg: VADConfig | None = None) -> None:
        self.cfg = cfg or VADConfig()
        self.vad = webrtcvad.Vad(self.cfg.aggressiveness)

        self._samples_per_frame = int(self.cfg.sample_rate * (self.cfg.frame_ms / 1000.0))
        self._ring_size = int(self.cfg.padding_ms / self.cfg.frame_ms)
        self._ring: Deque[Tuple[bytes, bool]] = collections.deque(maxlen=self._ring_size)
        self._triggered = False
        self._segment: List[bytes] = []

    def push(self, pcm16_mono_16k: bytes) -> List[bytes]:
        """
        Push raw PCM16 mono @ 16kHz. Returns a list of finalized voiced segments (bytes).
        """
        segments: List[bytes] = []
        for frame in frame_generator(pcm16_mono_16k, self.cfg.sample_rate, self.cfg.frame_ms):
            is_speech = self.vad.is_speech(frame, self.cfg.sample_rate)
            if not self._triggered:
                self._ring.append((frame, is_speech))
                num_voiced = len([1 for (_, s) in self._ring if s])
                if num_voiced > 0.9 * self._ring.maxlen:
                    # Start segment: flush ring
                    self._triggered = True
                    self._segment.extend([f for (f, _) in self._ring])
                    self._ring.clear()
            else:
                self._segment.append(frame)
                self._ring.append((frame, is_speech))
                num_unvoiced = len([1 for (_, s) in self._ring if not s])
                if num_unvoiced > 0.9 * self._ring.maxlen:
                    # End segment
                    segments.append(b"".join(self._segment))
                    self._segment.clear()
                    self._ring.clear()
                    self._triggered = False
        return segments
