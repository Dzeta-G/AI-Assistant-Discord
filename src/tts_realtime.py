from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import aiohttp


class TTSClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "TTSClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def speak_to_tempfile(self, text: str, fmt: str = "mp3") -> Path:
        if not self._session:
            self._session = aiohttp.ClientSession()
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": text, "voice": self.voice, "format": fmt}
        async with self._session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            audio = await resp.read()
        fd, path = tempfile.mkstemp(prefix="ark_tts_", suffix=f".{fmt}")
        os.write(fd, audio)
        os.close(fd)
        return Path(path)

