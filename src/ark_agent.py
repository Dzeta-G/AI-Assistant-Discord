from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import aiohttp


class ARKAgent:
    def __init__(self, api_key: str, system_prompt: str) -> None:
        self.api_key = api_key
        self.system_prompt = system_prompt

    async def reply(
        self,
        context_events: List[Dict[str, Any]],
        user_text: str,
        character: str | None = None,
        additional_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Compose messages
        messages = [{"role": "system", "content": self.system_prompt}]
        for ev in context_events[-8:]:
            who = f"{ev.get('speaker')}"
            messages.append({"role": "user", "content": f"[{who}] {ev.get('text')}"})
        if additional_context:
            for ev in additional_context:
                who = ev.get("speaker") or "context"
                label = ev.get("meta") or "context"
                messages.append({"role": "user", "content": f"[{label}] {who}: {ev.get('text')}"})
        persona = f"Персонаж: {character}. " if character else ""
        messages.append({"role": "user", "content": persona + user_text})

        payload = {
            "model": os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            "messages": messages,
            "temperature": 0.4,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                jd = await resp.json()
        return jd["choices"][0]["message"]["content"].strip()
