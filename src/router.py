import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import yaml


class Router:
    """
    Routes ASR events to ARK agent decisions.

    Input (ASR): {"user_id": int, "text": str, "final": bool, "ts": float}
    Output: {"speaker": int, "text": str, "needs_reply": bool, "codex_task": dict|None}
    """

    def __init__(
        self,
        campaign_cfg: Dict[str, Any],
        mapping_cfg: Dict[str, Any],
        context_max_turns: int = 12,
    ) -> None:
        self.campaign = campaign_cfg or {}
        self.mapping = mapping_cfg or {"characters": {}}
        ark_cfg = self.campaign.get("ark", {}) or {}
        asr_cfg: Dict[str, Any] = dict(self.campaign.get("asr", {}) or {})

        self.ark_enabled: bool = True  # runtime toggle; default on
        self.default_silent: bool = bool(
            ark_cfg.get("silence_by_default", self.campaign.get("rules", {}).get("default_silent", True))
        )
        self.wake_word: str = str(
            ark_cfg.get("wake_phrase")
            or self.campaign.get("wake_word")
            or self.campaign.get("rules", {}).get("wake_word")
            or "ARK"
        )
        self.triggers: Dict[str, Any] = self.campaign.get("triggers", {})
        activation_default = asr_cfg.get("active_window_duration", ark_cfg.get("activation_timeout", 30))
        try:
            self.activation_timeout = float(activation_default or 30)
        except Exception:
            self.activation_timeout = 30.0
        env_timeout = os.environ.get("ARK_ACTIVATION_TIMEOUT")
        if env_timeout:
            try:
                self.activation_timeout = float(env_timeout)
            except ValueError:
                pass
        try:
            self.local_context_seconds = float(asr_cfg.get("local_context_window") or asr_cfg.get("context_window") or 30)
        except Exception:
            self.local_context_seconds = 30.0
        try:
            self.api_context_window = float(asr_cfg.get("api_context_window") or asr_cfg.get("context_window") or 30)
        except Exception:
            self.api_context_window = 30.0
        local_env = os.environ.get("ASR_LOCAL_CONTEXT_WINDOW")
        if local_env:
            try:
                self.local_context_seconds = float(local_env)
            except ValueError:
                pass
        api_env = os.environ.get("ASR_CONTEXT_WINDOW")
        if api_env:
            try:
                self.api_context_window = float(api_env)
            except ValueError:
                pass
        dialogue_cfg: Dict[str, Any] = dict(self.campaign.get("dialogue", {}) or {})
        ark_dialogue = ark_cfg.get("dialogue")
        if isinstance(ark_dialogue, dict):
            dialogue_cfg.update(ark_dialogue)
        self.dialogue_keywords: List[str] = [str(kw).lower() for kw in dialogue_cfg.get("keywords", []) if isinstance(kw, str)]
        try:
            self.dialogue_min_words = int(dialogue_cfg.get("min_words", 0) or 0)
        except Exception:
            self.dialogue_min_words = 0
        self.dialogue_require_question: bool = bool(dialogue_cfg.get("require_question", False))
        self.dialogue_allow_plain: bool = bool(dialogue_cfg.get("allow_questions_without_keywords", False))
        try:
            self.asr_confidence_threshold = float(asr_cfg.get("confidence_threshold", 0.5))
        except Exception:
            self.asr_confidence_threshold = 0.5
        env_conf = os.environ.get("ASR_CONFIDENCE_THRESHOLD")
        if env_conf:
            try:
                self.asr_confidence_threshold = float(env_conf)
            except ValueError:
                pass
        self.asr_kws_enabled: bool = bool(asr_cfg.get("kws_enabled", False))
        kws_env = os.environ.get("ASR_KWS_ENABLED")
        if kws_env is not None:
            self.asr_kws_enabled = str(kws_env).lower() in ("1", "true", "yes", "on")
        self.asr_kws_wake_phrase: str = str(asr_cfg.get("kws_wake_phrase") or "").strip() or self.wake_word
        kws_phrase_env = os.environ.get("ASR_KWS_WAKE_PHRASE")
        if kws_phrase_env:
            self.asr_kws_wake_phrase = str(kws_phrase_env).strip()

        self._context: Deque[Dict[str, Any]] = deque(maxlen=context_max_turns)
        self._wake_active: bool = False
        self._wake_expiry: float = 0.0
        self._stream_buffers: Dict[int, str] = {}
        self._stream_max_chars: int = max(160, int(self.local_context_seconds * 20))

    @classmethod
    def from_files(
        cls,
        campaign_path: str,
        mapping_path: str,
        context_max_turns: int = 12,
    ) -> "Router":
        with open(campaign_path, "r", encoding="utf-8") as f:
            campaign_cfg = yaml.safe_load(f) or {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_cfg = yaml.safe_load(f) or {}
        return cls(campaign_cfg, mapping_cfg, context_max_turns)

    def set_wake_word(self, word: str) -> None:
        self.wake_word = word.strip()
        self._stream_buffers.clear()

    def set_ark_enabled(self, enabled: bool) -> None:
        self.ark_enabled = enabled
        if not enabled:
            self._wake_active = False
            self._wake_expiry = 0.0
            self._stream_buffers.clear()

    def set_character(self, user_id: int, character: str) -> None:
        self.mapping.setdefault("characters", {})[int(user_id)] = character

    def character_for(self, user_id: int) -> Optional[str]:
        chars = self.mapping.get("characters") if isinstance(self.mapping, dict) else None
        if not isinstance(chars, dict):
            return None
        return chars.get(int(user_id))

    def _has_wake_word(self, text: str) -> bool:
        if self._phrase_in_text(text, self.wake_word):
            return True
        if self.asr_kws_wake_phrase and self._phrase_in_text(text, self.asr_kws_wake_phrase):
            return True
        return False

    def _strip_wake_word(self, text: str) -> str:
        return self._strip_detected_phrases(text, None)

    def _strip_detected_phrases(self, text: str, triggered_phrase: Optional[str]) -> str:
        phrases: List[str] = []
        if triggered_phrase:
            phrases.append(triggered_phrase)
        if self.wake_word and (not triggered_phrase or triggered_phrase.lower() != str(self.wake_word).lower()):
            phrases.append(self.wake_word)
        if self.asr_kws_wake_phrase and (
            not triggered_phrase or triggered_phrase.lower() != str(self.asr_kws_wake_phrase).lower()
        ):
            phrases.append(self.asr_kws_wake_phrase)
        cleaned = text
        for phrase in phrases:
            cleaned = self._strip_phrase(cleaned, phrase)
        if triggered_phrase:
            cleaned = self._strip_partial_prefix(cleaned, triggered_phrase)
        return cleaned.strip() or text

    @staticmethod
    def _normalize_compact(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", text.lower())

    def _phrase_in_text(self, text: str, phrase: Optional[str]) -> bool:
        if not phrase:
            return False
        normalized = self._normalize_compact(text)
        target = self._normalize_compact(phrase)
        if target and target in normalized:
            return True
        pattern = rf"\b{re.escape(str(phrase))}\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _strip_phrase(self, text: str, phrase: Optional[str]) -> str:
        if not phrase:
            return text
        pattern = rf"\b{re.escape(str(phrase))}\b"
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if cleaned == text:
            collapsed = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
            target = re.sub(r"[\s\W_]+", "", str(phrase), flags=re.UNICODE)
            if target and target.lower() in collapsed.lower():
                spaced_pattern = r"[\s\W_]*".join(map(re.escape, str(phrase)))
                cleaned = re.sub(spaced_pattern, "", text, flags=re.IGNORECASE).strip()
        return cleaned

    def _strip_partial_prefix(self, text: str, phrase: str) -> str:
        if not text or not phrase:
            return text
        stripped = text.lstrip()
        leading = text[: len(text) - len(stripped)]
        lower_phrase = phrase.lower()
        lower_stripped = stripped.lower()
        for length in range(len(lower_phrase), 1, -1):
            suffix = lower_phrase[-length:]
            if lower_stripped.startswith(suffix):
                return (leading + stripped[length:]).lstrip()
        return text

    def _confidence_ok(self, confidence: Optional[float]) -> bool:
        if confidence is None:
            return True
        try:
            return float(confidence) >= float(self.asr_confidence_threshold or 0.0)
        except (TypeError, ValueError):
            return False

    def _evaluate_wake_detection(
        self,
        user_id: int,
        text: str,
        confidence: Optional[float],
        asr_event: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        flagged = bool(asr_event.get("wake_detected"))
        if flagged:
            phrase = str(asr_event.get("wake_phrase") or "").strip() or self.wake_word
            self._stream_buffers.pop(int(user_id), None)
            return True, phrase

        if not self.asr_kws_enabled:
            detected = self._has_wake_word(text)
            return detected, self.wake_word if detected else None

        buffer = self._stream_buffers.get(int(user_id), "")
        combined = f"{buffer} {text}".strip() if text else buffer
        detection_phrase: Optional[str] = None
        if combined:
            if self._phrase_in_text(combined, self.wake_word):
                detection_phrase = self.wake_word
            elif self.asr_kws_wake_phrase and self._phrase_in_text(combined, self.asr_kws_wake_phrase):
                if self._confidence_ok(confidence):
                    detection_phrase = self.asr_kws_wake_phrase
        if detection_phrase:
            self._stream_buffers.pop(int(user_id), None)
            return True, detection_phrase

        if combined:
            limited = combined[-self._stream_max_chars :]
            if asr_event.get("final"):
                limited = limited[-(self._stream_max_chars // 2) :]
            self._stream_buffers[int(user_id)] = limited
        return False, None

    def _should_auto_reply(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if not self.dialogue_keywords and not self.dialogue_require_question and not self.dialogue_min_words:
            return True
        words = stripped.split()
        if self.dialogue_min_words and len(words) < self.dialogue_min_words:
            return False
        lowered = stripped.lower()
        keyword_hit = any(kw in lowered for kw in self.dialogue_keywords) if self.dialogue_keywords else False
        question_hit = '?' in stripped
        if self.dialogue_require_question and not question_hit and not keyword_hit:
            return False
        if self.dialogue_keywords and not keyword_hit:
            if not (self.dialogue_allow_plain and question_hit):
                return False
        if not self.dialogue_keywords and self.dialogue_require_question:
            return question_hit
        return True

    def _check_triggers(self, text: str) -> Optional[Dict[str, Any]]:
        # Support both legacy !TRIGGER tokens and new phrase-based triggers
        upper_text = text.upper()
        lower_text = text.lower()
        for trig, meta in (self.triggers or {}).items():
            # Legacy bang-trigger detection
            token = f"!{trig.upper()}"
            if token in upper_text:
                return {"type": "trigger", "name": trig, "meta": meta}

            # Phrase-based detection
            phrases = meta.get("phrases") if isinstance(meta, dict) else None
            if phrases:
                if bool(meta.get("require_wake_phrase")) and not self._has_wake_word(text):
                    continue
                for phrase in phrases:
                    ph = str(phrase).strip().lower()
                    if not ph:
                        continue
                    # word-boundary match if possible; fallback to substring
                    pattern = rf"\b{re.escape(ph)}\b"
                    if re.search(pattern, lower_text, flags=re.IGNORECASE) or ph in lower_text:
                        return {"type": "trigger", "name": trig, "meta": meta}
        return None

    def _add_context(self, speaker: int, text: str) -> None:
        self._context.append({
            "speaker": int(speaker),
            "text": text,
            "ts": time.time(),
        })

    def context(self) -> list[Dict[str, Any]]:
        return list(self._context)

    def route(self, asr_event: Dict[str, Any]) -> Dict[str, Any]:
        user_id = int(asr_event.get("user_id", 0))
        text = (asr_event.get("text") or "").strip()
        is_final = bool(asr_event.get("final", False))
        now = time.time()
        if self._wake_active and now > self._wake_expiry:
            self._wake_active = False
            self._wake_expiry = 0.0

        source = str(asr_event.get("source") or "remote")

        result: Dict[str, Any] = {
            "speaker": user_id,
            "text": text,
            "needs_reply": False,
            "codex_task": None,
            "wake_detected": False,
            "activate_window": False,
            "source": source,
        }

        if not text:
            return result

        # Confidence gating for final utterances
        conf_raw = asr_event.get("confidence")
        try:
            conf_val = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            conf_val = None
        if is_final and conf_val is not None and self.asr_confidence_threshold > 0.0:
            if conf_val < float(self.asr_confidence_threshold):
                return result

        # Record all final utterances to context (post-threshold)
        wake_detected, wake_phrase = self._evaluate_wake_detection(user_id, text, conf_val, asr_event)
        if wake_detected:
            self._wake_active = True
            self._wake_expiry = now + self.activation_timeout
            stripped = self._strip_detected_phrases(text, wake_phrase)
            text = stripped
            result["text"] = stripped
            result["wake_detected"] = True
            if wake_phrase:
                result["wake_phrase"] = wake_phrase
            result["activate_window"] = True
            if source == "local":
                result["needs_reply"] = True
        if is_final:
            self._add_context(user_id, text)

        # Triggers override silence
        trig = self._check_triggers(text)
        if trig:
            result["needs_reply"] = True
            result["codex_task"] = trig
            return result

        # Intermediate segments generally don't trigger replies
        if not is_final:
            return result

        active_now = self._wake_active and now <= self._wake_expiry
        should_auto = self._should_auto_reply(text)
        # Wake-word logic
        if self.ark_enabled:
            if not self.default_silent:
                # Active mode: reply to finals unless addressed otherwise by config
                if should_auto:
                    result["needs_reply"] = True
            else:
                # Silent-by-default: only reply on wake-word
                if (wake_detected or active_now) and should_auto:
                    result["needs_reply"] = True

        return result
