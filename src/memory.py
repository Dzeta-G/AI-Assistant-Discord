from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MemoryEntry:
    timestamp: float
    user_id: int
    speaker: str
    text: str


class MemoryStore:
    async def log_event(self, entry: MemoryEntry) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def recent(self, seconds: float, now: Optional[float] = None, user_id: Optional[int] = None) -> List[MemoryEntry]:  # pragma: no cover
        raise NotImplementedError

    async def search(self, query: str, limit: int = 10, user_id: Optional[int] = None) -> List[MemoryEntry]:  # pragma: no cover
        raise NotImplementedError

    async def clear(self) -> None:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        pass


class JsonMemoryStore(MemoryStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8")) or []
            except Exception:
                self._entries = []
        self._lock = asyncio.Lock()

    async def log_event(self, entry: MemoryEntry) -> None:
        async with self._lock:
            self._entries.append(entry.__dict__)
            await asyncio.to_thread(self._flush)

    async def recent(self, seconds: float, now: Optional[float] = None, user_id: Optional[int] = None) -> List[MemoryEntry]:
        ref = now or time.time()
        cutoff = ref - float(seconds)
        async with self._lock:
            data = [
                MemoryEntry(**e)
                for e in self._entries
                if e.get("timestamp", 0) >= cutoff and (user_id is None or int(e.get("user_id", 0)) == int(user_id))
            ]
        return data[-200:]

    async def search(self, query: str, limit: int = 10, user_id: Optional[int] = None) -> List[MemoryEntry]:
        q = query.lower()
        async with self._lock:
            results: List[MemoryEntry] = []
            for raw in reversed(self._entries):
                if user_id is not None and int(raw.get("user_id", 0)) != int(user_id):
                    continue
                text = str(raw.get("text", ""))
                speaker = str(raw.get("speaker", ""))
                if q in text.lower() or q in speaker.lower():
                    results.append(MemoryEntry(**raw))
                if len(results) >= limit:
                    break
        results.reverse()
        return results

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            await asyncio.to_thread(self._flush)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


class SqliteMemoryStore(MemoryStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL NOT NULL,"
            "user_id INTEGER NOT NULL,"
            "speaker TEXT NOT NULL,"
            "text TEXT NOT NULL"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_user ON logs(user_id)")
        self._conn.commit()
        self._lock = asyncio.Lock()
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(text, speaker, content='logs', content_rowid='id')"
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False

    async def log_event(self, entry: MemoryEntry) -> None:
        async with self._lock:
            def _insert() -> None:
                cursor = self._conn.execute(
                    "INSERT INTO logs (timestamp, user_id, speaker, text) VALUES (?, ?, ?, ?)",
                    (entry.timestamp, entry.user_id, entry.speaker, entry.text),
                )
                rowid = cursor.lastrowid
                if self._fts_available:
                    self._conn.execute(
                        "INSERT INTO logs_fts(rowid, text, speaker) VALUES (?, ?, ?)",
                        (rowid, entry.text, entry.speaker),
                    )

            await asyncio.to_thread(_insert)
            await asyncio.to_thread(self._conn.commit)

    async def recent(self, seconds: float, now: Optional[float] = None, user_id: Optional[int] = None) -> List[MemoryEntry]:
        ref = now or time.time()
        cutoff = ref - float(seconds)
        query = "SELECT timestamp, user_id, speaker, text FROM logs WHERE timestamp >= ?"
        params: List[Any] = [cutoff]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(int(user_id))
        query += " ORDER BY timestamp DESC LIMIT 200"
        rows = await asyncio.to_thread(self._conn.execute, query, params)
        fetched = await asyncio.to_thread(rows.fetchall)
        entries = [MemoryEntry(*row) for row in fetched]
        entries.reverse()
        return entries

    async def search(self, query: str, limit: int = 10, user_id: Optional[int] = None) -> List[MemoryEntry]:
        if self._fts_available:
            match = query.replace("\"", " ")
            sql = (
                "SELECT timestamp, user_id, speaker, text FROM logs "
                "JOIN logs_fts ON logs_fts.rowid = logs.id "
                "WHERE logs_fts MATCH ?"
            )
            params: List[Any] = [match]
            if user_id is not None:
                sql += " AND user_id = ?"
                params.append(int(user_id))
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(int(limit))
            rows = await asyncio.to_thread(self._conn.execute, sql, params)
        else:
            like = f"%{query.lower()}%"
            sql = "SELECT timestamp, user_id, speaker, text FROM logs WHERE lower(text) LIKE ? OR lower(speaker) LIKE ?"
            params = [like, like]
            if user_id is not None:
                sql += " AND user_id = ?"
                params.append(int(user_id))
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(int(limit))
            rows = await asyncio.to_thread(self._conn.execute, sql, params)
        fetched = await asyncio.to_thread(rows.fetchall)
        entries = [MemoryEntry(*row) for row in fetched]
        entries.reverse()
        return entries

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.execute, "DELETE FROM logs")
            if self._fts_available:
                await asyncio.to_thread(self._conn.execute, "DELETE FROM logs_fts")
            await asyncio.to_thread(self._conn.commit)

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)


def create_memory_store(backend: str, file_path: str) -> MemoryStore:
    backend = (backend or "json").strip().lower()
    path = Path(file_path).expanduser()
    if backend == "sqlite":
        return SqliteMemoryStore(path)
    if backend == "json":
        return JsonMemoryStore(path)
    raise ValueError(f"Unsupported memory backend: {backend}")


RECALL_KEYWORDS = (
    "remember",
    "recall",
    "earlier",
    "before",
    "previous",
    "last time",
    "remind",
    "what did",
)


def is_recall_query(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in RECALL_KEYWORDS)
