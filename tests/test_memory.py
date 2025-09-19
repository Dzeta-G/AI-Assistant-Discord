import time

import pytest

from src.asr_local import LocalASRConfig, LocalTranscriber
from src.memory import MemoryEntry, create_memory_store, is_recall_query


def test_local_transcriber_prunes_old_segments(monkeypatch):
    cfg = LocalASRConfig(model_path="", binary_path="", context_seconds=1.0)
    transcriber = LocalTranscriber(cfg)
    user_id = 42
    transcriber._append_segment(user_id, "старый", 0.0, 0.1)
    transcriber._append_segment(user_id, "новый", 1.2, 1.3)
    monkeypatch.setattr("src.asr_local.time.time", lambda: 1.3)
    transcriber._prune_buffer(transcriber._buffers[user_id])
    remaining = [seg.text for seg in transcriber._buffers[user_id]]
    assert "новый" in remaining
    assert "старый" not in remaining


@pytest.mark.asyncio
async def test_memory_store_roundtrip(tmp_path):
    store = create_memory_store("json", tmp_path.joinpath("log.json").as_posix())
    entry = MemoryEntry(timestamp=time.time(), user_id=7, speaker="Operator", text="Artifact is unstable")
    await store.log_event(entry)
    recent = await store.recent(60)
    assert any(r.text == "Artifact is unstable" for r in recent)
    results = await store.search("artifact")
    assert results and results[0].text == "Artifact is unstable"
    await store.clear()
    assert not await store.recent(60)


@pytest.mark.asyncio
async def test_recall_query_hits_memory(tmp_path):
    store = create_memory_store("json", tmp_path.joinpath("session.json").as_posix())
    await store.log_event(MemoryEntry(timestamp=time.time(), user_id=9, speaker="Malran", text="The artifact resonates at dawn"))
    query = "What did Malran say earlier about the artifact?"
    assert is_recall_query(query)
    results = await store.search("artifact", limit=3)
    assert any("artifact" in r.text for r in results)
