from src.router import Router


def make_router(default_silent=True, wake_word="арк") -> Router:
    campaign = {
        "ark": {
            "silence_by_default": default_silent,
            "wake_phrase": wake_word,
            "activation_timeout": 12,
        },
        "asr": {
            "kws_enabled": True,
            "kws_wake_phrase": "арк",
            "confidence_threshold": 0.5,
            "active_window_duration": 12,
            "context_window": 30,
        },
        "triggers": {
            "ALERT": {"phrases": ["тревога", "опасность"], "require_wake_phrase": True}
        },
    }
    mapping = {"characters": {123: "Оператор"}}
    return Router(campaign, mapping, context_max_turns=4)


def test_trigger_phrase_requires_wakeword():
    r = make_router(default_silent=True, wake_word="ARK")
    # Without wakeword, should not trigger
    out = r.route({"user_id": 123, "text": "Тревога, всем приготовиться!", "final": True, "ts": 0.0, "source": "remote"})
    assert out["needs_reply"] is False
    # With wakeword, should trigger
    out2 = r.route({"user_id": 123, "text": "ARK тревога немедленно", "final": True, "ts": 0.0, "source": "remote"})
    assert out2["needs_reply"] is True
    assert out2["codex_task"]["name"] == "ALERT"


def test_wake_word_needed_in_silent_mode():
    r = make_router(default_silent=True)
    out = r.route({"user_id": 123, "text": "Просто говорю", "final": True, "ts": 0.0, "source": "remote"})
    assert out["needs_reply"] is False
    out2 = r.route({"user_id": 123, "text": "ARK, нужна помощь", "final": True, "ts": 0.0, "source": "remote"})
    assert out2["needs_reply"] is True


def test_active_mode_replies_on_finals():
    r = make_router(default_silent=False)
    out = r.route({"user_id": 123, "text": "Проверка", "final": True, "ts": 0.0, "source": "remote"})
    assert out["needs_reply"] is True


def test_intermediate_not_replied():
    r = make_router(default_silent=False)
    out = r.route({"user_id": 123, "text": "Промежуточный", "final": False, "ts": 0.0, "source": "remote"})
    assert out["needs_reply"] is False


def test_kws_across_chunk_boundary_triggers():
    r = make_router(default_silent=True, wake_word="скальдрон")
    first = r.route({
        "user_id": 123,
        "text": "Скал",
        "final": False,
        "confidence": None,
        "ts": 0.0,
        "source": "local",
    })
    assert first["needs_reply"] is False
    second = r.route({
        "user_id": 123,
        "text": "ьдрон, как слышно",
        "final": True,
        "confidence": 0.82,
        "ts": 0.5,
        "source": "local",
    })
    assert second["wake_detected"] is True
    assert second["needs_reply"] is True
    assert second["text"].startswith("как слышно")


def test_ark_confidence_threshold_applied():
    r_high = make_router(default_silent=True, wake_word="арк")
    high_conf = r_high.route({
        "user_id": 321,
        "text": "арк активируй протокол",
        "final": True,
        "confidence": 0.7,
        "ts": 0.0,
        "source": "remote",
    })
    assert high_conf["needs_reply"] is True
    assert high_conf["wake_detected"] is True

    r_low = make_router(default_silent=True, wake_word="арк")
    low_conf = r_low.route({
        "user_id": 321,
        "text": "арк повторяю",
        "final": True,
        "confidence": 0.2,
        "ts": 1.0,
        "source": "remote",
    })
    assert low_conf["needs_reply"] is False
    assert low_conf["wake_detected"] is False


def test_local_wake_word_activates_remote_window():
    r = make_router(default_silent=True, wake_word="арк")
    local_evt = r.route({
        "user_id": 777,
        "text": "арк включай защиту",
        "final": True,
        "confidence": 0.9,
        "ts": 0.0,
        "source": "local",
    })
    assert local_evt["activate_window"] is True
    assert local_evt["needs_reply"] is True

    remote_evt = r.route({
        "user_id": 777,
        "text": "Противник на шесть часов",
        "final": True,
        "confidence": 0.91,
        "ts": 0.5,
        "source": "remote",
    })
    assert remote_evt["needs_reply"] is True
    assert remote_evt["speaker"] == 777


def test_remote_speech_without_wakeword_remains_passive():
    r = make_router(default_silent=True, wake_word="арк")
    remote_evt = r.route({
        "user_id": 999,
        "text": "Просто комментарий без активации",
        "final": True,
        "confidence": 0.88,
        "ts": 0.0,
        "source": "remote",
    })
    assert remote_evt["needs_reply"] is False


def test_character_mapping_for_multiple_speakers():
    r = make_router(default_silent=False, wake_word="арк")
    r.set_character(111, "Оператор")
    r.set_character(222, "Пилот")

    evt_one = r.route({
        "user_id": 111,
        "text": "арк запрос статуса",
        "final": True,
        "confidence": 0.75,
        "ts": 0.0,
        "source": "local",
    })
    assert evt_one["activate_window"] is True
    assert r.character_for(111) == "Оператор"

    evt_two = r.route({
        "user_id": 222,
        "text": "Пилот докладывает, всё чисто",
        "final": True,
        "confidence": 0.9,
        "ts": 1.0,
        "source": "remote",
    })
    assert evt_two["speaker"] == 222
    assert r.character_for(222) == "Пилот"
