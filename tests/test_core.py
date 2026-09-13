"""Fast tests (no models): config, coach loading, retrieval (BM25 path), LLM provider wiring, missing-key behaviour."""
from __future__ import annotations

import asyncio
import os

import pytest

from server.coach.loader import _split_claims, load_all
from server.coach.retrieval import BM25, CoachRetriever, _tok
from server.config import settings


def _first_coach():
    coaches = load_all(settings.COACHES_DIR)
    assert coaches, "no coach folders found under coaches/"
    return next(iter(coaches.values()))


def test_coach_loads():
    c = _first_coach()
    assert c.name
    assert len(c.chunks) >= 3
    assert "simulation" in c.system_prompt.lower()
    assert c.public()["disclaimer"].startswith("AI simulation")


def test_claims_tagged():
    c = _first_coach()
    tagged = [ch for ch in c.chunks if ch.evidence in ("DIRECT", "INFERRED", "SUMMARY")]
    assert len(tagged) / len(c.chunks) > 0.8


def test_split_claims():
    body = "1. [DIRECT] First claim here. (src_01)\n   continued line\n\n2. [INFERRED] Second claim. (src_02)\n"
    claims = _split_claims(body)
    assert len(claims) == 2 and "continued" in claims[0]


def test_bm25_retrieval_without_embeddings():
    c = _first_coach()
    r = CoachRetriever(c, use_embeddings=False)
    res = r.search("how do you hire people, attitude or experience?")
    assert res and any("attitude" in ch.text.lower() or "hire" in ch.text.lower() for ch, _ in res)


def test_tokenizer_drops_conversational_words():
    assert "tell" not in _tok("what would you tell me") and "hire" in _tok("should I hire")


def test_llm_missing_key_message(monkeypatch):
    from server.llm.openai_compat import OpenAICompatProvider
    p = OpenAICompatProvider("xai", "https://api.x.ai/v1", None, ["grok-4.3"], key_env_name="XAI_API_KEY")
    assert not p.is_configured()
    assert "XAI_API_KEY" in p.missing_config_message()

    async def run():
        async for _ in p.stream([{"role": "user", "content": "hi"}], "grok-4.3"):
            pass
    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        asyncio.run(run())


def test_provider_auto_select(monkeypatch):
    from server import llm as llm_mod
    monkeypatch.setattr(settings, "XAI_API_KEY", None)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
    assert llm_mod.build_provider().name == "groq"
    monkeypatch.setattr(settings, "XAI_API_KEY", "xai-test")
    assert llm_mod.build_provider().name == "xai"


def test_xai_request_body_uses_reasoning_effort_none():
    from server.llm import XAI_MODELS
    assert XAI_MODELS["grok-4.3"]["reasoning_effort"] == "none"


def test_custom_voice_requires_consent(tmp_path, monkeypatch):
    from server.tts import custom_voices
    monkeypatch.setattr(settings, "VOICES_DIR", tmp_path)
    (tmp_path / "v1").mkdir(); (tmp_path / "v1" / "reference.wav").write_bytes(b"RIFF")
    (tmp_path / "v1" / "voice.json").write_text('{"name": "no consent", "consent": ""}')
    (tmp_path / "v2").mkdir(); (tmp_path / "v2" / "reference.wav").write_bytes(b"RIFF")
    (tmp_path / "v2" / "voice.json").write_text('{"name": "ok", "consent": "my own voice"}')
    ids = [v.id for v in custom_voices.load_custom_voices()]
    assert ids == ["v2"]
