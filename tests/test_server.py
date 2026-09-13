"""Integration tests against a running server (start ./start.sh first). Skipped if it's not up."""
from __future__ import annotations

import os

import httpx
import pytest

BASE = os.environ.get("AI_COACH_URL", "http://127.0.0.1:3000")


def _up():
    try:
        return httpx.get(f"{BASE}/healthz", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="server not running")


def test_config():
    c = httpx.get(f"{BASE}/api/config", timeout=10).json()
    assert c["coaches"]
    assert {"kokoro", "pocket", "breeze", "macos_say"} <= {e["id"] for e in c["tts_engines"]}
    assert "AI simulation" in c["disclaimer"]


def test_retrieve():
    coach = httpx.get(f"{BASE}/api/config", timeout=10).json()["coaches"][0]["id"]
    r = httpx.get(f"{BASE}/api/retrieve", params={"coach": coach, "q": "should I hire for attitude or experience"}, timeout=10).json()
    assert r["results"]
    assert r["ms"] < 500


def test_tts_kokoro_latency():
    r = httpx.get(f"{BASE}/api/tts", params={"text": "Look, here's the thing. Make the call.", "engine": "kokoro"}, timeout=60)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert int(r.headers["X-Synthesis-Ms"]) < 1500
