"""TTS registry. Engines are created lazily (each loads a model)."""
from __future__ import annotations

import logging
from dataclasses import asdict

from ..config import settings
from .base import TTSProvider, VoiceInfo
from .custom_voices import load_custom_voices

log = logging.getLogger(__name__)

ENGINES = {
    "breeze": ("Breeze TTS 2 — best voice clone (needs a strong Mac)", "server.tts.breeze_mlx", "BreezeMLXTTS"),
    "pocket": ("Pocket TTS — voice clone, works on any Mac", "server.tts.pocket_mlx", "PocketMLXTTS"),
    "kokoro": ("Kokoro-82M — fastest, stock voices only", "server.tts.kokoro_mlx", "KokoroMLXTTS"),
    "macos_say": ("macOS say — basic fallback", "server.tts.macos_say", "MacOSSayTTS"),
}
ENGINE_NOTES = {
    "breeze": "Best-sounding clone, but heavy: on a Mac that isn't strong enough the call gets stuck or the coach "
              "pauses for many seconds between sentences. If that happens, pick Pocket TTS instead.",
    "pocket": "Cloned voice that runs fast on any Apple Silicon Mac.",
    "kokoro": "Instant, but stock voices only (no clone).",
    "macos_say": "Built-in macOS voices; no models needed.",
}

_DEFAULT_VOICE = {"kokoro": "af_heart", "pocket": "alba", "breeze": "warm_female", "macos_say": "Samantha"}


def available_engines() -> list[dict]:
    return [{"id": k, "label": v[0], "note": ENGINE_NOTES.get(k, "")} for k, v in ENGINES.items()]


def default_voice_for(engine: str) -> str:
    if engine == settings.TTS_ENGINE and settings.TTS_VOICE:
        return settings.TTS_VOICE
    return _DEFAULT_VOICE.get(engine, "")


def build_tts(engine: str) -> TTSProvider:
    if engine not in ENGINES:
        raise ValueError(f"Unknown TTS engine {engine!r}; choose from {list(ENGINES)}")
    import importlib
    _, module, cls = ENGINES[engine]
    try:
        return getattr(importlib.import_module(module), cls)()
    except Exception as e:  # noqa: BLE001
        if engine != "macos_say":
            log.exception("TTS engine %s failed to load (%s); falling back to macos_say", engine, e)
            from .macos_say import MacOSSayTTS
            return MacOSSayTTS()
        raise


def list_all_voices() -> list[dict]:
    """Voice catalogue without loading heavy models (static lists + consent-gated custom voices)."""
    from .breeze_mlx import DESIGNED_VOICES
    from .kokoro_mlx import KOKORO_VOICES
    from .pocket_mlx import BUILTIN as POCKET_BUILTIN

    voices: list[VoiceInfo] = [VoiceInfo(id=k, name=v[0], engine="kokoro") for k, v in KOKORO_VOICES.items()]
    voices += [VoiceInfo(id=v, name=f"{v.capitalize()} (Pocket built-in)", engine="pocket") for v in POCKET_BUILTIN]
    voices += [VoiceInfo(id=k, name=f"{k.replace('_', ' ').title()} (Breeze designed)", engine="breeze", description=v)
               for k, v in DESIGNED_VOICES.items()]
    for cv in load_custom_voices():
        for eng in ("pocket", "breeze"):
            voices.append(VoiceInfo(id=f"custom:{cv.id}", name=cv.name, engine=eng, kind="custom",
                                    consent=cv.consent, description=cv.notes))
    try:
        from .macos_say import MacOSSayTTS
        voices += MacOSSayTTS().voices()
    except Exception:  # noqa: BLE001
        pass
    return [asdict(v) for v in voices]


__all__ = ["TTSProvider", "VoiceInfo", "build_tts", "available_engines", "list_all_voices", "default_voice_for"]
