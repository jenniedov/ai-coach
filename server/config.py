"""Central configuration. Everything is read from environment variables / .env.
No secrets are hard-coded anywhere in this project."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


class Settings:
    # --- paths
    ROOT = ROOT
    COACHES_DIR = Path(_env("COACHES_DIR", str(ROOT / "coaches")))
    VOICES_DIR = Path(_env("VOICES_DIR", str(ROOT / "voices")))
    MODELS_DIR = Path(_env("MODELS_DIR", str(ROOT / "models")))
    WEB_DIR = ROOT / "web"

    # --- server
    HOST = _env("HOST", "127.0.0.1")
    PORT = _env_int("PORT", 3000)

    # --- LLM (provider is modular: xai | openai | ollama | any OpenAI-compatible)
    LLM_PROVIDER = _env("LLM_PROVIDER", "auto")  # auto | xai | groq | openai | ollama
    LLM_MODEL = _env("LLM_MODEL")  # None -> provider's first model
    XAI_API_KEY = _env("XAI_API_KEY")
    XAI_BASE_URL = _env("XAI_BASE_URL", "https://api.x.ai/v1")
    GROQ_API_KEY = _env("GROQ_API_KEY")
    GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    OPENAI_API_KEY = _env("OPENAI_API_KEY")
    OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 260)
    LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.7)
    HISTORY_TURNS = _env_int("HISTORY_TURNS", 12)

    # --- STT
    STT_ENGINE = _env("STT_ENGINE", "parakeet")  # parakeet | whisper
    STT_MODEL = _env("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
    STT_FALLBACK_MODEL = _env("STT_FALLBACK_MODEL", "mlx-community/whisper-small-mlx")

    # --- TTS
    TTS_ENGINE = _env("TTS_ENGINE", "kokoro")  # kokoro | pocket | breeze | macos_say
    TTS_VOICE = _env("TTS_VOICE", "af_heart")
    TTS_SPEED = _env_float("TTS_SPEED", 1.05)

    # --- VAD / turn detection (16 kHz, 512-sample frames = 32 ms)
    VAD_THRESHOLD = _env_float("VAD_THRESHOLD", 0.6)
    VAD_MIN_VOLUME = _env_float("VAD_MIN_VOLUME", 0.3)  # pipecat default 0.6 misses quiet mics
    VAD_END_SILENCE_MS = _env_int("VAD_END_SILENCE_MS", 650)
    VAD_MIN_SPEECH_MS = _env_int("VAD_MIN_SPEECH_MS", 200)
    VAD_PRE_ROLL_MS = _env_int("VAD_PRE_ROLL_MS", 300)
    VAD_MAX_UTTERANCE_S = _env_int("VAD_MAX_UTTERANCE_S", 30)
    BARGE_IN_ENABLED = _env("BARGE_IN_ENABLED", "true").lower() in ("1", "true", "yes")
    TURN_MODE = _env("TURN_MODE", "smart")  # smart (Smart Turn v3 semantic end-of-turn) | silence
    BARGE_IN_MIN_SPEECH_MS = _env_int("BARGE_IN_MIN_SPEECH_MS", 350)
    BARGE_IN_THRESHOLD = _env_float("BARGE_IN_THRESHOLD", 0.7)

    # --- retrieval
    RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 6)
    EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    LOG_LEVEL = _env("LOG_LEVEL", "INFO")


settings = Settings()
