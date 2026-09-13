from __future__ import annotations

import logging
import re

from ..config import settings
from .base import LLMProvider
from .openai_compat import OpenAICompatProvider

log = logging.getLogger(__name__)

# xAI: model ids current as of 2026-09 (docs.x.ai/developers/models). grok-4.3 accepts
# reasoning_effort="none" which is xAI's recommended low-latency mode for voice.
XAI_MODELS = {
    "grok-4.3": {"reasoning_effort": "none"},
    "grok-4.20-0309-non-reasoning": {},
    "grok-4.3-reasoning-low": {"model": "grok-4.3", "reasoning_effort": "low"},
    "grok-4.5": {"reasoning_effort": "low"},
    "grok-4.6": {"reasoning_effort": "low"},
}
# Groq: gpt-oss models reason before answering; "low" keeps first-token latency small. qwen3 accepts "none".
GROQ_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b",
               "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "llama-3.1-8b-instant"]
GROQ_OVERRIDES = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "qwen/qwen3.8-27b": {"reasoning_effort": "none"},
    "qwen/qwen3.6-27b": {"reasoning_effort": "none"},
}
GROQ_EXCLUDE = re.compile(r"whisper|orpheus|prompt-guard|safeguard|tts|guard|embed", re.I)
OPENAI_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini"]


def build_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.LLM_PROVIDER or "auto").lower()
    if name == "auto":
        if settings.XAI_API_KEY:
            name = "xai"
        elif settings.GROQ_API_KEY:
            name = "groq"
        elif settings.OPENAI_API_KEY:
            name = "openai"
        else:
            name = "xai"  # unconfigured; UI will show the missing-key message
        log.info("LLM provider auto-selected: %s", name)
    if name == "xai":
        p = OpenAICompatProvider("xai", settings.XAI_BASE_URL, settings.XAI_API_KEY, list(XAI_MODELS),
                                 key_env_name="XAI_API_KEY", per_model=XAI_MODELS, use_max_completion_tokens=True)
        p.discover_models(list(XAI_MODELS), include_pattern=re.compile(r"^grok-4\.[2-9]|^grok-4\.\d\d", re.I),
                          exclude_pattern=re.compile(r"image|video|voice|vision|imagine", re.I))
        # keep our alias entries that map onto real models
        for alias, ov in XAI_MODELS.items():
            if ov.get("model") in p.models and alias not in p.models:
                p.models.append(alias)
        return p
    if name == "groq":
        p = OpenAICompatProvider("groq", settings.GROQ_BASE_URL, settings.GROQ_API_KEY, GROQ_MODELS,
                                 key_env_name="GROQ_API_KEY", per_model=GROQ_OVERRIDES, use_max_completion_tokens=True)
        p.discover_models(GROQ_MODELS, exclude_pattern=GROQ_EXCLUDE)
        return p
    if name == "openai":
        p = OpenAICompatProvider("openai", settings.OPENAI_BASE_URL, settings.OPENAI_API_KEY, OPENAI_MODELS,
                                 key_env_name="OPENAI_API_KEY", use_max_completion_tokens=True)
        p.discover_models(OPENAI_MODELS, include_pattern=re.compile(r"^gpt-(4\.1|5)", re.I),
                          exclude_pattern=re.compile(r"audio|realtime|search|transcribe|tts|image|codex", re.I))
        return p
    if name == "ollama":
        return OpenAICompatProvider("ollama", settings.OLLAMA_BASE_URL, "ollama",
                                    [settings.LLM_MODEL or "llama3.2"], key_env_name="(none)")
    raise ValueError(f"Unknown LLM_PROVIDER {name!r}")


__all__ = ["LLMProvider", "OpenAICompatProvider", "build_provider"]
