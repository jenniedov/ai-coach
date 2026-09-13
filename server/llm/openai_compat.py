"""Generic OpenAI-compatible chat-completions streaming client (httpx, no SDK dependency).
Used for xAI Grok (https://api.x.ai/v1), OpenAI, Ollama, LM Studio, etc."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .base import LLMProvider

log = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, base_url: str, api_key: str | None, models: list[str],
                 key_env_name: str = "API_KEY", extra_body: dict | None = None,
                 per_model: dict[str, dict] | None = None, use_max_completion_tokens: bool = False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = models
        self.key_env_name = key_env_name
        self.extra_body = extra_body or {}
        self.per_model = per_model or {}
        self.use_max_completion_tokens = use_max_completion_tokens
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), http2=False)

    def available_models(self) -> list[str]:
        return self.models

    def discover_models(self, preferred: list[str], include_pattern=None, exclude_pattern=None) -> list[str]:
        """Ask the provider which models this account can use; keep `preferred` order first.
        Falls back to the static list on any failure."""
        if not self.api_key:
            return self.models
        try:
            r = httpx.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=8)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception as e:  # noqa: BLE001
            log.warning("%s: could not list models (%s); using static list", self.name, e)
            return self.models
        if exclude_pattern:
            ids = [i for i in ids if not exclude_pattern.search(i)]
        if include_pattern:
            ids = [i for i in ids if include_pattern.search(i)]
        ordered = [m for m in preferred if m in ids] + sorted(i for i in ids if i not in preferred)
        if ordered:
            self.models = ordered
            log.info("%s models available: %s", self.name, ordered)
        return self.models

    def is_configured(self) -> bool:
        return bool(self.api_key) or self.name in ("ollama", "lmstudio")

    def missing_config_message(self) -> str:
        return (f"{self.key_env_name} is not set. Add it to .env (see .env.example) and restart. "
                f"The coach cannot answer without an LLM.")

    async def stream(self, messages, model, *, max_tokens=400, temperature=0.7) -> AsyncIterator[str]:
        if not self.is_configured():
            raise RuntimeError(self.missing_config_message())
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        overrides = dict(self.per_model.get(model, {}))
        real_model = overrides.pop("model", model)
        body = {
            "model": real_model,
            "messages": messages,
            "stream": True,
            ("max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"): max_tokens,
            "temperature": temperature,
            **self.extra_body,
            **overrides,
        }
        async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                       headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"{self.name} API error {resp.status_code}: {text[:500]}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    async def aclose(self):
        await self._client.aclose()
