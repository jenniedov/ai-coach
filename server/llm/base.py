from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """Modular LLM interface. Swap Grok for anything OpenAI-compatible or custom."""

    name: str = "base"

    @abstractmethod
    async def stream(self, messages: list[dict], model: str, *,
                     max_tokens: int = 400, temperature: float = 0.7) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        yield ""  # pragma: no cover

    @abstractmethod
    def available_models(self) -> list[str]: ...

    def is_configured(self) -> bool:
        return True

    def missing_config_message(self) -> str:
        return ""
