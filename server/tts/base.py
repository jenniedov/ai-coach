from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class VoiceInfo:
    id: str
    name: str
    engine: str
    kind: str = "builtin"         # builtin | custom
    consent: str = "n/a"          # n/a | user_attested
    description: str = ""


class TTSProvider(ABC):
    """Provider interface. All engines return float32 mono PCM at self.sample_rate."""
    name: str = "base"
    sample_rate: int = 24000
    supports_cloning: bool = False

    @abstractmethod
    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray: ...

    def stream(self, text: str, voice: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        """Default: one chunk. Engines with internal streaming override this."""
        yield self.generate(text, voice, speed)

    def stop(self) -> None:
        """Best-effort cancellation hook (the pipeline also cancels at chunk boundaries)."""
        self._stop = True

    @abstractmethod
    def voices(self) -> list[VoiceInfo]: ...

    def warmup(self, voice: str) -> None:
        try:
            self.generate("Hello there.", voice)
        except Exception:  # noqa: BLE001
            pass
