from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTProvider(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio16k: np.ndarray) -> str:
        """audio16k: float32 mono at 16 kHz. Blocking; call from a thread executor."""

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))
