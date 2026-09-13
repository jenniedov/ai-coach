"""NVIDIA Parakeet TDT 0.6B v3 on MLX (parakeet-mlx). Fast, accurate English STT on Apple Silicon."""
from __future__ import annotations

import logging

import numpy as np

from .base import STTProvider

log = logging.getLogger(__name__)


class ParakeetSTT(STTProvider):
    name = "parakeet-tdt-0.6b-v3"

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx = mx
        self._get_logmel = get_logmel
        self.model = from_pretrained(model_id)
        self.name = model_id.split("/")[-1]

    def transcribe(self, audio16k: np.ndarray) -> str:
        if audio16k.size < 1600:  # <0.1 s
            return ""
        x = self._mx.array(audio16k.astype(np.float32))
        mel = self._get_logmel(x, self.model.preprocessor_config)
        result = self.model.generate(mel)[0]
        return (result.text or "").strip()
