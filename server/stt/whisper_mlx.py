"""Whisper on MLX via mlx-audio (torch-free). Fallback STT; also handles non-English."""
from __future__ import annotations

import logging

import numpy as np

from .base import STTProvider

log = logging.getLogger(__name__)


class WhisperMLXSTT(STTProvider):
    name = "whisper-mlx"

    def __init__(self, model_id: str = "mlx-community/whisper-small-mlx", language: str | None = "en"):
        from mlx_audio.stt.utils import load_model

        self.model = load_model(model_id)
        self.language = language
        self.name = model_id.split("/")[-1]

    def transcribe(self, audio16k: np.ndarray) -> str:
        if audio16k.size < 1600:
            return ""
        out = self.model.generate(audio16k.astype(np.float32), language=self.language, temperature=0.0,
                                  condition_on_previous_text=False, verbose=False)
        text = getattr(out, "text", None) or (out.get("text") if isinstance(out, dict) else "") or ""
        return text.strip()
