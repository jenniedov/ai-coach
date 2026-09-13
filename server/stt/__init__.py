from __future__ import annotations

import logging

from ..config import settings
from .base import STTProvider

log = logging.getLogger(__name__)


def build_stt(engine: str | None = None) -> STTProvider:
    engine = (engine or settings.STT_ENGINE).lower()
    if engine == "parakeet":
        try:
            from .parakeet_mlx import ParakeetSTT
            return ParakeetSTT(settings.STT_MODEL)
        except Exception as e:  # noqa: BLE001
            log.warning("Parakeet unavailable (%s); falling back to Whisper MLX", e)
            engine = "whisper"
    if engine in ("whisper", "mlx_whisper", "whisper_mlx"):
        from .whisper_mlx import WhisperMLXSTT
        return WhisperMLXSTT(settings.STT_FALLBACK_MODEL)
    raise ValueError(f"Unknown STT_ENGINE {engine!r}")


__all__ = ["STTProvider", "build_stt"]
