"""Kokoro-82M on MLX via mlx-audio. Default low-latency engine (~0.3-0.5 s per sentence on M1 Pro)."""
from __future__ import annotations

import logging

import numpy as np

from .base import TTSProvider, VoiceInfo

log = logging.getLogger(__name__)

# name -> (label, lang_code)  a = American English, b = British English
KOKORO_VOICES = {
    "af_heart": ("Heart (US female, default)", "a"),
    "af_bella": ("Bella (US female)", "a"),
    "af_nicole": ("Nicole (US female, soft)", "a"),
    "af_sarah": ("Sarah (US female)", "a"),
    "af_sky": ("Sky (US female)", "a"),
    "af_nova": ("Nova (US female)", "a"),
    "am_adam": ("Adam (US male)", "a"),
    "am_michael": ("Michael (US male)", "a"),
    "am_fenrir": ("Fenrir (US male)", "a"),
    "bf_emma": ("Kokoro stock voice \"bf_emma\" (UK female, not a clone)", "b"),
    "bf_isabella": ("Isabella (UK female)", "b"),
    "bf_alice": ("Alice (UK female)", "b"),
    "bf_lily": ("Lily (UK female)", "b"),
    "bm_george": ("George (UK male)", "b"),
    "bm_daniel": ("Daniel (UK male)", "b"),
    "bm_lewis": ("Lewis (UK male)", "b"),
}


class KokoroMLXTTS(TTSProvider):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, model_id: str = "mlx-community/Kokoro-82M-bf16"):
        from mlx_audio.tts.utils import load_model

        self.model = load_model(model_id)
        self._stop = False

    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self._stop = False
        voice = voice if voice in KOKORO_VOICES else "af_heart"
        lang = KOKORO_VOICES[voice][1]
        chunks = []
        for r in self.model.generate(text=text, voice=voice, speed=speed, lang_code=lang, split_pattern=r"\n+"):
            chunks.append(np.array(r.audio, dtype=np.float32).reshape(-1))
            if self._stop:
                break
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id=k, name=v[0], engine=self.name) for k, v in KOKORO_VOICES.items()]
