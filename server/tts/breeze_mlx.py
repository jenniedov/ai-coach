"""Breeze TTS 2 on MLX via mlx-audio (4-bit). LLM-based TTS with voice design (instruction) and
voice cloning (reference clip + exact transcript). Higher quality/expressiveness, higher latency
than Kokoro. Non-commercial license upstream (BreezeBlue)."""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from .base import TTSProvider, VoiceInfo
from .custom_voices import load_custom_voices

log = logging.getLogger(__name__)

DESIGNED_VOICES = {
    "warm_female": "A warm, confident woman in her forties with a clear, direct voice and natural pacing.",
    "british_female": "A confident British woman with a clear London accent, warm and direct, speaking at a natural pace.",
    "calm_male": "A calm, grounded man with a clear voice and natural pacing.",
}


class BreezeMLXTTS(TTSProvider):
    name = "breeze"
    sample_rate = 24000
    supports_cloning = True

    def __init__(self, model_id: str = "mlx-community/Breeze-TTS-2-mlx-4bit"):
        from mlx_audio.tts.utils import load_model

        self.model = load_model(model_id)
        try:
            from . import breeze_fast
            breeze_fast.apply(self.model)  # ~25% faster depth decoding, same output distribution
        except Exception as e:  # noqa: BLE001
            log.warning("breeze speed patch not applied: %s", e)
        self._custom = {v.id: v for v in load_custom_voices()}
        self._stop = False

    def _kwargs(self, voice: str) -> dict:
        if voice.startswith("custom:"):
            cv = self._custom.get(voice.split(":", 1)[1])
            if cv is None:
                raise ValueError(f"Unknown custom voice {voice}")
            kw = {"ref_audio": str(cv.reference), "ref_text": cv.reference_text or None}
            return kw
        instruct = DESIGNED_VOICES.get(voice, DESIGNED_VOICES["warm_female"])
        return {"instruct": instruct, "cfg_scale": 4.0}

    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self._stop = False
        chunks = []
        for r in self.model.generate(text=text, temperature=0.8, seed=None, split_pattern=None, **self._kwargs(voice)):
            chunks.append(np.array(r.audio, dtype=np.float32).reshape(-1))
            if self._stop:
                break
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stream(self, text: str, voice: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        self._stop = False
        for r in self.model.generate(text=text, temperature=0.8, stream=True, streaming_interval=0.6,
                                     split_pattern=None, **self._kwargs(voice)):
            if self._stop:
                break
            yield np.array(r.audio, dtype=np.float32).reshape(-1)

    def voices(self) -> list[VoiceInfo]:
        out = [VoiceInfo(id=k, name=f"{k.replace('_', ' ').title()} (Breeze designed)", engine=self.name,
                         description=v) for k, v in DESIGNED_VOICES.items()]
        for cv in self._custom.values():
            out.append(VoiceInfo(id=f"custom:{cv.id}", name=cv.name, engine=self.name, kind="custom",
                                 consent=cv.consent, description=cv.notes))
        return out
