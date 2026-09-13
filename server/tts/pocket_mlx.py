"""Kyutai Pocket TTS on MLX via mlx-audio. 100M params, streams, and supports consent-based voice
cloning from a 5-20 s reference clip. The reference embedding is computed once per voice and cached."""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from .base import TTSProvider, VoiceInfo
from .custom_voices import load_custom_voices

log = logging.getLogger(__name__)

BUILTIN = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


class PocketMLXTTS(TTSProvider):
    name = "pocket"
    sample_rate = 24000
    supports_cloning = True

    def __init__(self, model_id: str = "mlx-community/pocket-tts"):
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        self._mx = mx
        self.model = load_model(model_id)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self._custom = {v.id: v for v in load_custom_voices()}
        self._state_cache: dict[str, object] = {}
        self._stop = False

    def _prompt_for(self, voice: str):
        if voice.startswith("custom:"):
            cv = self._custom.get(voice.split(":", 1)[1])
            if cv is None:
                raise ValueError(f"Unknown custom voice {voice}")
            return str(cv.reference)
        return voice if voice in BUILTIN else "alba"

    def _state(self, voice: str):
        """Cache the encoded audio prompt so cloning doesn't re-encode the reference every sentence.
        Generation mutates the state (KV cache), so callers always get a fresh copy."""
        if voice not in self._state_cache:
            prompt = self._prompt_for(voice)
            from mlx_audio.utils import load_audio
            if prompt not in BUILTIN:
                prompt = load_audio(prompt, sample_rate=self.sample_rate)
            self._state_cache[voice] = self.model.get_state_for_audio_prompt(prompt)
        return self._clone(self._state_cache[voice])

    def _clone(self, obj):
        mx = self._mx
        if isinstance(obj, mx.array):
            return mx.array(obj)
        if isinstance(obj, dict):
            return {k: self._clone(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clone(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._clone(v) for v in obj)
        import copy
        try:
            return copy.deepcopy(obj)
        except Exception:  # noqa: BLE001
            return copy.copy(obj)

    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self._stop = False
        try:
            state = self._state(voice)
            audio = self.model.generate_audio(model_state=state, text_to_generate=text, frames_after_eos=None)
            return np.array(audio, dtype=np.float32).reshape(-1)
        except AttributeError:
            # API drift fallback: go through the public generate()
            prompt = self._prompt_for(voice)
            kw = {"ref_audio": prompt} if prompt not in BUILTIN else {"voice": prompt}
            chunks = [np.array(r.audio, dtype=np.float32).reshape(-1) for r in self.model.generate(text=text, **kw)]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stream(self, text: str, voice: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        self._stop = False
        state = self._state(voice)
        for chunk in self.model.generate_audio_stream(model_state=state, text_to_generate=text, frames_after_eos=None):
            if self._stop:
                break
            yield np.array(chunk, dtype=np.float32).reshape(-1)

    def voices(self) -> list[VoiceInfo]:
        out = [VoiceInfo(id=v, name=f"{v.capitalize()} (Pocket built-in)", engine=self.name) for v in BUILTIN]
        for cv in self._custom.values():
            out.append(VoiceInfo(id=f"custom:{cv.id}", name=cv.name, engine=self.name, kind="custom",
                                 consent=cv.consent, description=cv.notes))
        return out
