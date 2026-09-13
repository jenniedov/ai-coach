"""Zero-dependency fallback using the macOS `say` command (AVSpeechSynthesizer voices)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import TTSProvider, VoiceInfo


class MacOSSayTTS(TTSProvider):
    name = "macos_say"
    sample_rate = 24000

    def __init__(self):
        self._voices = None

    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        rate = int(175 * speed)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.wav"
            subprocess.run(["say", "-v", voice or "Samantha", "-r", str(rate),
                            "--data-format=LEI16@24000", "-o", str(out), text],
                           check=True, capture_output=True, timeout=60)
            audio, sr = sf.read(str(out), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio

    def voices(self) -> list[VoiceInfo]:
        if self._voices is None:
            self._voices = []
            try:
                out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
                for line in out.splitlines():
                    if "en_" in line or "en-" in line:
                        name = line.split("  ")[0].strip()
                        if name:
                            self._voices.append(VoiceInfo(id=name, name=f"{name} (macOS)", engine=self.name))
            except Exception:  # noqa: BLE001
                pass
            if not self._voices:
                self._voices = [VoiceInfo(id="Samantha", name="Samantha (macOS)", engine=self.name)]
        return self._voices
