"""Consent-gated custom voices.

voices/<voice_id>/voice.json:
{
  "name": "Display name",
  "reference": "reference.wav",          # 5-20 s clean mono clip of ONE speaker
  "reference_text": "exact transcript",  # needed by engines that condition on text (Breeze)
  "consent": "user_provided",            # must be present & non-empty, else the voice is ignored
  "provided_by": "...", "notes": "..."
}
Nothing here is uploaded anywhere; reference audio stays on disk."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class CustomVoice:
    id: str
    name: str
    reference: Path
    reference_text: str
    consent: str
    notes: str = ""


def load_custom_voices() -> list[CustomVoice]:
    out: list[CustomVoice] = []
    if not settings.VOICES_DIR.exists():
        return out
    for d in sorted(settings.VOICES_DIR.iterdir()):
        vj = d / "voice.json"
        if not d.is_dir() or not vj.exists():
            continue
        try:
            meta = json.loads(vj.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("Bad voice.json in %s: %s", d, e)
            continue
        consent = (meta.get("consent") or "").strip()
        if not consent or consent.lower() in ("no", "false", "none", "unknown"):
            log.warning("Voice %s skipped: no consent attestation in voice.json", d.name)
            continue
        ref = d / meta.get("reference", "reference.wav")
        if not ref.exists():
            log.warning("Voice %s skipped: reference %s missing", d.name, ref)
            continue
        out.append(CustomVoice(id=d.name, name=meta.get("name", d.name), reference=ref,
                               reference_text=meta.get("reference_text", ""), consent=consent,
                               notes=meta.get("notes", "")))
    return out
