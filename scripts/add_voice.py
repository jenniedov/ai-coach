#!/usr/bin/env python
"""Register an authorized custom voice for cloning (Pocket TTS / Breeze TTS 2).

  uv run python scripts/add_voice.py --name "My voice" --sample path/to/clip.(wav|mp3|mp4|m4a) \
      --id my_voice --i-have-permission --start 0 --duration 12

What it does (everything stays on this machine):
  1. extracts a mono 24 kHz WAV segment (5-20 s of ONE clean speaker) with ffmpeg
  2. transcribes it locally with Parakeet (Breeze needs the exact reference text)
  3. writes voices/<id>/voice.json with a consent attestation

The server only loads voices whose voice.json has a non-empty "consent" field. You are responsible
for having the right to clone the voice you register."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="display name shown in the UI")
    ap.add_argument("--sample", required=True, help="audio/video file containing the speaker")
    ap.add_argument("--id", help="folder name under voices/ (default: derived from --name)")
    ap.add_argument("--start", type=float, default=0.0, help="segment start (seconds)")
    ap.add_argument("--duration", type=float, default=12.0, help="segment length (5-20 s recommended)")
    ap.add_argument("--i-have-permission", action="store_true",
                    help="attest that you have the speaker's permission (or it is your own voice)")
    ap.add_argument("--consent", default="", help="free-text consent note, e.g. 'my own voice' / 'written permission 2026-09-01'")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-transcribe", action="store_true")
    a = ap.parse_args()

    if not a.i_have_permission:
        sys.exit("Refusing: pass --i-have-permission to attest you may clone this voice. "
                 "Do not clone voices of people who haven't agreed to it.")
    print("Reminder: the segment must contain ONLY the coach speaking (no interviewer, no music, no crosstalk).\n"
          "Check the transcript printed below; if it reads like two people, pick a different --start/--duration.")
    if not (3 <= a.duration <= 30):
        sys.exit("--duration should be between 3 and 30 seconds (5-20 s is ideal)")

    vid = a.id or re.sub(r"[^a-z0-9]+", "_", a.name.lower()).strip("_")
    vdir = ROOT / "voices" / vid
    vdir.mkdir(parents=True, exist_ok=True)
    ref = vdir / "reference.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start), "-t", str(a.duration), "-i", a.sample,
                    "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(ref)], check=True)
    print(f"reference written: {ref}")

    text = ""
    if not a.no_transcribe:
        import numpy as np
        import soundfile as sf
        from server.stt import build_stt
        audio, sr = sf.read(str(ref), dtype="float32")
        n = int(len(audio) * 16000 / sr)
        a16 = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio).astype(np.float32)
        stt = build_stt()
        text = stt.transcribe(a16)
        print(f"transcript: {text}")

    meta = {
        "name": a.name,
        "reference": "reference.wav",
        "reference_text": text,
        "consent": a.consent or "user_attested",
        "attested_on": date.today().isoformat(),
        "source_file": str(Path(a.sample).name),
        "segment": {"start": a.start, "duration": a.duration},
        "notes": a.notes,
        "engines": ["pocket", "breeze"],
    }
    (vdir / "voice.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"voice.json written: {vdir / 'voice.json'}")
    print(f"Select it in the UI as 'custom:{vid}' under the Pocket or Breeze TTS engine (restart the server).")


if __name__ == "__main__":
    main()
