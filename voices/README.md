# Voices

Built-in voices need nothing here (Kokoro: 16 voices; Pocket TTS: 8; Breeze: designed-by-instruction).

Custom (cloned) voices live in `voices/<id>/`:

```
voices/my_voice/
  reference.wav    5–20 s, mono, one clean speaker (created by scripts/add_voice.py)
  voice.json       {"name", "reference", "reference_text", "consent", ...}
```

Register one with an authorized sample (your own voice, or a person who gave permission):

```bash
uv run python scripts/add_voice.py --name "My voice" --id my_voice --sample ~/clip.m4a --start 3 --duration 12 --i-have-permission --consent "my own voice"
```

Rules enforced by the server: a voice without a non-empty `consent` field is never loaded; reference
audio never leaves this machine; cloning engines are Pocket TTS (fast) and Breeze TTS 2 (slow on M1).
`reference*.wav` files are git-ignored.
