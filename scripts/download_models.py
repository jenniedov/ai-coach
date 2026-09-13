"""Download all local models into the Hugging Face cache (idempotent). Run via setup.sh."""
from __future__ import annotations

import argparse
import sys
import time

from huggingface_hub import snapshot_download

MODELS = {
    "kokoro": ("mlx-community/Kokoro-82M-bf16", dict(ignore_patterns=["samples/*", "*.mp4"])),
    "pocket": ("mlx-community/pocket-tts", {}),
    "parakeet": ("mlx-community/parakeet-tdt-0.6b-v3", {}),
    "whisper": ("mlx-community/whisper-small-mlx", {}),
    "breeze": ("mlx-community/Breeze-TTS-2-mlx-4bit", dict(ignore_patterns=["*.mp4"])),  # 3 GB, optional
}
DEFAULT = ["kokoro", "pocket", "parakeet", "whisper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-breeze", action="store_true", help="also download Breeze TTS 2 (3 GB)")
    ap.add_argument("--only", nargs="*", help="subset of: " + ", ".join(MODELS))
    a = ap.parse_args()
    names = a.only or DEFAULT + (["breeze"] if a.with_breeze else [])
    for n in names:
        repo, kw = MODELS[n]
        t = time.time()
        p = snapshot_download(repo, **kw)
        print(f"{n:9s} {repo} -> {p} ({time.time() - t:.0f}s)")
    # embeddings model (66 MB) via fastembed
    try:
        from fastembed import TextEmbedding
        from server.config import settings
        TextEmbedding(model_name=settings.EMBEDDING_MODEL, cache_dir=str(settings.MODELS_DIR / "fastembed"))
        print("embeddings ok")
    except Exception as e:  # noqa: BLE001
        print("embeddings download failed (retrieval will fall back to BM25):", e, file=sys.stderr)


if __name__ == "__main__":
    main()
