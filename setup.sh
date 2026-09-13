#!/usr/bin/env bash
# One-time setup for AI Coach on Apple Silicon: Python env (uv), dependencies, speech models.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This project targets Apple Silicon macOS (MLX). Continuing anyway..."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)..."
  if command -v brew >/dev/null 2>&1; then brew install uv; else curl -LsSf https://astral.sh/uv/install.sh | sh; fi
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; installing with Homebrew (needed for audio decoding)..."
  brew install ffmpeg
fi

echo "==> Creating virtualenv + installing dependencies (Python 3.12, no PyTorch)"
uv python install 3.12 >/dev/null
uv sync --python 3.12

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — add XAI_API_KEY (or GROQ_API_KEY) to it."
fi

echo "==> Downloading speech models (~3.2 GB: Kokoro, Pocket TTS, Parakeet STT, Whisper fallback, embeddings)"
uv run python scripts/download_models.py "$@"

echo
echo "Setup complete. Next:"
echo "  1) put your key in .env  (XAI_API_KEY=...  or GROQ_API_KEY=...)"
echo "  2) ./start.sh   then open http://localhost:3000"
