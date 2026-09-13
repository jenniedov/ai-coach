#!/usr/bin/env bash
# Launch the AI Coach server (FastAPI + Pipecat) on http://localhost:3000
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then echo "No .venv found — run ./setup.sh first."; exit 1; fi
if [[ ! -f .env ]]; then cp .env.example .env; echo "Created .env — add your XAI_API_KEY / GROQ_API_KEY."; fi
export PYTHONUNBUFFERED=1
PORT="${PORT:-3000}"
HOST="${HOST:-127.0.0.1}"
echo "AI Coach -> http://localhost:${PORT}   (Ctrl-C to stop)"
exec .venv/bin/python -m uvicorn server.main:app --host "$HOST" --port "$PORT" --log-level warning
