#!/usr/bin/env bash
# Copies the Groq API key stored by the FreeFlow app into this project's .env (GROQ_API_KEY=...).
# Run it yourself; the key never leaves your machine.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$HOME/Library/Application Support/FreeFlow/.settings"
[[ -f "$SRC" ]] || { echo "FreeFlow settings not found at $SRC"; exit 1; }
KEY=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('groq_api_key',''))" "$SRC")
[[ -n "$KEY" ]] || { echo "No groq_api_key in FreeFlow settings"; exit 1; }
[[ -f .env ]] || cp .env.example .env
if grep -q '^GROQ_API_KEY=' .env; then
  sed -i '' "s|^GROQ_API_KEY=.*|GROQ_API_KEY=${KEY}|" .env
else
  echo "GROQ_API_KEY=${KEY}" >> .env
fi
chmod 600 .env
echo "GROQ_API_KEY written to .env (${#KEY} chars). Restart ./start.sh."
