# AI Coach

Have a voice conversation with an AI coach built from a public figure's **publicly expressed** ideas,
running on your own Mac. You press the phone button, talk, and the coach talks back in a couple of
seconds, in a voice you chose. You can interrupt it, mute yourself, or just type.

![AI Coach](docs/screenshot.png)

Everything about your voice and the coach's knowledge stays on your Mac. Only the text of the
conversation goes to the language-model provider you pick.

> Every coach is an **AI simulation based on publicly available material, not affiliated with or
> endorsed by the person represented**. The app says so on screen, and nothing it says is a quotation.

## What you need

- A Mac with Apple Silicon (M1 or newer). 16 GB of memory is enough; 8 GB works with the light speech engine.
- About 5 GB of free disk space (the speech models).
- Google Chrome (best microphone and echo handling).
- A language-model API key: **Groq** has a free tier (https://console.groq.com/keys), or **xAI Grok**
  (https://console.x.ai, prepaid).
- [Claude Code](https://claude.com/claude-code) if you want the guided setup below. Optional.

## Install (guided, with Claude Code)

```bash
git clone https://github.com/jenniedov/ai-coach.git
cd ai-coach
claude
```

Then say: **"Set this up and help me add my first coach."**

Claude reads the instructions in this repo, runs the installer, asks you for your API key
(you paste it into `.env` yourself), starts the app, and then asks:

1. Who is the coach you want to add?
2. Which public sources should it learn from? (website, articles, interview or podcast transcript pages, talks)
3. Do you have a short clip of **only that person speaking** that you have the right to use? If yes, it
   registers the voice; if not, the coach gets a stock voice.

Ten minutes later you're talking to your coach.

## Install (by hand)

```bash
git clone https://github.com/jenniedov/ai-coach.git
cd ai-coach
./setup.sh                        # Python env + ~3 GB of local speech models (a few minutes)
cp .env.example .env              # then put GROQ_API_KEY=... or XAI_API_KEY=... in .env
./start.sh                        # open http://localhost:3000 in Chrome
```

The repo ships with a generic **Example Coach** so you can test the call right away.

### Add a coach

```bash
uv run python add_coach.py
```

It asks for a name and public links (5 to 20 is a good range: the person's site, long interviews,
transcripts, talks), fetches them, has the language model extract evidence-tagged notes, and writes a
profile folder under `coaches/`. Restart the app and the coach appears in the sidebar.

### Add a voice (optional)

```bash
uv run python scripts/add_voice.py --name "Coach name" --id coach_voice \
    --sample ~/clip.mp4 --start 12 --duration 12 --i-have-permission --consent "single-speaker clip I recorded"
```

Rules, because they matter:

- The clip must be **only the coach talking**: no interviewer, no music, no crosstalk. 10 to 20 seconds
  is ideal. The script prints a transcript of the segment so you can check.
- Use audio you have the right to use. The tool never downloads from YouTube or anywhere else; you
  give it a local file, and the file never leaves your Mac.
- A voice without a consent note is ignored by the app.

Then in `coaches/<id>/profile.json` set `"tts_engine": "pocket"` and `"voice": {"pocket": "custom:coach_voice"}`.

## Using it

- **Phone button** starts and ends the call. Talk normally; the coach answers when you pause.
- Talk over the answer to **interrupt**, or press Interrupt.
- **Mute mic** (M) if the room is noisy; **Mute coach** (S) to read instead of listen.
- The gear opens **Settings**: speech engine, voice, language model, turn detection.

### Speech engines

| Engine | Voice clone | Speed on an M1 Pro | Pick it when |
|---|---|---|---|
| **Breeze TTS 2** | yes, best | ~12 s per sentence | you have a strong Mac (M3 Max and up feel live) |
| **Pocket TTS** | yes | under 1 s per sentence | any Apple Silicon Mac (default) |
| **Kokoro** | no, stock voices | ~0.4 s | you want it instant |
| macOS say | no | instant | nothing else works |

If a Breeze call gets stuck, the Mac isn't strong enough for it: end the call and choose Pocket TTS.
The app tells you this itself.

## How it works

```
mic ──WebRTC──▶ Silero VAD ▶ Smart Turn v3 ▶ Parakeet STT ▶ knowledge retrieval ▶ LLM ▶ TTS ──WebRTC──▶ speaker
                └─────────────── Pipecat pipeline (interruptions, sentence chunking, metrics) ───────────────┘
```

| Stage | Component | Where |
|---|---|---|
| Transport | WebRTC (Opus), browser echo cancellation | local |
| Voice activity / end of turn | Silero VAD + Smart Turn v3.2 (semantic), or a fixed pause | local |
| Speech-to-text | Parakeet TDT 0.6B v3 on MLX (fallback: Whisper-small MLX) | local |
| Knowledge retrieval | BM25 + bge-small embeddings over `coaches/<id>/knowledge` | local |
| Language model | Groq / xAI Grok / OpenAI / Ollama (model list read from your account) | remote (Ollama: local) |
| Text-to-speech | Breeze TTS 2 · Pocket TTS · Kokoro-82M · macOS say | local |
| Conversation engine | [Pipecat](https://github.com/pipecat-ai/pipecat) | local |

Measured on an M1 Pro: speech-to-text 100–300 ms, LLM first token 0.3–0.7 s (Groq), and about 2–3 s
from the moment you stop talking to the coach speaking with Pocket TTS.

## Project layout

```
server/            FastAPI + Pipecat bot (bot.py: VAD → STT → RAG → LLM → TTS, barge-in, latency)
coaches/           example_coach/ ships; coaches you build stay local (git-ignored) — see coaches/README.md
voices/            custom voices (voice.json + reference.wav), git-ignored — see voices/README.md
web/               the UI
scripts/           e2e_call.py (automated call test) · add_voice.py · download_models.py
add_coach.py       research + build a coach from public URLs
CLAUDE.md          the guided-setup instructions Claude Code follows
```

## Settings (`.env`)

`LLM_PROVIDER` (auto|xai|groq|openai|ollama), `LLM_MODEL`, `TTS_ENGINE`, `TTS_VOICE`, `TURN_MODE`
(smart|silence), `VAD_THRESHOLD` (raise in noisy rooms), `BARGE_IN_ENABLED`, `LLM_MAX_TOKENS`.
See `.env.example`.

## Tests

```bash
uv run pytest -q
uv run python scripts/e2e_call.py                # spoken question through the whole pipeline, prints latency
uv run python scripts/e2e_call.py --typed "hi"   # typed turn
```

## Limitations

- Install under a short path such as `~/ai-coach`: the bundled espeak-ng (used by Kokoro) truncates paths longer than about 150 characters.
- Breeze TTS 2 needs a strong Mac to feel live.
- Parakeet is English-first; set `STT_ENGINE=whisper` for other languages.
- Barge-in relies on the browser's echo cancellation; headphones make it rock solid.
- The knowledge base is public material only; where it has no position the coach reasons from its
  stated principles and says so.

## License

MIT for the code. Models keep their own licenses (Breeze TTS 2 is non-commercial).
