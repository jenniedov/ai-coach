# AI Coach — instructions for Claude Code

You are helping someone set up **AI Coach** on their own Mac and talk to their first coach. Be friendly,
ask one thing at a time, and do the technical steps yourself. Do not skip the consent questions.

## First run (do this when the repo has just been cloned)

1. **Install.** Run `./setup.sh`. It installs `uv`, a Python 3.12 environment, and about 3 GB of local
   speech models. Takes a few minutes. If `ffmpeg` or Homebrew is missing, it tells you what to do.
2. **LLM key.** Ask which language-model provider they want. Groq has a free tier and is fastest to
   set up (https://console.groq.com/keys); xAI Grok is prepaid (https://console.x.ai). Tell them to paste
   the key into `.env` themselves as `GROQ_API_KEY=...` or `XAI_API_KEY=...` (never ask them to paste a
   key into the chat; never read or print key values).
3. **Start it.** Run `./start.sh`, wait for "Startup complete" in the log, open http://localhost:3000
   in Chrome, and confirm the Example Coach answers a typed message. If it doesn't, read the server log
   and fix it before moving on.

## Adding their coach (ask, then build)

Ask these in order, waiting for each answer:

1. **"Who is the coach you want to add?"** (a public figure whose ideas they want to learn from)
2. **"Give me public sources: their website, articles, interview or podcast transcript pages, talks."**
   Ask for 5 to 20 links. Explain that the coach is a simulation built only from what this person has
   said publicly, and that the profile will say so on screen.
3. **"Do you have a short audio or video clip of ONLY this person speaking (10 to 20 seconds, no
   interviewer, no music), that you have the right to use?"**
   - If they say no, or the clip has anyone else talking, skip the voice: the coach uses a stock voice.
   - If yes, ask for the file path and which seconds to use. Tell them the file never leaves their Mac.
   - Before registering it, play or transcribe the segment (`scripts/add_voice.py` transcribes it) and
     check with them that it is one speaker only. If the transcript looks like two people, ask for a
     different segment.

Then run, in this order, reporting progress in plain words:

```bash
uv run python add_coach.py --name "<Name>" --website <url> --source <url> ...   # builds coaches/<id>/
uv run python scripts/add_voice.py --name "<Name>" --id <id>_voice --sample <file> --start <s> --duration 12 \
    --i-have-permission --consent "<what they told you, e.g. 'my own recording of a public talk, single speaker'>"
```

Open `coaches/<id>/profile.json` and set `"tts_engine": "pocket"` and `"voice": {"pocket": "custom:<id>_voice"}`
if a voice was registered (use `"breeze"` only if their Mac is an M3 Max or better). Restart `./start.sh`,
reload the page, select the coach in the sidebar, and ask them to press the phone and say hello.

## Rules that never change

- Everything is a **simulation of publicly expressed ideas**, never impersonation. Keep the disclaimer.
- Never download videos or audio from YouTube or other platforms for them. They supply local files.
- Never register a voice without their explicit "yes, I have the right to use this" and a single-speaker clip.
- Never commit `.env`, coach folders, or voice samples. `.gitignore` already excludes them; keep it so.
- Speech engines: Pocket TTS for any Mac; Breeze TTS 2 only on a strong Mac; Kokoro when they want speed.
- If a call gets stuck with Breeze, switch the coach to Pocket in Settings and in `profile.json`.

## Useful commands

```bash
./start.sh                                   # run the app
uv run pytest -q                             # tests
uv run python scripts/e2e_call.py --typed "hello"   # automated call test, prints latency
uv run python scripts/e2e_call.py            # spoken test question through the whole pipeline
```
