"""FastAPI app: serves the web UI, config/debug endpoints, and the WebRTC signalling endpoint that
hands each browser connection to a Pipecat bot (see bot.py)."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from .config import settings
from .engines import Engines

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-coach")

engines: Engines | None = None
webrtc_handler = SmallWebRTCRequestHandler(ice_servers=None)
_bot_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    t0 = time.time()
    engines = Engines()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(engines.mlx_executor, engines.load_stt)
    default_tts = engines.get_tts(settings.TTS_ENGINE)
    await loop.run_in_executor(engines.mlx_executor, default_tts.warmup, settings.TTS_VOICE)
    if default_tts.name == "kokoro":  # warm the British pipeline too
        await loop.run_in_executor(engines.mlx_executor, default_tts.warmup, "bf_isabella")
    # pre-load every engine a coach prefers (e.g. Pocket TTS with a cloned voice) so the first call is fast
    from .tts import default_voice_for
    wanted = {"pocket"}  # always keep the light clone engine warm as the fallback for Breeze
    for c in engines.coaches.values():
        wanted.add(c.profile.get("tts_engine") or "")
    for eng in sorted(e for e in wanted if e):
        c = next((c for c in engines.coaches.values() if c.profile.get("tts_engine") == eng), None)
        voice = ((c.profile.get("voice") or {}).get(eng) if c else None)
        if eng != default_tts.name:
            tts = await loop.run_in_executor(engines.mlx_executor, engines.get_tts, eng)
            await loop.run_in_executor(engines.mlx_executor, tts.warmup, voice or default_voice_for(eng))
    log.info("Startup complete in %.1fs. Coaches: %s. LLM: %s (%s). Open http://%s:%s",
             time.time() - t0, list(engines.coaches), engines.llm.name,
             "configured" if engines.llm.is_configured() else "NOT CONFIGURED", settings.HOST, settings.PORT)
    if not engines.llm.is_configured():
        log.warning(engines.llm.missing_config_message())
    yield
    for t in list(_bot_tasks):
        t.cancel()
    await webrtc_handler.close()


app = FastAPI(title="AI Coach", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.WEB_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(settings.WEB_DIR / "index.html")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "stt": engines.stt.name if engines and engines.stt else None,
            "llm": engines.llm.name if engines else None,
            "llm_configured": bool(engines and engines.llm.is_configured()),
            "coaches": list(engines.coaches) if engines else []}


@app.get("/api/config")
async def api_config():
    from .tts import default_voice_for
    tts_engines = engines.tts_engines()
    models = engines.llm.available_models()
    return {
        "coaches": [c.public() for c in engines.coaches.values()],
        "llm_provider": engines.llm.name,
        "llm_models": models,
        "default_llm_model": settings.LLM_MODEL if settings.LLM_MODEL in models else models[0],
        "llm_configured": engines.llm.is_configured(),
        "llm_missing_message": engines.llm.missing_config_message(),
        "tts_engines": tts_engines,
        "default_tts_engine": settings.TTS_ENGINE,
        "voices": engines.voices(),
        "default_voice": {e["id"]: default_voice_for(e["id"]) for e in tts_engines},
        "stt_engine": engines.stt.name if engines.stt else "none",
        "barge_in": settings.BARGE_IN_ENABLED,
        "turn_mode": settings.TURN_MODE,
        "disclaimer": "AI simulation based on publicly available material. "
                      "Not affiliated with or endorsed by the person represented.",
    }


@app.get("/api/coaches")
async def api_coaches():
    return [c.public() for c in engines.coaches.values()]


@app.get("/api/coaches/{coach_id}/sources")
async def api_sources(coach_id: str):
    c = engines.coaches.get(coach_id)
    if not c:
        return JSONResponse({"error": "unknown coach"}, status_code=404)
    return c.sources


@app.get("/api/retrieve")
async def api_retrieve(coach: str, q: str, k: int = 6):
    r = engines.retrievers.get(coach)
    if not r:
        return JSONResponse({"error": "unknown coach"}, status_code=404)
    t0 = time.perf_counter()
    res = r.search(q, top_k=k)
    return {"ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": [{"id": c.id, "topic": c.topic, "evidence": c.evidence, "score": round(s, 3), "text": c.text}
                        for c, s in res]}


@app.get("/api/tts")
async def api_tts(text: str, engine: str | None = None, voice: str | None = None, speed: float = 1.0):
    """Debug helper: synthesize text and return a WAV (headers carry synthesis time)."""
    import soundfile as sf
    from .tts import default_voice_for
    eng = await asyncio.get_running_loop().run_in_executor(engines.mlx_executor, engines.get_tts, engine or settings.TTS_ENGINE)
    v = voice or default_voice_for(eng.name)
    t0 = time.perf_counter()
    audio = await asyncio.get_running_loop().run_in_executor(engines.mlx_executor, eng.generate, text, v, speed)
    ms = round((time.perf_counter() - t0) * 1000)
    buf = io.BytesIO()
    sf.write(buf, audio, eng.sample_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Synthesis-Ms": str(ms), "X-Audio-Seconds": f"{audio.size / eng.sample_rate:.2f}"})


@app.post("/api/offer")
async def offer(request: Request):
    """WebRTC signalling: browser posts an SDP offer (+ session options in request_data)."""
    from .bot import SessionOptions, run_bot
    from .tts import default_voice_for

    body = await request.json()
    req = SmallWebRTCRequest.from_dict(body)
    data = body.get("request_data") or {}
    coach_id = data.get("coach") or next(iter(engines.coaches), None)
    if coach_id not in engines.coaches:
        return JSONResponse({"error": f"unknown coach {coach_id}"}, status_code=400)
    coach_pref = engines.coaches[coach_id].profile
    tts_engine = data.get("tts_engine") or coach_pref.get("tts_engine") or settings.TTS_ENGINE
    models = engines.llm.available_models()
    opts = SessionOptions(
        coach_id=coach_id,
        voice=data.get("voice") or (coach_pref.get("voice") or {}).get(tts_engine) or default_voice_for(tts_engine),
        llm_model=data.get("llm_model") or settings.LLM_MODEL or models[0],
        tts_engine=tts_engine,
        barge_in=bool(data.get("barge_in", settings.BARGE_IN_ENABLED)),
        speed=float(data.get("speed") or settings.TTS_SPEED),
        turn_mode=data.get("turn_mode") or settings.TURN_MODE,
    )

    # MLX models must be created and used from the same thread: load the engine in the MLX worker first
    await asyncio.get_running_loop().run_in_executor(engines.mlx_executor, engines.get_tts, tts_engine)

    async def on_connection(connection):
        t = asyncio.create_task(run_bot(connection, engines, opts))
        _bot_tasks.add(t)
        t.add_done_callback(_bot_tasks.discard)

    answer = await webrtc_handler.handle_web_request(request=req, webrtc_connection_callback=on_connection)
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: Request):
    body = await request.json()
    await webrtc_handler.handle_patch_request(SmallWebRTCPatchRequest(**body))
    return {"status": "success"}
