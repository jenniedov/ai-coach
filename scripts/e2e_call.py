#!/usr/bin/env python
"""Automated end-to-end call test (no browser needed).

Connects to the running server over WebRTC exactly like the web UI (aiortc), plays a spoken question
into the call, and records what comes back: RTVI events (transcript, LLM text, speaking state),
latency numbers from the server, and the bot's audio (saved as a WAV). Optionally sends a second
question mid-answer to test barge-in.

  uv run python scripts/e2e_call.py                       # speaks the default test question
  uv run python scripts/e2e_call.py --question "..." --tts-engine pocket --voice custom:emma_grede_sample
  uv run python scripts/e2e_call.py --barge-in-after 2.5   # interrupt the answer after 2.5 s
  uv run python scripts/e2e_call.py --typed "..."          # typed turn instead of audio
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame

ROOT = Path(__file__).resolve().parent.parent
RTVI = "rtvi-ai"
SR = 48000  # WebRTC opus rate
FRAME = 960  # 20 ms


class PlaylistTrack(AudioStreamTrack):
    """Plays queued float32 mono clips (at 48 kHz) into the call; silence otherwise."""

    def __init__(self):
        super().__init__()
        self.queue: list[np.ndarray] = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.pts = 0
        self.t0 = None
        self.play_start_times: list[float] = []

    def play(self, clip: np.ndarray):
        self.queue.append(clip)

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        # pace at real time
        target = self.t0 + self.pts / SR
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if self.buf.size < FRAME and self.queue:
            self.buf = np.concatenate([self.buf, self.queue.pop(0)])
            self.play_start_times.append(time.perf_counter())
        if self.buf.size >= FRAME:
            chunk, self.buf = self.buf[:FRAME], self.buf[FRAME:]
        else:
            chunk = np.zeros(FRAME, dtype=np.float32)
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = SR
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, SR)
        self.pts += FRAME
        return frame


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n = int(len(x) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)


async def synth_question(base: str, text: str) -> np.ndarray:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.get(f"{base}/api/tts", params={"text": text, "engine": "kokoro", "voice": "am_adam", "speed": 1.0})
        r.raise_for_status()
    import io
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return resample(audio, sr, SR)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--question", default="I'm thinking about starting three businesses at once. What would you tell me?")
    ap.add_argument("--typed", help="send a typed turn instead of audio")
    ap.add_argument("--tts-engine", default=None)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--turn-mode", default=None)
    ap.add_argument("--barge-in-after", type=float, default=None, help="seconds after bot starts speaking to interrupt with a 2nd question")
    ap.add_argument("--second-question", default="Sorry to interrupt. Which one should I start first?")
    ap.add_argument("--out", default=str(ROOT / "tests" / "out"))
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"synthesizing question audio via {a.base}/api/tts ...")
    q_audio = await synth_question(a.base, a.question)
    q2_audio = await synth_question(a.base, a.second_question) if a.barge_in_after else None

    pc = RTCPeerConnection()
    track = PlaylistTrack()
    pc.addTrack(track)
    dc = pc.createDataChannel("chat", ordered=True)
    events: list[tuple[float, dict]] = []
    bot_audio: list[np.ndarray] = []
    T0 = time.perf_counter()
    state = {"bot_ready": asyncio.Event(), "bot_started": asyncio.Event(), "bot_stopped": asyncio.Event(),
             "llm_text": "", "transcripts": [], "latency": [], "retrieval": None, "errors": [], "interrupted": False,
             "first_audio": None, "bot_start_t": None}

    @pc.on("track")
    def on_track(t):
        async def reader():
            while True:
                try:
                    f = await t.recv()
                except Exception:  # noqa: BLE001
                    return
                arr = f.to_ndarray()
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768
                mono = arr.reshape(-1) if arr.ndim == 1 else arr[0]
                if state["first_audio"] is None and np.abs(mono).max() > 0.01:
                    state["first_audio"] = time.perf_counter()
                bot_audio.append(mono.copy())
        asyncio.ensure_future(reader())

    def send(type_, data):
        dc.send(json.dumps({"label": RTVI, "type": type_, "id": f"m{int(time.time()*1000)}", "data": data}))

    @dc.on("open")
    def on_open():
        send("client-ready", {"version": "1.0.0", "about": {"library": "e2e_call.py", "library_version": "0.1"}})

    @dc.on("message")
    def on_message(msg):
        if isinstance(msg, str) and msg.startswith("pong"):
            return
        try:
            m = json.loads(msg)
        except Exception:  # noqa: BLE001
            return
        t = time.perf_counter() - T0
        events.append((t, m))
        typ, d = m.get("type"), m.get("data") or {}
        if typ == "bot-ready":
            state["bot_ready"].set()
        elif typ == "user-transcription" and d.get("final"):
            state["transcripts"].append(d["text"]); print(f"  [{t:6.2f}s] transcript: {d['text']}")
        elif typ == "bot-llm-text":
            state["llm_text"] += d.get("text", "")
        elif typ == "bot-started-speaking":
            state["bot_start_t"] = time.perf_counter(); state["bot_started"].set(); print(f"  [{t:6.2f}s] bot started speaking")
        elif typ == "bot-stopped-speaking":
            state["bot_stopped"].set(); print(f"  [{t:6.2f}s] bot stopped speaking")
        elif typ == "user-started-speaking":
            print(f"  [{t:6.2f}s] user started speaking (VAD)")
        elif typ == "user-stopped-speaking":
            print(f"  [{t:6.2f}s] user stopped speaking (turn end)")
        elif typ == "bot-interrupted":
            state["interrupted"] = True; print(f"  [{t:6.2f}s] bot interrupted")
        elif typ == "server-message":
            if d.get("t") == "latency":
                state["latency"].append(d)
            elif d.get("t") == "retrieval":
                state["retrieval"] = d; print(f"  [{t:6.2f}s] retrieval {d['ms']} ms: " + "; ".join(c['topic'] for c in d['chunks'][:4]))
            elif d.get("t") == "session":
                print(f"  [{t:6.2f}s] session: {d['llm_provider']}/{d['llm_model']} tts={d['tts_engine']} stt={d['stt_engine']} configured={d['llm_configured']}")
        elif typ == "error":
            state["errors"].append(d); print(f"  [{t:6.2f}s] ERROR: {d}")
        elif typ == "metrics":
            for p in d.get("ttfb", []):
                print(f"  [{t:6.2f}s] ttfb {p.get('processor')}: {p.get('value', 0)*1000:.0f} ms")
            for p in d.get("processing", []):
                print(f"  [{t:6.2f}s] processing {p.get('processor')}: {p.get('value', 0)*1000:.0f} ms")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    request_data = {k: v for k, v in {"coach": "emma_grede", "voice": a.voice, "tts_engine": a.tts_engine,
                                      "llm_model": a.llm_model, "turn_mode": a.turn_mode, "barge_in": True}.items() if v is not None}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{a.base}/api/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
                                                       "request_data": request_data})
        r.raise_for_status()
        ans = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    print("connected; waiting for bot-ready ...")
    try:
        await asyncio.wait_for(state["bot_ready"].wait(), 20)
    except asyncio.TimeoutError:
        print("bot-ready not received"); await pc.close(); sys.exit(1)

    ping = asyncio.ensure_future(_pinger(dc))
    await asyncio.sleep(1.0)
    if a.typed:
        print(f"typing: {a.typed}")
        state["speech_end"] = time.perf_counter()
        send("client-message", {"t": "text", "d": {"text": a.typed}})
    else:
        print(f"speaking question ({len(q_audio)/SR:.1f}s): {a.question}")
        track.play(q_audio)
        await asyncio.sleep(len(q_audio) / SR)
        state["speech_end"] = time.perf_counter()

    try:
        await asyncio.wait_for(state["bot_started"].wait(), a.timeout)
    except asyncio.TimeoutError:
        print("bot never started speaking")
    else:
        if a.barge_in_after:
            await asyncio.sleep(a.barge_in_after)
            print(f"  -> barge-in: speaking second question ({len(q2_audio)/SR:.1f}s)")
            state["bot_stopped"].clear(); state["bot_started"].clear()
            track.play(q2_audio)
            try:
                await asyncio.wait_for(state["bot_started"].wait(), a.timeout)
            except asyncio.TimeoutError:
                print("bot never answered the second question")
        try:
            await asyncio.wait_for(state["bot_stopped"].wait(), a.timeout)
        except asyncio.TimeoutError:
            print("bot never stopped speaking")
    await asyncio.sleep(0.5)
    ping.cancel()
    await pc.close()

    # ---- report
    print("\n=== RESULT")
    print("transcripts :", state["transcripts"])
    print("answer      :", state["llm_text"].strip()[:600])
    if state["speech_end"] and state["first_audio"]:
        print(f"client-measured speech_end -> first bot audio heard: {(state['first_audio'] - state['speech_end'])*1000:.0f} ms")
    for l in state["latency"]:
        print("server latency:", {k: v for k, v in l.items() if k != "t"})
    if state["errors"]:
        print("errors      :", state["errors"])
    if bot_audio:
        audio = np.concatenate(bot_audio)
        wav = out_dir / "bot_answer.wav"
        sf.write(str(wav), audio, SR)
        print(f"bot audio   : {len(audio)/SR:.1f}s saved to {wav}")
    (out_dir / "events.json").write_text(json.dumps(events, indent=1, default=str))
    ok = bool(state["transcripts"] or a.typed) and bool(state["llm_text"]) and bool(bot_audio) and not state["errors"]
    print("STATUS      :", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


async def _pinger(dc):
    while True:
        await asyncio.sleep(1)
        try:
            dc.send("ping")
        except Exception:  # noqa: BLE001
            return


if __name__ == "__main__":
    asyncio.run(main())
