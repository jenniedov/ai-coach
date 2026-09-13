"""The conversation engine, built on Pipecat (https://github.com/pipecat-ai/pipecat).

Pipecat provides the hard parts that shouldn't be reinvented: Silero VAD, semantic end-of-turn
detection (Smart Turn v3), interruption / barge-in handling, sentence aggregation into TTS, WebRTC
transport (with browser echo cancellation), and per-service metrics. This module plugs our local
engines (Parakeet STT, Kokoro / Pocket / Breeze TTS) and the coach knowledge retrieval into it.

Pipeline:  mic (WebRTC) -> VAD -> STT -> user aggregator -> coach context injector (RAG)
           -> LLM (OpenAI-compatible: xAI Grok / Groq / ...) -> TTS -> speaker (WebRTC)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator

import numpy as np
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from .coach.retrieval import format_context
from .config import settings
from .engines import Engines

log = logging.getLogger(__name__)

# Whisper-style hallucinations on silence / breath noise
HALLUCINATIONS = re.compile(
    r"^(thank(s| you)( for watching)?[.!]?|you[.]?|bye[.!]?|\.+|hmm+[.]?|uh+[.]?|okay[.]?|so[.]?|"
    r"subtitles? by .*|thanks for listening[.!]?|the end[.]?)$", re.I)


@dataclass
class SessionOptions:
    coach_id: str
    voice: str
    llm_model: str
    tts_engine: str
    barge_in: bool = True
    speed: float = 1.0
    turn_mode: str = "smart"  # smart (Smart Turn v3 semantic endpointing) | silence (fixed timeout)


# --------------------------------------------------------------------------- STT
class LocalSTTService(SegmentedSTTService):
    """Wraps our STTProvider (Parakeet MLX / Whisper MLX) as a Pipecat segmented STT service."""

    def __init__(self, provider, executor, **kwargs):
        super().__init__(sample_rate=16000, settings=STTSettings(model=provider.name, language=Language.EN), **kwargs)
        self.provider = provider
        self.executor = executor

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw int16 PCM

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            await self.start_processing_metrics()
            pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self.executor, self.provider.transcribe, pcm)
            await self.stop_processing_metrics()
            text = (text or "").strip()
            secs = pcm.size / 16000
            if not text or HALLUCINATIONS.match(text) or (secs < 0.5 and len(text) < 8):
                log.info("STT dropped %.2fs utterance: %r", secs, text)
                return
            log.info("STT (%.2fs audio): %s", secs, text)
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
        except Exception as e:  # noqa: BLE001
            log.exception("STT error")
            yield ErrorFrame(error=f"STT error: {e}")


# --------------------------------------------------------------------------- TTS
class LocalTTSService(TTSService):
    """Wraps our TTSProvider (Kokoro / Pocket / Breeze / macOS say) as a Pipecat TTS service.
    Pipecat aggregates the LLM stream into sentences and calls run_tts per sentence; audio is
    generated in the MLX worker thread and streamed back chunk by chunk."""

    def __init__(self, provider, voice: str, speed: float, executor, **kwargs):
        super().__init__(sample_rate=provider.sample_rate, push_start_frame=True, push_stop_frames=True,
                         max_consecutive_zero_audio_contexts=0,  # a silent fragment must never disable the engine
                         settings=TTSSettings(model=provider.name, voice=voice, language=Language.EN), **kwargs)
        self.provider = provider
        self.voice = voice
        self.speed = speed
        self.executor = executor

    def can_generate_metrics(self) -> bool:
        return True

    async def push_error(self, error_msg: str, *args, **kwargs):
        # Pipecat reports a TTS context that ended without audio (an empty trailing fragment, or a chunk
        # cancelled by an interruption) as a non-fatal error. It is harmless here; keep it out of the UI.
        if "completed with no audio" in str(error_msg):
            log.debug("ignored: %s", error_msg)
            return
        await super().push_error(error_msg, *args, **kwargs)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        if not re.search(r"[A-Za-z0-9\u0590-\u05ff]", text):
            # punctuation-only fragment (e.g. a lone dash or quote): emit a beat of silence instead of nothing
            yield TTSAudioRawFrame(audio=bytes(int(self.sample_rate * 0.06) * 2), sample_rate=self.sample_rate,
                                   num_channels=1, context_id=context_id)
            return

        def worker():
            try:
                for chunk in self.provider.stream(text, self.voice, self.speed):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        await self.start_tts_usage_metrics(text)
        fut = loop.run_in_executor(self.executor, worker)
        produced = 0
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    log.error("TTS error: %s", item)
                    yield ErrorFrame(error=f"TTS error: {item}")
                    break
                if item is None or getattr(item, "size", 0) == 0:
                    continue
                await self.stop_ttfb_metrics()
                pcm = (np.clip(item, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                produced += len(pcm)
                yield TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
            if produced == 0:
                log.warning("TTS produced no audio for %r; sending 60 ms of silence", text)
                yield TTSAudioRawFrame(audio=bytes(int(self.sample_rate * 0.06) * 2), sample_rate=self.sample_rate,
                                       num_channels=1, context_id=context_id)
        finally:
            try:
                self.provider.stop()
            except Exception:  # noqa: BLE001
                pass
            await self.stop_ttfb_metrics()
            if not fut.done():
                fut.add_done_callback(lambda f: None)


# --------------------------------------------------------------------------- RAG injector
class CoachContextInjector(FrameProcessor):
    """Before every LLM call: retrieve the coach knowledge relevant to the latest user message and
    rebuild the system prompt with it. Keeps the per-request prompt small and targeted."""

    def __init__(self, agent, rtvi: RTVIProcessor, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.rtvi = rtvi

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content or "")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                msgs = list(frame.context.messages)
                users = [self._text(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if users:
                    t0 = time.perf_counter()
                    results = self.agent.retriever.search(users[-1], history=users[:-1])
                    ms = (time.perf_counter() - t0) * 1000
                    sys_text = self.agent.system_prompt(format_context(results))
                    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                        msgs[0] = {"role": "system", "content": sys_text}
                    else:
                        msgs.insert(0, {"role": "system", "content": sys_text})
                    frame.context.set_messages(msgs)
                    await self.rtvi.send_server_message({
                        "t": "retrieval", "ms": round(ms),
                        "chunks": [{"id": c.id, "topic": c.topic, "evidence": c.evidence, "score": round(s, 3),
                                    "text": c.text[:240]} for c, s in results],
                    })
            except Exception:  # noqa: BLE001
                log.exception("retrieval failed; continuing without it")
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------- latency observer
class LatencyObserver(BaseObserver):
    """End-to-end turn timing: speech_end -> transcript -> LLM first token -> first TTS audio -> bot speaking."""

    def __init__(self, rtvi: RTVIProcessor, **kwargs):
        super().__init__(**kwargs)
        self.rtvi = rtvi
        self.turn: dict = {}
        self.turn_id = 0
        self._seen: set[int] = set()
        self.last: dict | None = None

    def start_typed_turn(self):
        self.turn_id += 1
        now = time.perf_counter()
        self.turn = {"id": self.turn_id, "speech_end": now, "vad_speech_end": now, "stt_done": now, "typed": True}

    def _ms(self, a: str, b: str):
        if a in self.turn and b in self.turn:
            v = round((self.turn[b] - self.turn[a]) * 1000)
            return v if v >= 0 else None  # out-of-order frames (typed turns, interruptions) -> not a real number
        return None

    def summary(self) -> dict:
        return {
            "turn_id": self.turn.get("id"),
            # STT compute time for the last segment (from Pipecat's processing metric)
            "stt_ms": self.turn.get("stt_processing_ms"),
            "endpoint_ms": self._ms("vad_speech_end", "speech_end"),        # end-of-turn decision time
            "llm_first_token_ms": self._ms("llm_start", "llm_first_token"),
            "llm_total_ms": self._ms("llm_start", "llm_done"),
            "tts_first_audio_ms": self._ms("llm_first_token", "tts_first_audio"),
            "total_ms": self._ms("vad_speech_end", "tts_first_audio"),      # you stop talking -> first audio
            "total_from_turn_end_ms": self._ms("speech_end", "tts_first_audio"),
            "bot_started_ms": self._ms("vad_speech_end", "bot_started"),
            "interrupted": self.turn.get("interrupted", False),
            "typed": self.turn.get("typed", False),
        }

    def _reset_response(self):
        for k in ("llm_first_token", "llm_done", "tts_first_audio", "bot_started", "bot_stopped"):
            self.turn.pop(k, None)

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if f.id in self._seen:
            return
        self._seen.add(f.id)
        if len(self._seen) > 20000:
            self._seen.clear()
        t = time.perf_counter()
        if isinstance(f, UserStartedSpeakingFrame):
            if self.turn.get("bot_started") and not self.turn.get("bot_stopped"):
                self.turn["interrupted"] = True
                await self._send("latency")
            self.turn_id += 1
            self.turn = {"id": self.turn_id, "speech_start": t}
        elif isinstance(f, VADUserStoppedSpeakingFrame):
            self.turn["vad_speech_end"] = t
        elif isinstance(f, UserStoppedSpeakingFrame):
            self.turn["speech_end"] = t
            self.turn.setdefault("vad_speech_end", t)
        elif isinstance(f, TranscriptionFrame):
            self.turn["stt_done"] = t
        elif isinstance(f, MetricsFrame):
            for d in f.data:
                if isinstance(d, ProcessingMetricsData) and "STT" in str(d.processor) and d.value:
                    self.turn["stt_processing_ms"] = round(d.value * 1000)
        elif isinstance(f, LLMFullResponseStartFrame):
            self.turn["llm_start"] = t
            self._reset_response()
        elif isinstance(f, LLMTextFrame):
            self.turn.setdefault("llm_first_token", t)
        elif isinstance(f, LLMFullResponseEndFrame):
            self.turn["llm_done"] = t
        elif isinstance(f, TTSAudioRawFrame):
            self.turn.setdefault("tts_first_audio", t)
        elif isinstance(f, BotStartedSpeakingFrame):
            if "bot_started" not in self.turn:
                self.turn["bot_started"] = t
                await self._send("latency")
        elif isinstance(f, BotStoppedSpeakingFrame):
            self.turn["bot_stopped"] = t
            await self._send("latency")

    async def _send(self, kind: str):
        if "llm_start" not in self.turn:
            return  # nothing measurable (aborted turn)
        s = self.summary()
        self.last = s
        log.info("Turn %s latency: %s", s["turn_id"], s)
        try:
            await self.rtvi.send_server_message({"t": kind, **s})
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- LLM factory
def build_llm_service(engines: Engines, model: str) -> OpenAILLMService:
    p = engines.llm  # our provider object carries base_url / api_key / per-model overrides
    extra: dict = {"max_completion_tokens": settings.LLM_MAX_TOKENS}
    ov = dict(getattr(p, "per_model", {}).get(model, {}))
    real_model = ov.pop("model", model)
    extra.update(ov)
    params = OpenAILLMService.InputParams(temperature=settings.LLM_TEMPERATURE, extra=extra)
    return OpenAILLMService(api_key=p.api_key or "missing", base_url=p.base_url, model=real_model, params=params)


# --------------------------------------------------------------------------- bot
async def run_bot(connection: SmallWebRTCConnection, engines: Engines, opts: SessionOptions):
    coach = engines.coaches[opts.coach_id]
    agent = engines.agents[opts.coach_id]
    tts_provider = engines.get_tts(opts.tts_engine)

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True,
                               audio_in_sample_rate=16000, audio_out_sample_rate=tts_provider.sample_rate),
    )
    rtvi = RTVIProcessor(transport=transport)
    stt = LocalSTTService(engines.stt, engines.mlx_executor)
    tts = LocalTTSService(tts_provider, opts.voice, opts.speed, engines.mlx_executor)
    llm = build_llm_service(engines, opts.llm_model)

    vad = SileroVADAnalyzer(params=VADParams(confidence=settings.VAD_THRESHOLD, start_secs=0.2, stop_secs=0.2,
                                             min_volume=settings.VAD_MIN_VOLUME))
    start = [VADUserTurnStartStrategy(enable_interruptions=opts.barge_in),
             TranscriptionUserTurnStartStrategy(enable_interruptions=opts.barge_in)]
    if opts.turn_mode == "smart":
        stop = [TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=settings.VAD_END_SILENCE_MS / 1000 + 0.35)))]
    else:
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=settings.VAD_END_SILENCE_MS / 1000)]

    context = LLMContext(messages=[{"role": "system", "content": agent.system_prompt("")}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad,
                                            user_turn_strategies=UserTurnStrategies(start=start, stop=stop)),
    )
    injector = CoachContextInjector(agent, rtvi)
    latency = LatencyObserver(rtvi)

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        aggregators.user(),
        injector,
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True,
                              audio_in_sample_rate=16000, audio_out_sample_rate=tts_provider.sample_rate),
        observers=[RTVIObserver(rtvi, params=RTVIObserverParams(bot_tts_enabled=True, metrics_enabled=True)),
                   latency],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_proc):
        await rtvi_proc.set_bot_ready()
        await rtvi_proc.send_server_message({
            "t": "session", "coach": coach.public(), "voice": opts.voice, "llm_model": opts.llm_model,
            "llm_provider": engines.llm.name, "llm_configured": engines.llm.is_configured(),
            "llm_missing_message": engines.llm.missing_config_message(),
            "tts_engine": tts_provider.name, "stt_engine": engines.stt.name, "barge_in": opts.barge_in,
            "turn_mode": opts.turn_mode,
        })

    @rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi_proc, msg):
        data = msg.data or {}
        if msg.type == "text" and isinstance(data, dict) and data.get("text", "").strip():
            latency.start_typed_turn()
            await task.queue_frames([LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": data["text"].strip()}], run_llm=True)])
        elif msg.type == "interrupt":
            await task.queue_frames([InterruptionFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_, conn):
        log.info("Client disconnected; ending session")
        await task.cancel()

    log.info("Session start: coach=%s voice=%s llm=%s/%s tts=%s stt=%s barge_in=%s turn=%s",
             coach.id, opts.voice, engines.llm.name, opts.llm_model, tts_provider.name, engines.stt.name,
             opts.barge_in, opts.turn_mode)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    log.info("Session ended")
