"""Process-wide singletons: coaches + retrievers, LLM provider, STT engine, lazily-loaded TTS engines,
and one dedicated MLX worker thread (MLX models are called from a single thread to avoid contention)."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .config import settings

log = logging.getLogger(__name__)


class Engines:
    def __init__(self):
        from .coach.agent import CoachAgent
        from .coach.loader import load_all
        from .coach.retrieval import CoachRetriever
        from .llm import build_provider

        self.coaches = load_all(settings.COACHES_DIR)
        self.retrievers = {cid: CoachRetriever(c) for cid, c in self.coaches.items()}
        self.agents = {cid: CoachAgent(c, self.retrievers[cid]) for cid, c in self.coaches.items()}
        for r in self.retrievers.values():  # warm the ONNX embedder so the first real query is fast
            r.search("warm up")
        self.llm = build_provider()
        self.stt = None
        self._tts: dict[str, object] = {}
        self.mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def load_stt(self):
        from .stt import build_stt
        t0 = time.time()
        self.stt = build_stt()
        self.stt.warmup()
        log.info("STT ready (%s) in %.1fs", self.stt.name, time.time() - t0)

    def get_tts(self, engine: str):
        from .tts import build_tts
        if engine not in self._tts:
            t0 = time.time()
            self._tts[engine] = build_tts(engine)
            log.info("TTS engine %s loaded in %.1fs", engine, time.time() - t0)
        return self._tts[engine]

    def tts_engines(self) -> list[dict]:
        from .tts import available_engines
        return available_engines()

    def voices(self) -> list[dict]:
        from .tts import list_all_voices
        return list_all_voices()
