"""Small local RAG over a coach's knowledge chunks.

Hybrid retrieval: BM25 (pure python, always available) + dense embeddings (fastembed / ONNX,
optional). Embeddings are cached on disk per coach keyed by a content hash so restarts are fast.
Nothing here leaves the machine."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import settings
from .loader import Coach, KnowledgeChunk

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or but if then than so to of in on at for from by with about as is are was were be been
being i you he she it we they me him her us them my your his its our their this that these those what which who whom
how when where why not no do does did have has had can could would should will just like very really also there here
im i'm ive i've dont don't its it's thinking think want going go get got make made one two three four five lot
tell told someone something anything everything would should could might may feel feeling feels need needs know
knows kind sort thing things once actually right really maybe guess okay ok yeah yes way ways say says said give
much many more most some any every all lots trying try wanted thought""".split())


def _tok(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 1.0
        self.df: Counter = Counter()
        for d in docs:
            self.df.update(set(d))
        self.tf = [Counter(d) for d in docs]

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        for i, d in enumerate(self.docs):
            dl = len(d)
            s = 0.0
            tf = self.tf[i]
            for q in query:
                f = tf.get(q, 0)
                if not f:
                    continue
                s += self.idf(q) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


class Embedder:
    """Lazy fastembed wrapper (ONNX, CPU, ~130 MB model). If unavailable, retrieval is BM25-only."""
    _instance = None

    def __init__(self):
        self.model = None
        self.ok = False
        try:
            from fastembed import TextEmbedding  # type: ignore
            self.model = TextEmbedding(model_name=settings.EMBEDDING_MODEL,
                                       cache_dir=str(settings.MODELS_DIR / "fastembed"))
            self.ok = True
        except Exception as e:  # noqa: BLE001
            log.warning("Dense embeddings unavailable (%s). Using BM25 only.", e)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray | None:
        if not self.ok:
            return None
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs

    def embed_query(self, text: str) -> np.ndarray | None:
        """bge models expect a retrieval instruction prefix on queries; fastembed adds it here."""
        if not self.ok:
            return None
        fn = getattr(self.model, "query_embed", None)
        vec = np.array(list(fn([text]) if fn else self.model.embed([text])), dtype=np.float32)[0]
        return vec / (np.linalg.norm(vec) + 1e-8)


class CoachRetriever:
    def __init__(self, coach: Coach, use_embeddings: bool = True):
        self.coach = coach
        self.chunks: list[KnowledgeChunk] = coach.chunks
        self.bm25 = BM25([_tok(f"{c.topic} {' '.join(c.tags)} {c.text}") for c in self.chunks])
        self.emb: np.ndarray | None = None
        if use_embeddings and self.chunks:
            self._build_embeddings()

    def _build_embeddings(self):
        idx_dir = self.coach.dir / ".index"
        idx_dir.mkdir(exist_ok=True)
        h = hashlib.sha1(("\n".join(c.text for c in self.chunks) + settings.EMBEDDING_MODEL).encode()).hexdigest()[:16]
        cache = idx_dir / f"emb_{h}.npy"
        if cache.exists():
            self.emb = np.load(cache)
            log.info("Loaded cached embeddings for %s (%s)", self.coach.id, cache.name)
            return
        t0 = time.time()
        emb = Embedder.get()
        if not emb.ok:
            return
        texts = [f"{c.topic}: {c.text}" for c in self.chunks]
        self.emb = emb.embed(texts)
        np.save(cache, self.emb)
        for old in idx_dir.glob("emb_*.npy"):
            if old != cache:
                old.unlink(missing_ok=True)
        log.info("Built embeddings for %s: %d chunks in %.1fs", self.coach.id, len(texts), time.time() - t0)

    def search(self, query: str, top_k: int | None = None, history: list[str] | None = None) -> list[tuple[KnowledgeChunk, float]]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        if not self.chunks:
            return []
        q = query
        if history:
            # light context carry-over: previous user turn helps with follow-ups like "why?"
            q = f"{history[-1]} {query}" if len(_tok(query)) < 4 else query
        qtok = _tok(q)
        bm = self.bm25.scores(qtok)
        # lexical hits on a single conversational word are noise: require >=2 matched terms for full credit
        if len(qtok) >= 2:
            hits = np.array([len(set(qtok) & set(d)) for d in self.bm25.docs])
            bm = np.where(hits >= 2, bm, bm * 0.4)
        if bm.max() > 0:
            bm = bm / bm.max()
        score = bm.copy()
        if self.emb is not None:
            qv = Embedder.get().embed_query(q)
            if qv is not None:
                cos = self.emb @ qv
                cos = np.clip((cos - 0.3) / 0.5, 0, 1)  # bge-small cosines mostly live in 0.3-0.8
                score = 0.3 * bm + 0.7 * cos
        order = np.argsort(-score)[: top_k * 2]
        out: list[tuple[KnowledgeChunk, float]] = []
        seen_files: Counter = Counter()
        for i in order:
            c = self.chunks[int(i)]
            if score[i] <= 0.05:
                continue
            if seen_files[c.file] >= 3:  # diversity: at most 3 claims per topic file
                continue
            seen_files[c.file] += 1
            out.append((c, float(score[i])))
            if len(out) >= top_k:
                break
        return out


def format_context(results: list[tuple[KnowledgeChunk, float]]) -> str:
    if not results:
        return ""
    lines = []
    for c, s in results:
        tag = {"DIRECT": "direct", "INFERRED": "inferred", "SUMMARY": "theme"}.get(c.evidence, "note")
        text = re.sub(r"\[(DIRECT|INFERRED|SUMMARY)\]\s*", "", c.text)
        lines.append(f"- ({c.topic}; {tag}) {text}")
    return "\n".join(lines)
