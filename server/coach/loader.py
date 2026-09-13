"""Loads coach profiles from coaches/<id>/ directories.

Layout:
  coaches/<id>/profile.json      – identity, bio, expertise, disclaimer
  coaches/<id>/system_prompt.md  – behavioural system prompt
  coaches/<id>/style.md          – communication style guide (also retrievable)
  coaches/<id>/sources.json      – citations for the research
  coaches/<id>/knowledge/*.md    – topic files with numbered, source-tagged claims
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class KnowledgeChunk:
    id: str
    coach_id: str
    topic: str
    tags: list[str]
    text: str
    file: str
    evidence: str  # DIRECT | INFERRED | UNKNOWN


@dataclass
class Coach:
    id: str
    dir: Path
    profile: dict
    system_prompt: str
    style: str
    sources: list[dict]
    chunks: list[KnowledgeChunk] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.profile.get("display_name") or self.profile.get("name") or self.id

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tagline": self.profile.get("tagline", ""),
            "short_bio": self.profile.get("short_bio", ""),
            "disclaimer": self.profile.get("disclaimer",
                "AI simulation based on publicly available material. "
                "Not affiliated with or endorsed by the person represented."),
            "expertise": self.profile.get("expertise", []),
            "not_expertise": self.profile.get("not_expertise", []),
            "n_sources": len(self.sources),
            "n_chunks": len(self.chunks),
            "voice": self.profile.get("voice"),
            "tts_engine": self.profile.get("tts_engine"),
        }


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"').strip("'")
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v
    return meta, text[m.end():]


CLAIM_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s+(.*)$")


def _split_claims(body: str) -> list[str]:
    """Split a knowledge file into retrieval units. Numbered / bulleted claims become one
    chunk each (continuation lines are folded in). Prose paragraphs become chunks too."""
    chunks: list[str] = []
    cur: list[str] = []

    def flush():
        if cur:
            t = " ".join(s.strip() for s in cur).strip()
            if len(t) > 25:
                chunks.append(t)
            cur.clear()

    for line in body.splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            flush()
            continue
        if CLAIM_RE.match(line) and not line.startswith("  "):
            flush()
            cur.append(CLAIM_RE.match(line).group(1))
        else:
            cur.append(line)
    flush()
    return chunks


def _evidence(text: str) -> str:
    if "[DIRECT]" in text:
        return "DIRECT"
    if "[INFERRED]" in text:
        return "INFERRED"
    return "UNKNOWN"


def load_coach(coach_dir: Path) -> Coach | None:
    profile_path = coach_dir / "profile.json"
    if not profile_path.exists():
        return None
    profile = json.loads(profile_path.read_text())
    coach_id = profile.get("id") or coach_dir.name
    system_prompt = (coach_dir / "system_prompt.md").read_text() if (coach_dir / "system_prompt.md").exists() else ""
    style = (coach_dir / "style.md").read_text() if (coach_dir / "style.md").exists() else ""
    sources = json.loads((coach_dir / "sources.json").read_text()) if (coach_dir / "sources.json").exists() else []
    coach = Coach(id=coach_id, dir=coach_dir, profile=profile, system_prompt=system_prompt,
                  style=style, sources=sources)

    kdir = coach_dir / "knowledge"
    for f in sorted(kdir.glob("*.md")) if kdir.exists() else []:
        meta, body = _parse_frontmatter(f.read_text())
        topic = meta.get("topic") or f.stem.replace("-", " ").title()
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        summary = meta.get("summary") or ""
        if summary:
            # topic-level chunk so questions that match a theme (not a specific claim) still land
            coach.chunks.append(KnowledgeChunk(
                id=f"{f.stem}#summary", coach_id=coach_id, topic=topic, tags=tags,
                text=f"[SUMMARY] {summary}", file=f.name, evidence="SUMMARY"))
        for i, claim in enumerate(_split_claims(body)):
            coach.chunks.append(KnowledgeChunk(
                id=f"{f.stem}#{i}", coach_id=coach_id, topic=topic, tags=tags,
                text=claim, file=f.name, evidence=_evidence(claim)))
    # style.md is NOT indexed for retrieval: the system prompt already carries the style guide.
    log.info("Loaded coach %s: %d knowledge chunks, %d sources", coach_id, len(coach.chunks), len(sources))
    return coach


def load_all(coaches_dir: Path) -> dict[str, Coach]:
    coaches: dict[str, Coach] = {}
    if not coaches_dir.exists():
        return coaches
    for d in sorted(coaches_dir.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            try:
                c = load_coach(d)
                if c:
                    coaches[c.id] = c
            except Exception as e:  # noqa: BLE001
                log.exception("Failed to load coach %s: %s", d, e)
    return coaches
