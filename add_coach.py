#!/usr/bin/env python
"""Add a new coach: research public sources, build the knowledge base, style guide, system prompt
and source list, so the coach appears in the UI on the next server start.

  uv run python add_coach.py                 # interactive prompts
  uv run python add_coach.py --name "Jane Doe" --website https://... --youtube URL --podcast URL --source URL

Pipeline (public-source research happens remotely; everything else is local):
  1. fetch each URL: web pages via trafilatura (readable text), YouTube via youtube-transcript-api
  2. ask the configured LLM (xAI Grok / Groq / ...) to extract, per source, evidence-tagged claims
  3. merge into coaches/<id>/knowledge/*.md (numbered [DIRECT]/[INFERRED] claims with source ids),
     style.md, profile.json, system_prompt.md, sources.json
Only public material you point it at is used; it does not scrape social feeds or clone voices."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.config import settings  # noqa: E402
from server.llm import build_provider  # noqa: E402

TOPICS = [
    "business principles", "decision-making frameworks", "entrepreneurship mindset", "branding and product",
    "hiring and team", "leadership", "execution and focus", "marketing", "partnerships and negotiation",
    "money and fundraising", "recurring stories and examples", "questions they ask", "common advice and sayings",
    "biography and track record", "risk and resilience", "career and ambition",
]

EXTRACT_PROMPT = """You are building a research-backed knowledge base about {name} for an AI coaching simulation.
Below is the text of ONE public source (id {sid}). Extract everything this source supports about {name}'s views,
frameworks, advice, stories and communication style. Rules:
- Only include claims this text supports. Never invent. Mark each claim [DIRECT] (explicitly stated by {name})
  or [INFERRED] (a pattern you infer). Quote verbatim only in quotation marks; otherwise paraphrase.
- Output JSON: {{"claims": [{{"topic": "<one of: {topics}>", "claim": "..."}}],
  "style_observations": ["..."], "bio_facts": ["..."]}}
- 5 to 40 claims. Each claim one to three sentences, concrete, useful for coaching.

SOURCE TITLE: {title}
SOURCE URL: {url}
TEXT:
{text}
"""

SYNTH_PROMPT = """You are writing the coaching persona files for an AI simulation of {name}, built ONLY from the
research notes below (already extracted from public sources, with source ids). Do not add facts.
Return JSON with keys:
  "profile": {{"tagline": "...", "short_bio": "3-4 factual sentences", "roles": [...], "expertise": [...],
               "not_expertise": [...], "signature_themes": [up to 10 short phrases]}},
  "style_md": "markdown: sections 'Observed (DIRECT)', 'Inferred patterns (INFERRED)', 'Areas of demonstrated
               expertise', 'Areas with NO demonstrated public expertise'; cite source ids",
  "system_prompt_md": "markdown system prompt (300-500 words): 'You are an AI coaching simulation built from
               {name}'s publicly expressed ideas...', how this coach thinks (from notes), voice and manner,
               'Using the background notes' section that forbids quoting them as real quotations"
RESEARCH NOTES:
{notes}
"""


async def llm_json(llm, model, prompt: str, max_tokens: int = 4000) -> dict:
    out = ""
    async for d in llm.stream([{"role": "system", "content": "Reply with valid JSON only."},
                               {"role": "user", "content": prompt}], model, max_tokens=max_tokens, temperature=0.2):
        out += d
    m = re.search(r"\{.*\}", out, re.S)
    return json.loads(m.group(0)) if m else {}


def fetch_text(url: str) -> tuple[str, str]:
    """Return (title, text) for a URL. YouTube -> transcript; otherwise readable article text."""
    yt = re.search(r"(?:youtu\.be/|v=|/shorts/)([A-Za-z0-9_-]{11})", url)
    if yt:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            api = YouTubeTranscriptApi()
            tr = api.fetch(yt.group(1))
            text = " ".join(s.text for s in tr)
        except Exception as e:  # noqa: BLE001
            return "", f"(no transcript available: {e})"
        return f"YouTube {yt.group(1)}", text
    import trafilatura
    html = trafilatura.fetch_url(url)
    if not html:
        return "", ""
    text = trafilatura.extract(html, include_comments=False) or ""
    md = trafilatura.extract_metadata(html)
    title = (md.title if md and md.title else url)
    return title, text


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


async def build(name: str, urls: list[str], model: str | None):
    llm = build_provider()
    if not llm.is_configured():
        sys.exit(llm.missing_config_message())
    model = model or settings.LLM_MODEL or llm.available_models()[0]
    cid = slug(name)
    cdir = settings.COACHES_DIR / cid
    (cdir / "knowledge").mkdir(parents=True, exist_ok=True)

    sources, claims, style_obs, bio = [], {}, [], []
    for i, url in enumerate(urls, 1):
        sid = f"src_{i:02d}"
        print(f"[{sid}] fetching {url}")
        title, text = fetch_text(url)
        if len(text) < 200:
            print(f"   skipped (no usable text)")
            continue
        sources.append({"id": sid, "title": title or url, "url": url, "type": "youtube" if "youtu" in url else "web",
                        "accessed": date.today().isoformat(), "chars": len(text)})
        text = text[:60000]
        try:
            res = await llm_json(llm, model, EXTRACT_PROMPT.format(name=name, sid=sid, topics=", ".join(TOPICS),
                                                                   title=title, url=url, text=text))
        except Exception as e:  # noqa: BLE001
            print(f"   extraction failed: {e}")
            continue
        for c in res.get("claims", []):
            t = c.get("topic") or "business principles"
            claims.setdefault(t, []).append(f"{c.get('claim', '').strip()} ({sid})")
        style_obs += [f"{s} ({sid})" for s in res.get("style_observations", [])]
        bio += [f"{s} ({sid})" for s in res.get("bio_facts", [])]
        print(f"   {len(res.get('claims', []))} claims")

    if not claims:
        sys.exit("No claims extracted; check the URLs.")

    for topic, items in claims.items():
        fn = cdir / "knowledge" / f"{slug(topic)}.md"
        body = [f"---\ntopic: \"{topic.title()}\"\ntags: [{', '.join(slug(topic).split('_'))}]\n"
                f"summary: \"{name}'s publicly expressed views on {topic}.\"\n---\n"]
        for j, it in enumerate(items, 1):
            if not it.startswith("["):
                it = "[DIRECT] " + it
            body.append(f"{j}. {it}\n")
        fn.write_text("\n".join(body))
    if bio:
        (cdir / "knowledge" / "biography-and-track-record.md").write_text(
            f"---\ntopic: \"Biography and track record\"\ntags: [biography, career]\nsummary: \"Facts about {name}'s life and career.\"\n---\n\n"
            + "\n".join(f"{j}. [DIRECT] {b}\n" for j, b in enumerate(bio, 1)))

    notes = "\n".join(f"- ({t}) {c}" for t, cs in claims.items() for c in cs) + "\nSTYLE:\n" + "\n".join(f"- {s}" for s in style_obs)
    synth = await llm_json(llm, model, SYNTH_PROMPT.format(name=name, notes=notes[:80000]), max_tokens=6000)
    prof = synth.get("profile", {})
    profile = {
        "id": cid, "name": name, "display_name": name,
        "tagline": prof.get("tagline", ""), "short_bio": prof.get("short_bio", ""), "roles": prof.get("roles", []),
        "expertise": prof.get("expertise", []), "not_expertise": prof.get("not_expertise", []),
        "signature_themes": prof.get("signature_themes", []),
        "disclaimer": f"AI simulation based on publicly available material. Not affiliated with or endorsed by {name}.",
        "research_date": date.today().isoformat(),
    }
    (cdir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    (cdir / "style.md").write_text(synth.get("style_md", "# Style\n\n" + "\n".join(f"- {s}" for s in style_obs)))
    (cdir / "system_prompt.md").write_text(synth.get("system_prompt_md",
        f"# Coach persona: {name} (AI simulation)\n\nYou are an AI coaching simulation built from {name}'s publicly expressed ideas."))
    (cdir / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    (cdir / "research_notes.md").write_text(f"# Research notes\n\nGenerated by add_coach.py on {date.today()} from "
                                            f"{len(sources)} sources using {llm.name}/{model}.\n")
    print(f"\nCoach '{name}' written to {cdir} ({sum(len(v) for v in claims.values())} claims, {len(sources)} sources).")
    print("Restart ./start.sh and it will appear in the coach dropdown. Review the files — this is a first draft.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--website", action="append", default=[])
    ap.add_argument("--youtube", action="append", default=[])
    ap.add_argument("--podcast", action="append", default=[])
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--model")
    a = ap.parse_args()
    name = a.name or input("Name: ").strip()
    urls = a.website + a.youtube + a.podcast + a.source
    if not urls:
        print("Enter sources one per line (website, YouTube URLs, podcast/transcript pages, articles). Empty line to finish.")
        for label in ("Website", "YouTube URLs", "Podcast URLs", "Other sources"):
            while True:
                u = input(f"{label}: ").strip()
                if not u:
                    break
                urls.append(u)
    if not name or not urls:
        sys.exit("Need a name and at least one source URL.")
    asyncio.run(build(name, urls, a.model))


if __name__ == "__main__":
    main()
