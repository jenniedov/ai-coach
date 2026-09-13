"""Builds the message list for the LLM from the coach profile, retrieved knowledge and history."""
from __future__ import annotations

import time

from ..config import settings
from .loader import Coach
from .retrieval import CoachRetriever, format_context

GLOBAL_RULES = """
## Non-negotiable rules
- You are an AI coaching simulation built from {name}'s publicly expressed ideas. You are NOT {name}.
  If asked whether you are the real person, say plainly that you are an AI simulation and not affiliated
  with or endorsed by them. Otherwise don't keep bringing it up.
- Never present anything as an actual quotation from {name}. Speak in first person as the coach persona
  giving advice ("I'd push back on that...") rather than reporting ("{name} has said...").
- Use the background notes as the basis for your positions. If the notes don't cover the question, reason
  from the principles they do contain and make it clear you're extrapolating ("my instinct here would be..."),
  never claim a specific real-world opinion the notes don't support.
- Outside the coach's areas of demonstrated expertise (see profile), be honest about the limit and steer
  to what you can help with; don't give legal, medical, tax or investment instructions.
- This is a SPOKEN conversation. Answer in 3-7 sentences (about 20-60 seconds aloud). No lists, no
  headings, no markdown, no emojis. Short sentences. Contractions. One idea at a time.
- Be direct, practical, willing to disagree, and end with one sharp follow-up question when it helps the
  person move forward. Don't pad, don't recap the question, don't say "great question".
""".strip()


class CoachAgent:
    def __init__(self, coach: Coach, retriever: CoachRetriever):
        self.coach = coach
        self.retriever = retriever

    def system_prompt(self, context: str) -> str:
        p = self.coach.profile
        parts = [self.coach.system_prompt.strip() or f"You are a coaching simulation of {self.coach.name}.",
                 GLOBAL_RULES.format(name=self.coach.name)]
        if p.get("expertise") or p.get("not_expertise"):
            parts.append("## Expertise boundaries\nDemonstrated expertise: " + ", ".join(p.get("expertise", []))
                         + "\nNo demonstrated public expertise: " + ", ".join(p.get("not_expertise", [])))
        if context:
            parts.append("## Background notes retrieved for this question (public statements; 'direct' = "
                         "explicitly stated, 'inferred' = pattern across sources). Use them, don't recite them.\n"
                         + context)
        return "\n\n".join(parts)

    def build_messages(self, user_text: str, history: list[dict]) -> tuple[list[dict], list[dict], float]:
        t0 = time.perf_counter()
        prev_user = [m["content"] for m in history if m["role"] == "user"]
        results = self.retriever.search(user_text, history=prev_user)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        context = format_context(results)
        messages = [{"role": "system", "content": self.system_prompt(context)}]
        messages.extend(history[-settings.HISTORY_TURNS * 2:])
        messages.append({"role": "user", "content": user_text})
        debug = [{"id": c.id, "topic": c.topic, "evidence": c.evidence, "score": round(s, 3),
                  "text": c.text[:220]} for c, s in results]
        return messages, debug, retrieval_ms
