# Coaches

Each coach is a folder:

```
coaches/<id>/
  profile.json       identity, bio, expertise / not-expertise, disclaimer, default engine + voices
  system_prompt.md   behavioural prompt (a simulation of publicly expressed ideas, never impersonation)
  style.md           communication style notes
  sources.json       cited public sources (id, title, url, type, date)
  knowledge/*.md     topic files: frontmatter (topic, tags, summary) + numbered [DIRECT]/[INFERRED] claims
```

`example_coach/` is a generic placeholder so the app runs out of the box. Build real coaches with
`uv run python add_coach.py` (from public URLs you point it at); they appear in the sidebar on the
next start. Coach folders other than the example are git-ignored: research about real people stays on
your machine unless you choose to publish it.
