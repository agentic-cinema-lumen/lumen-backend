# Demo film (out-of-corpus) — for us, not for the judge

**Title: Gangs of New York (2002), dir. Martin Scorsese.**
Screenplay: IMSDb, 3rd draft (1993) by Jay Cocks, Steven Zaillian, Kenneth Lonergan.
Source: https://imsdb.com/scripts/Gangs-of-New-York.html
Stills: https://film-grab.com/2014/08/04/gangs-of-new-york/ (10 frames, evenly sampled)

Not one of the 100 films in `data/movies/movies_manifest.json`. The model has never
seen it.

## Actual vs corpus

| Number | Value |
|---|---|
| Actual IMDb rating | **7.5** (tt0217505, ~397k votes) |
| Corpus mean rating | **7.80** |
| Actual theatrical runtime | 167 min |
| Parsed runtime from this draft | 194 min |

## Why this film

Two craft debates a judge recognises:
1. **Excessive runtime.** The film was cut down from a much longer assembly and still
   runs 167 minutes. Reviewers called it bloated. The parser reads this draft at
   194 minutes.
2. **Dark, murky period photography.** The gaslit Five Points interiors read as
   crushed blacks on consumer displays. The vision pass measures a 0.50 dark-frame
   ratio, which trips the HIGH cinematography flag.

## Hidden-title demo instructions

Run it with the title replaced by "Untitled" and a paraphrased logline. The IMSDb
title header (8 lines) has already been stripped from `script.txt`, so the file
starts at the first slugline.

```
LUMEN_AGENT_MODEL=gemini-3.7-flash ./venv/bin/python3 scripts/run_premortem.py \
  --script data/demo/gangs_of_new_york/script.txt \
  --keyframes data/demo/gangs_of_new_york/keyframes \
  --title "Untitled" \
  --logline "<paraphrased logline below>" \
  --export data/demo/gangs_of_new_york/report.json
```

Then reveal the title and compare the projected rating against 7.5 and against the
corpus mean of 7.80.

**Leakage caveat to state out loud:** the screenplay body still contains its own
character names. A reader who knows the film can identify it from the text. We hide
the title from the *prompt*, not from the *script*. Judge the leakage question on
whether the research agent's claims name the film, not on whether the text could.

## Paraphrased logline (for the story field — no names, no title)

> A young man returns to the immigrant slum where his father was killed in a
> street battle two decades earlier, and works his way into the confidence of the
> warlord who did it. His revenge ripens while the city around him boils over into
> a riot against a wartime draft.

## Run results (2026-09-06)

Two runs. Both hid the title.

`report.json` — `LUMEN_AGENT_MODEL=gemini-3.7-flash`. **Degraded.** The research and
synthesis agents both failed on the Gemini API (503 UNAVAILABLE, then
RESOURCE_EXHAUSTED) across six attempts. Genre came back `None` and no research
claims or prose were produced. The deterministic half still ran in full.

`report_flash_lite.json` — `LUMEN_AGENT_MODEL=gemini-3.1-flash-lite`. Complete
except for search: `PARALLEL_API_KEY` is empty in `.env`, so the search results are
mock output, not live retrieval. Use this one for the demo.

Numbers, identical in both runs (the deterministic half never touches the LLM):

| Item | Value |
|---|---|
| Projected rating | **8.09** / 10 |
| Genre prior | 7.83 |
| Craft residual | +0.26 |
| Model error | cv_mae ±0.435, cv_r2 0.145 |
| Actual IMDb | **7.5** |
| Corpus mean | 7.80 |
| Genre inferred by the agent | Historical Crime Drama (correct, title hidden) |
| Risk flags | 1 — Cinematography HIGH, 50% of frames sub-40 luminance |
| Counterfactuals | all 15 rows inside the noise floor |

The model overshoots by 0.59, which is 1.36x its own cv_mae. The residual of +0.26
is smaller than the ±0.435 error bar, so the pipeline's own honest reading is
"indistinguishable from the genre prior". Say that out loud rather than claiming a
hit.

**Leakage: none.** The three research claims are generic craft precedents about dark
cinematography and trope fatigue, sourced to a television lighting controversy and a
tropes article. No claim names the film, its director, its year, or its setting. The
agent's own search queries paraphrase the logline back ("films about immigrant slums
warlord revenge wartime draft riots") rather than naming the title.
