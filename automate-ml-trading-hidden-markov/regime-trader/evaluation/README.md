# Research agent evaluation

A small, manual-friendly benchmark for the Phase 10 research agent
(`research/`). It measures **research quality**, not investment advice — never
score which system gives the "better" recommendation.

## Workflow

1. Run the benchmark through this project's agent with whichever provider you want to
   evaluate (local Ollama is free; hosted providers spend quota, so start small):

   ```bash
   python -m evaluation.evaluator run --provider ollama --limit 3
   python -m evaluation.evaluator run --provider gemini --limit 3     # same questions, another backend
   ```

   Each result is saved to `evaluation/results/<provider>/<SYMBOL>_<n>.json`
   (`n` = the question's position in `research_questions.json`; git-ignored).

2. **Optional manual reference.** For the same question, open Google Finance
   Deep Search in your own browser (Google AI Pro) and paste/save the report to
   `evaluation/google_finance_reference/<SYMBOL>_<n>.txt` (same stem as the
   result file, `.txt`). This is done by hand: Google Finance is **never**
   scraped or browser-automated, and nothing here depends on it.

3. Score:

   ```bash
   python -m evaluation.evaluator score            # every provider, side by side
   python -m evaluation.evaluator score --provider ollama
   ```

## What is measured

| Metric | How |
|---|---|
| Source coverage | `source_count`, distinct evidence domains |
| Source freshness | median age of evidence that carries a publication date (and the fraction that does) |
| Citation completeness | evidence items with a source URL + title (always 1.0 by construction — citations come from grounding metadata) |
| Schema completeness | summary/evidence/queries/sources/unknowns/stance present |
| Contradiction detection | `conflicts_reported`, `unknowns_reported` |
| Unsupported-claim handling | `dropped_unsupported_claims` (model claims whose cited source was not in the grounded set) |
| Latency / cost | `latency_ms`, `gemini_calls` |
| Reference overlap (if a reference exists) | share of the reference's source domains this agent also found |

## Human rubric (not automated)

Read both reports side by side and note, per question:

- **Material-event recall** — of the material events in either report, how
  many did each find? (List the events, tick each system.)
- **Unsupported-claim rate** — claims with no verifiable source / total claims.

Keep the benchmark small; expand it only to answer a specific limitation you
observed.
