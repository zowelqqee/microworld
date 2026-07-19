# Constrained-creative v1 — build notes

The design recommended "proceed to build," so this is the scoped v1. Isolated research path:
no production module modified (only new files added).

## What was built

- `worldpgt/reasoning/constrained_creative_v1.py`:
  - `ConstraintSpec` / `Fact` — the bounded fact set for a subject.
  - `select_facts(overlay, subject, n)` — deterministic selector from the overlay slice (dedupes).
  - `generate_constrained(spec)` — v1 generator: template + connectives. Includes every fact and
    asserts nothing else **by construction**; creative freedom is limited to connective/ordering.
  - `verify(text, spec)` — **the shared post-hoc constraint verifier**: inclusion, fidelity
    (right-attachment proxy), and hallucination (extra-content-token proxy). Applied identically
    to our output and to an LLM's — the symmetry is the point.
  - `proxy_fluency(text, attested_trigrams)` — labelled proxy from corpus trigrams.
- `worldpgt/tests/test_constrained_creative_v1.py` — 10 tests, all passing. Critically, the
  verifier tests prove it **detects a missing fact, a mis-attached fact, and hallucinated
  content** — it is a real instrument, not a rubber stamp.
- `worldpgt/experiments/run_constrained_creative_v1.py` — runs over the real overlay; writes
  `build_v1_results.json`; emits the exact LLM prompt per subject and scores a dropped-in
  `qwen_outputs.json` with the same verifier if present.

## Result (MicroWorld side, 5 subjects, N=3)

| Metric | Value | Interpretation |
|---|---:|---|
| mean inclusion | **1.00** | every required fact present |
| mean fidelity | **1.00** | every fact correctly attached |
| mean hallucination | **0.00** | nothing beyond the fact set |
| mean proxy fluency | **0.00** | template output over graph vocab is unattested in the literary corpus |

This is **exactly the trade-off the design predicted**: constraint adherence maxed, proxy fluency
floored. It is not a bug — it is the hypothesis. The point of the A/B is to see whether an LLM,
given the same constrained prompt and scored by the same verifier, trades the other way
(higher fluency, but non-zero hallucination / <1.0 inclusion).

## The LLM comparison is wired but NOT run (no model in this session)

`run_constrained_creative_v1.py` prints the constrained prompt per subject and will score a
`qwen_outputs.json` (subject → text) with the identical verifier. Running
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` over these prompts is the remaining step to complete
the A/B — deferred because no model runs here. The measurement instrument is ready; only the
generation call is pending.

## Honest limitations (v1, not blocking)

- **Verifier is surface-token based.** Inclusion/fidelity/hallucination are proxies, labelled as
  such (the design's §3 caveat). Fidelity is a "same-sentence-as-subject" heuristic; hallucination
  counts extra content tokens. These can mis-score paraphrase (an LLM saying "builds" for
  "develops" would read as a missing fact) — a real limitation to note when interpreting the LLM
  side. A paraphrase-aware verifier is future work.
- **Generator is deliberately non-fluent** (template). Genuine constrained *recombination* (the
  harder, more interesting fluency comparison) is explicitly out of scope for v1, per the design.
- **Proxy fluency of 0.0** partly reflects that graph vocabulary (proper nouns) rarely appears in
  the literary trigram corpus; it is a floor, and mainly meaningful as a relative comparison to
  the LLM's score on the same metric, not as an absolute.

## No open architectural questions

Unlike the reflective core, this build surfaced no fork requiring your decision. The one judgment
call — surface-token verifier vs paraphrase-aware — is a known limitation with a clear future-work
path, not an ambiguous fork; noted above, not escalated.
