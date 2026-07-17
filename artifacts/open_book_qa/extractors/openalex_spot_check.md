# OpenAlex source-specific extractor: pre-diagnostic and spot-check

**Scope:** proposal-only re-parsing of the existing OpenAlex quarantine slice.
No accepted memory, promoted overlay, or serving memory was modified.

The lane has seven unique relations for seven held-out-pool subjects, so the
manual sample is the full lane (deterministic seed `20260718`). Predicate
novelty was checked before extraction: all seven predicates already existed for
their subject in the clean relation baseline.

| # | Candidate | Pre-extraction novelty | Final parser verdict |
|---:|---|---|---|
| 1 | Meltwater flushing through cracks — enables — organic burial and submarine deposition of airborne volcanic ash | duplicate | reject: object span is not bounded |
| 2 | Every economic activity — supports — considerable diversity of talent and significant inequality… | duplicate | reject: object span is not bounded |
| 3 | Nitrogen fertilizer production — uses — large amounts of natural gas and some coal | duplicate | reject: object span is not bounded |
| 4 | Textural evidence also — supports — interpretation of a metasomatic-hydrothermal origin | duplicate | reject: discourse subject / non-atomic endpoint |
| 5 | GLiM — enables — regional analysis of Earth surface processes at global scales | duplicate | reject: object span is not bounded |
| 6 | Seminal — works_by — Susan Boynton, Kenneth Levy, Peter Jeffery | duplicate | reject: fragmentary subject |
| 7 | OpenMEE also — supports — data importing and exporting, exploratory data analysis… | duplicate | reject: discourse subject / non-atomic endpoint |

There were no false accepts. Three source sentences express a relation but use
an unbounded object span; this lane intentionally keeps them quarantined rather
than silently shortening or rewriting the fact. The remaining four are
non-referential or fragmentary candidates. The full result is in
[`openalex_full_run_report.json`](openalex_full_run_report.json).
