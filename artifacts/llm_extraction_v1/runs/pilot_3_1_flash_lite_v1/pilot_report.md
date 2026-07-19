# Gemini 3.1 Flash-Lite extraction pilot v1

## Scope

This is a separate 15-sentence stored-arXiv pilot using the stable
`gemini-3.1-flash-lite` API model, temperature 0, structured JSON output, and a
fixed 5-second inter-request delay. The LLM is only a stateless extractor: it
does not participate in reasoning, serving, or answer generation. No candidate
was promoted and no precision-gate logic changed.

## Results

| Measure | Result |
|---|---:|
| Completed calls / source sentences | 15 / 15 |
| Raw triples | 32 |
| Literal-support failures | 4 / 32 (12.5%) |
| Unclean/generic endpoint rejects | 29 / 32 (90.6%) |
| Clean, literal, new-predicate proposals | 1 / 32 (3.1%) |
| Existing precision gate: raw admitted | 0 / 32 |
| Existing precision gate: raw quarantined | 32 / 32 |

`manual_review.md` contains the row-level review. The raw, unchanged validator
outcome is in `precision_gate_outcome.json`: it quarantined every candidate
(`missing_explicit_evidence` on 31, `ambiguous_relation` on 27). No predicate
canonicalization, manual repair, or overlay write preceded the check.

## Verdict

Gemini 3.1 Flash-Lite is operationally a much better fit for this low-volume
pilot than Gemini 2.5 Flash on the current quota: the entire run completed with
no 429 responses. Its raw structural precision is also better than the Gemini
2.5 Flash v2 run (12.5% versus 16.7% literal-support failures), including the
removal of the prior wrong-agent `Random Forest classification prunes…` output.

However, the predeclared gate is `<10%`, and 12.5% does not pass. **Honest
stop:** do not broaden, promote, or weaken the production precision gate. A
future fresh held-out pilot may test a stricter contract that disallows
authorial subjects and nominal-list inference and requires each subject/object
to be an exact text span.

## Cross-model finding

Across the OpenAI, Gemini 2.5 Flash, and Gemini 3.1 Flash-Lite pilots, the
same bottleneck dominates: all models can recover many literal relations, but
about 90% of raw candidates are rejected because one or both endpoints are
generic, authorial, event-like, or otherwise unsuitable as durable graph nodes.
The next research step should therefore target deterministic entity
normalization and node-quality verification—not another model swap or a weaker
precision gate.
