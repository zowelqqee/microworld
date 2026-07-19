# OpenAI LLM-assisted relation-extraction pilot v1

## Scope and boundary

This is a 15-sentence, stored-arXiv-material pilot on the same difficult
construction family used by the grammar pilot. `gpt-4o-mini` was used only as a
stateless text-to-JSON extractor with Structured Outputs, temperature 0, and a
fixed 5-second pause between requests. It has no memory, no reasoning role, and
no access to the production answer path. No triple was promoted, and the
existing precision gate was not changed or bypassed.

## Measured run

| Measure | Result |
|---|---:|
| Source sentences / completed calls | 15 / 15 |
| Raw triples | 33 |
| Literal-support failures | 7 / 33 (21.2%) |
| Generic or self-reference endpoint rejects | 31 / 33 (93.9%) |
| Clean, literal, existing-predicate candidate | 1 / 33 (3.0%) |
| Clean, literal, new-predicate proposal | 1 / 33 (3.0%) |
| Existing precision gate, raw output admitted | 0 / 33 |
| Existing precision gate, raw output quarantined | 33 / 33 |
| Grammar pilot net-new accepted groups on the same construction family | 0 |

The full row-by-row audit, including every source-aligned triple, is in
`manual_review.md`; raw model output and OpenAI usage counters are in
`raw_responses.json`. `precision_gate_outcome.json` records the exact unchanged
validator result: all 33 raw candidates were quarantined (all had
`missing_explicit_evidence`; 29 also had `ambiguous_relation`). No manual
normalization or repair was fed to that gate.

## Cost

Measured usage was 2,539 input and 1,298 output tokens (3,837 total). At the
then-published `gpt-4o-mini` standard rates of $0.15/M input and $0.60/M output
tokens, this run cost approximately **$0.00116**. Recheck provider pricing
before extrapolating to a larger run.

## Result and gate decision

The pilot does demonstrate a construction-recognition gain: `SciServer builds
upon SkyServer` is a clean new-predicate proposal where the grammar pilot
produced no accepted new group. The dominant rejection mode is endpoint quality:
generic, authorial, or non-resolvable surfaces account for 31 of 33 outputs.
There are also seven structural failures—wrong semantic role, attachment error,
passive-agent hallucination, or nominal-list completion—so the strict
hallucination rate is **21.2%**, above the predeclared `<10%` gate. Only one of
33 raw triples is a clean new-predicate proposal.

**Verdict: honest stop.** Do not broaden the pilot, promote any candidate, or
modify the precision gate on this evidence. The next legitimate experiment, if
desired, is a prompt/schema change that forbids self-reference and requires
verbatim subject/object spans, followed by a fresh held-out pilot—not relaxed
review criteria or automatic admission.
