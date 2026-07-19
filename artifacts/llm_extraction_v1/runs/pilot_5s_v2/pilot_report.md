# Gemini 2.5 Flash extraction pilot v2 — paid key, 5-second pacing

## Scope

Gemini was used only as a stateless text-to-JSON extractor on 15 stored arXiv
sentences from the grammar-pilot construction family. The run used temperature
0, a fixed 5-second delay between requests, and persisted each response. It has
no memory, no reasoning role, and no access to the serving or answer path.

## Results

| Measure | Result |
|---|---:|
| Completed calls / source sentences | 15 / 15 |
| Raw triples | 36 |
| Literal-support failures | 6 / 36 (16.7%) |
| Unclean/generic endpoint rejects | 33 / 36 (91.7%) |
| Clean, literal, new-predicate proposals | 1 / 36 (2.8%) |
| Existing precision gate: raw admitted | 0 / 36 |
| Existing precision gate: raw quarantined | 36 / 36 |
| Grammar pilot net-new accepted groups on this construction family | 0 |

The complete row-by-row audit is in `manual_review.md`. The unchanged validator
outcome is in `precision_gate_outcome.json`: every raw candidate was
quarantined, with `missing_explicit_evidence` on all 36 and
`ambiguous_relation` on 32. There was no predicate normalization, manual
repair, or overlay write before that check.

## Interpretation and decision

Gemini yields somewhat better raw structure than the OpenAI pilot: it separates
`SciServer builds upon SkyServer` and `SciServer extends SkyServer`, and it
captures the valid existing-schema `AutoSlim uses Random Forest classification`
relation. It also has the same material failure classes: generic endpoints,
wrong semantic agent, attachment ambiguity, and nominal-list completion.

The predeclared hallucination gate is `<10%`; this run measures **16.7%**.
Therefore the verdict is **honest stop**: do not promote a candidate, broaden
the source lane, or relax the precision gate. A future experiment must use a
new held-out pilot with a stricter extraction contract (verbatim subject/object
spans, no nominal-list inference, no authorial subjects) and be re-audited from
scratch.
