# Targeted extraction pilot v1 — completed manual review

## Prompt and scope

The frozen targeted prompt is in `prompt.md` and was run as profile
`targeted_class_member_named_system_v1` with Gemini 3.1 Flash Lite. It asks
only for class/member relations and named technical-system properties, and
instructs the model to return `[]` rather than force another relation type.

The general extraction prompt was not changed. Node-quality filtering and all
grouping heuristics were unchanged. This lane is manual-review-only.

## Fresh material and overlap result

| Field | Result |
|---|---|
| batch | `arxiv_targeted_prompt_100_20260720` |
| source | official arXiv API, bounded page 1 of the `computer science` query |
| source sentences | 100 |
| source-record cap | 1 sentence per record |
| distinct source records | 100 |
| excluded prior source records | 143 |
| overlap with 120/40/15 source set | 0 |
| overlap with 74-sentence holdout | 0 |
| Gemini spacing | 5 seconds (12 RPM maximum) |

The initial official API page after exclusions supplied only 64 candidate
sentences. A second bounded API page supplied 100; no overlapping or
lower-quality substitute was used.

## Extraction and triage result

| Stage | Count |
|---|---:|
| source sentences | 100 |
| raw Gemini triples | 59 |
| unchanged node-quality exclusions | 14 |
| candidates ready for manual review | 45 |
| manual verdicts recorded | 45 |
| manually accepted proposals | 34 |
| manually rejected candidates | 11 |
| acceptance rate | **75.6%** |

Primary review groups: anaphora 1, temporal 1, generic-property 41,
clean/no-flag 2, attachment 0. These labels remain review-order aids, not
automated decisions.

The restrictive prompt produced fewer raw triples per sentence than earlier
general batches. That is an expected yield effect from allowing empty output.
All retained candidates have now received a human verdict.

## Manual-review result and comparison

The completed decision encoding is `manual_review_decisions.json`. The 34
manual accepts are present only in `manual_accepted_proposal_overlay.json`.

| Comparison | Acceptance rate | Interpretation |
|---|---:|---|
| Targeted prompt pilot | **34/45 = 75.6%** | Wilson 95% CI: 61.3%–85.8%. |
| Prior general batch | 36/148 = 24.3% | Targeted versus general: one-sided Fisher p = 8.3e-10. |
| Holdout generic-property subgroup | 21/36 = 58.3% | Targeted point estimate is higher; one-sided Fisher p = 0.079, so this single comparison is not conclusive at 0.05. |

The pilot clears the requested practical bar of 58%+ and strongly exceeds the
general-batch baseline. It is evidence that the targeted prompt can create a
high-yield manual-review queue, but it is still one prompt run on one source
page. A second independent targeted-prompt batch is needed before claiming a
systematic advantage rather than a source-page effect.

## Guardrail

Nothing was auto-admitted or promoted; precision gate and serving memory are
unchanged, and no commit was created.
