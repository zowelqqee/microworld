# Curated Gemini manual-review batch

Status: **human review completed; 10 relations are proposal-only**.

This bounded batch used 40 distinct stored official arXiv source records with
`gemini-3.1-flash-lite`. Requests were spaced by five seconds, for a maximum
of 12 RPM. It produced 67 raw triples; the unchanged node-quality filter,
evaluated with a proposal-local literal index for manual triage only, retained
54 review candidates and excluded 13 candidates. Human review accepted 10
relations (18.5%), including one new-predicate proposal, and rejected 44.

## Review workflow

1. `manual_review_decisions.json` records the returned human decisions for all
   54 candidates; `review_summary.md` gives the completed counts.
2. `manual_accepted_proposal_overlay.json` contains only the 10 manually
   accepted relations and remains proposal-only.
3. Do not treat this batch, the local literal index, or a repeated predicate
   as an automatic admission rule.

## Artifact map

- `source_batch.json` — deterministic source sentence selection and arXiv
  provenance.
- `raw_responses.json` — Gemini responses plus usage metadata, saved after
  every request.
- `manual_review.md` — the human annotation sheet.
- `manual_review_decisions.json` — completed human decisions for every review
  candidate.
- `review_summary.md` — audited accept/reject numbers and accepted relations.
- `manual_accepted_proposal_overlay.json` — manually accepted relations only;
  this is not a serving overlay.
- `manual_review_candidates.json` — same review rows in structured form, with
  intentionally blank manual fields.
- `node_quality_rejections.json` — excluded pre-review rows and the unchanged
  filter reasons.
- `learning_log.md` — post-review observational template, not an automation
  specification.
- `batch_manifest.json` — batch scope and proposal-only guardrails.

No precision gate was run, no serving or accepted overlay was written, and no
serving or accepted memory changed.
