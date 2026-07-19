# Grouped Gemini manual-review batch

Status: **manual review complete; 36 proposal-only relations selected.**

This batch contains 120 bounded sentences from 69 distinct stored official
arXiv records (at most two sentences from one record). Gemini 3.1 Flash Lite
returned 191 raw triples. The unchanged node-quality filter, evaluated with
the same proposal-local literal index used by the prior curated batch, retained
148 review candidates and excluded 43.

## Grouping is a review aid, not an admission rule

Every retained candidate was manually reviewed. The returned decision encoding
is in `manual_review_decisions.json`; the original grouped review sheet remains
an unmodified review template. Grouping did not auto-reject or auto-accept.

Primary review groups are mutually exclusive to avoid duplicate rows; the
`group_labels` field in `candidate_grouping.json` retains overlaps. Primary
group priority is anaphora, temporal referent, generic property, attachment
shape, then clean/no-flag.

| Primary group | Candidates |
|---|---:|
| Anaphora-likely | 15 |
| Temporal-referent-likely | 1 |
| Generic-property-likely | 51 |
| Attachment-shape-flag | 1 |
| Clean / no-flag | 80 |

Overlapping label counts: anaphora 15, temporal 1, generic-property 58, and
attachment-shape 1. Thus the grouping visibly concentrates 68 of 148 review
candidates (45.9%) into a known failure-pattern primary group, while 80
remain clean/no-flag and require normal careful review.

## Explicit heuristics

- **Anaphora-likely:** subject or object contains `it`, `this`, `that`,
  `they`, or `their`.
- **Temporal-referent-likely:** the source/candidate contains an explicit
  date/time reference (year, date-related word, or non-modal month name) and
  the subject lacks a clear capitalized entity anchor.
- **Generic-property-likely:** predicate begins with an explicit generic
  attribute/activity list: `is`, `are`, `was`, `were`, `has`, `have`,
  `show(s)`, `indicate(s)`, `means`, `offer(s)`, `provide(s)`, `support(s)`,
  or `enable(s)`.
- **Attachment-shape-flag:** source sentence has a colon followed by a
  comma-list, matching the existing node-quality list-context pattern.
- **Clean/no-flag:** no rule above matched.

These are transparent labels from the prior learning log, not evidence that a
candidate is correct or incorrect.

## Artifact map

- `manual_review.md` — grouped review sheet; use it for batch decisions with
  explicit per-row exceptions.
- `candidate_grouping.json` — primary group, all matching labels, and reasons
  for each candidate.
- `manual_review_candidates.json` — complete candidate data with blank manual
  decision fields.
- `node_quality_rejections.json` — 43 candidates removed by the unchanged
  filter before review.
- `manual_review_decisions.json` — completed decision ledger for all 148
  candidates.
- `review_summary.md` — audited outcomes and observations.
- `manual_accepted_proposal_overlay.json` — the 36 manual accepts only,
  explicitly proposal-only; resolves full relation data from
  `manual_review_candidates.json`.
- `learning_log.md` — observations from the completed review.
- `source_batch.json` and `raw_responses.json` — stored official provenance
  and resumable Gemini output.

Requests were spaced by five seconds (maximum 12 RPM). No precision gate ran,
no serving memory changed, and no commit was created.
