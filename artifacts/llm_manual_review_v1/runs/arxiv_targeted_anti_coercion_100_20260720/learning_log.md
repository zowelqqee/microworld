# Manual-review learning log — LLM extraction v1

Status: **PENDING HUMAN REVIEW.** Do not derive or deploy an automated admission rule from this batch. Add observations only after the reviewer has completed `manual_review.md`.

Batch: `arxiv_targeted_anti_coercion_100_20260720`
Model: `gemini-3.1-flash-lite`
Review candidates before human review: 14

## Pre-review grouping distribution

These are review-order labels, not decisions. Primary groups are exclusive; flag counts may overlap.

| Primary review group | Candidates |
|---|---:|
| Anaphora-likely | 0 |
| Temporal-referent-likely | 1 |
| Generic-property-likely | 13 |
| Attachment-shape-flag | 0 |
| Clean / no-flag | 0 |

| Flag label | Candidates |
|---|---:|
| Anaphora-likely | 0 |
| Temporal-referent-likely | 1 |
| Generic-property-likely | 14 |
| Attachment-shape-flag | 0 |

## Post-review summary (fill after review)

| Metric | Count |
|---|---:|
| reviewed candidates |  |
| manually accepted proposals |  |
| manually rejected candidates |  |
| accept rate |  |

## Repeated legitimate patterns

| Candidate IDs | Entity shape/type | Predicate surface | Source characteristic | Reviewer evidence | Notes |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Repeated false-positive patterns

| Candidate IDs | Entity/predicate shape | Filter limitation observed | Reviewer rejection reason | Possible future blocklist hypothesis (not a rule) |
|---|---|---|---|---|
|  |  |  |  |  |

## Guardrail

This log is observational. It does not authorize automatic admission, changes to `node_quality_filter.py`, changes to the precision gate, or writes to serving memory.
