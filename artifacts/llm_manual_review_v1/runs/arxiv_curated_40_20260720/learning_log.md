# Manual-review learning log — LLM extraction v1

Status: **updated from completed human review.** Do not derive or deploy an automated admission rule from this batch.

Batch: `arxiv_curated_40_20260720`
Model: `gemini-3.1-flash-lite`
Reviewed candidates: 54
Manually accepted proposals: 10
Manually rejected candidates: 44

## Post-review summary (fill after review)

| Metric | Count |
|---|---:|
| reviewed candidates | 54 |
| manually accepted proposals | 10 |
| manually rejected candidates | 44 |
| accept rate | 18.5% |

## Repeated legitimate patterns

| Candidate IDs | Entity shape/type | Predicate surface | Source characteristic | Reviewer evidence | Notes |
|---|---|---|---|---|---|
| arxiv-006:0, arxiv-016:0, arxiv-025:0, arxiv-028:1, arxiv-038:0 | named research system/software/method + bounded concrete object | `uses` | direct subject-led capability/dependency sentence | literal support and clean entities manually confirmed | 5 accepts; this is an observation, not an automatic criterion |
| arxiv-011:1 | named system + named system | `extends` | direct two-system relation in a subject-led sentence | literal support and clean entities manually confirmed | one accepted existing-predicate relation |
| arxiv-003:0, arxiv-006:1, arxiv-025:1 | bounded technical/quantitative node pairs | `is`, `is used in` | direct apposition or numeric/domain statement | literal support and clean entities manually confirmed | three accepted descriptive relations |
| arxiv-011:0 | named system + named system | `builds upon` | direct two-system relation in a subject-led sentence | literal support and clean entities manually confirmed | one new-predicate proposal; one example is insufficient for automation |

## Repeated false-positive patterns

| Candidate IDs | Entity/predicate shape | Filter limitation observed | Reviewer rejection reason | Possible future blocklist hypothesis (not a rule) |
|---|---|---|---|---|
| arxiv-002:0, arxiv-005:0, arxiv-027:0, arxiv-031:0–3 | temporary/anaphoric subject | mixed | existing filter did not block every discourse referent | not stable graph entity | future blocklist hypothesis only: discourse/anaphora detector |
| arxiv-001:0, arxiv-007:0, arxiv-008:0–1, arxiv-009:0–1, arxiv-010:0–2, arxiv-014:0–1, arxiv-015:0–1, arxiv-017:0–1, arxiv-020:0, arxiv-021:0–1, arxiv-022:0, arxiv-033:0, arxiv-034:0–037:0 | generic activity, property, class, or outcome endpoint | mixed | generic-head rules do not cover every abstract or broad phrase | not a durable graph node | future blocklist hypothesis only: broader abstract/activity taxonomy |
| arxiv-002:1, arxiv-016:1, arxiv-028:2, arxiv-033:1 | process, participial clause, infinitive/goal, or non-relational attachment | `created by`, `providing`, `generate`, `manages` | literal wording did not preserve a valid atomic graph relation | process/attachment invalid after review | future blocklist hypothesis only: clause/attachment shape checks |

## Guardrail

This log is observational. It does not authorize automatic admission, changes to `node_quality_filter.py`, changes to the precision gate, or writes to serving memory. The 10 accepted relations live only in `manual_accepted_proposal_overlay.json` with `proposal_only=true` and `safe_for_general_runtime=false`.
