# Targeted anti-coercion v1 — pre-review report

## Frozen intervention

This run uses the prior targeted class/member + named-system-property prompt,
with the anti-coercion addendum frozen in
[`prompt.md`](prompt.md). It requires an exact, literal predicate and
direction, and explicitly prohibits converting possessives, purpose clauses,
and satellite/incidental events into class/member or system-property claims.
No other prompt or node-quality-filter change was made.

## Independent material

- Run: `arxiv_targeted_anti_coercion_100_20260720`
- Source: official arXiv API, computer-science query, one sentence per source
  record.
- Source sentences / distinct source records: **100 / 100**.
- Explicit source-URL overlap checks: **0** with each of
  `arxiv_grouped_120_20260720`, `arxiv_holdout_74_20260720`, and
  `arxiv_targeted_prompt_100_20260720`.
- Model: Gemini 3.1 Flash Lite; 5-second request interval (maximum 12 RPM).

## Pre-review extraction result

| Stage | Count |
|---|---:|
| Raw Gemini triples | 20 |
| Excluded by unchanged node-quality triage | 6 |
| Candidates awaiting manual review | **14** |
| Generic-property-likely primary group | 13 |
| Temporal-referent-likely primary group | 1 |
| Anaphora / attachment / clean-no-flag primary groups | 0 / 0 / 0 |

The ready-to-label ledger is
[`manual_review.md`](../runs/arxiv_targeted_anti_coercion_100_20260720/manual_review.md).
Raw output and structured candidate evidence remain in the same run directory.

## Completed manual-review gate

The user reviewed every candidate: **11 ACCEPT / 3 REJECT = 78.6%**. The
decision ledger is
[`manual_review_decisions.json`](../runs/arxiv_targeted_anti_coercion_100_20260720/manual_review_decisions.json).
The 11 verified relations are recorded only in the explicitly non-serving
[`manual proposal overlay`](../runs/arxiv_targeted_anti_coercion_100_20260720/manual_accepted_proposal_overlay.json).

| Comparison | Previous targeted v1 | Anti-coercion v1 | Reading |
|---|---:|---:|---|
| Overall manual acceptance | 34/45 = 75.6% | 11/14 = **78.6%** | Point estimate is consistent with replication; anti-coercion v1 is too small (n=14) to establish an improvement. |
| Relation-coercion family among rejects | 10/11 = 90.9% | 2/3 = **66.7%** | Direction is favorable, but only three rejects exist; it cannot establish addendum effectiveness. |

### Rejected-case classification

| ID | Review finding | Classification |
|---|---|---|
| `arxiv-034:0` | `extension of` was converted to `is a type of`. | Relation coercion: semantic relation mismatch. |
| `arxiv-020:0` | A method used to achieve/develop something was converted to `is a type of`. | Relation coercion: contextual/method relation mismatch. |
| `arxiv-068:0` | `√iSWAP` in the text was normalized to the different subject `iSWAP`. | Literal endpoint substitution; an exactness error, but not the prior coercion family. |

### Verdict

**Replication: provisionally confirmed.** The 78.6% result is close to the
earlier 75.6% on source-disjoint material, while its 14-candidate denominator
is necessarily imprecise.

**Addendum effectiveness: promising but unconfirmed.** The coercion share fell
from 90.9% to 66.7%, yet that comparison is based on only 3 new rejects. It is
an observation to carry into another independent review, not evidence for a
production-prompt change or automated admission rule.

Nothing was auto-admitted. The accepted rows are proposal-only; precision gate,
serving memory, and promoted overlays remain unchanged. No commit was made.
