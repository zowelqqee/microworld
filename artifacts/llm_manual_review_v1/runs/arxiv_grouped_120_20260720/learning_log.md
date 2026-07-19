# Manual-review learning log — LLM extraction v1

Status: **HUMAN REVIEW COMPLETE.** Do not derive or deploy an automated admission rule from this batch.

Batch: `arxiv_grouped_120_20260720`
Model: `gemini-3.1-flash-lite`
Review candidates before human review: 148

## Pre-review grouping distribution

These are review-order labels, not decisions. Primary groups are exclusive; flag counts may overlap.

| Primary review group | Candidates |
|---|---:|
| Anaphora-likely | 15 |
| Temporal-referent-likely | 1 |
| Generic-property-likely | 51 |
| Attachment-shape-flag | 1 |
| Clean / no-flag | 80 |

| Flag label | Candidates |
|---|---:|
| Anaphora-likely | 15 |
| Temporal-referent-likely | 1 |
| Generic-property-likely | 58 |
| Attachment-shape-flag | 1 |

## Post-review summary

| Metric | Count |
|---|---:|
| reviewed candidates | 148 |
| manually accepted proposals | 36 |
| manually rejected candidates | 112 |
| accept rate | 24.3% |

## Repeated legitimate patterns

| Candidate IDs | Entity shape/type | Predicate surface | Source characteristic | Reviewer evidence | Notes |
|---|---|---|---|---|---|
| `arxiv-018:0`, `arxiv-018:1`, `arxiv-019:0`, `arxiv-021:0`, `arxiv-027:0`, `arxiv-042:0`, `arxiv-048:1`, `arxiv-066:0`, `arxiv-075:0`, `arxiv-085:1`, `arxiv-091:0`, `arxiv-101:0`, `arxiv-103:0`, `arxiv-107:0`, `arxiv-117:0`, `arxiv-117:1` | Named systems / methods with a specific technique, architecture, or method endpoint | `uses`, `extends`, `builds upon`, `performs` | Direct declarative arXiv sentence naming both endpoints | Literal support and clean endpoints confirmed | Strongest recurring admitted shape; observation only. |
| `arxiv-005:0`, `arxiv-009:1`, `arxiv-018:3`, `arxiv-069:1`, `arxiv-088:0`, `arxiv-092:0` | Bounded quantity, apposition, named-data service, or physical mechanism | `is`, `has been serving`, `is enabled by` | Explicit identification / bounded statement | Literal support and clean endpoints confirmed | Less frequent than named-system `uses` relations. |

## Repeated false-positive patterns

| Candidate IDs | Entity/predicate shape | Filter limitation observed | Reviewer rejection reason | Possible future blocklist hypothesis (not a rule) |
|---|---|---|---|---|
| All 15 anaphora candidates; `arxiv-054:0`; `arxiv-015:0` | Pronouns/demonstratives, temporal value without stable anchor, or aggregate description | Various | A literal span can still lack a graph-resolvable endpoint | Unresolved anaphora / temporary or aggregate referent | Reconfirm as review-order labels, never auto-reject. |
| `arxiv-002:0`, `arxiv-012:0`, `arxiv-012:1`, `arxiv-019:1`, `arxiv-021:1`, `arxiv-050:0`, `arxiv-060:0`, `arxiv-081:0`, `arxiv-086:0`, `arxiv-087:0`, `arxiv-095:0`, `arxiv-097:0`, `arxiv-104:0`, `arxiv-112:0`, `arxiv-113:0` | Generic capability, property, activity, or long descriptive phrase | Mostly generic predicate surfaces | Node-quality filter does not remove all generic endpoints | Non-resolvable generic node | Generic-property label retains 13 accepts, so it is unsuitable as a blocklist rule. |
| `arxiv-028:0`, `arxiv-028:1`, `arxiv-063:0`, `arxiv-078:1`, `arxiv-116:0`, plus 9 clean-group rows | Predicate/attachment semantic mismatch | Various | Surface extraction relation not supported by the full clause | No literal relation support | Keep literal-evidence check manual. |

## Guardrail

This log is observational. It does not authorize automatic admission, changes to `node_quality_filter.py`, changes to the precision gate, or writes to serving memory. The 36 accepted relations live only in `manual_accepted_proposal_overlay.json` with `proposal_only=true` and `safe_for_general_runtime=false`.
