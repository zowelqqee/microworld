# Completed manual review — grouped Gemini batch

Reviewer decisions cover all 148 node-quality-triaged candidates. The
authoritative decision encoding is `manual_review_decisions.json`; relation
and literal-evidence provenance remains in `manual_review_candidates.json`.

| Metric | Count |
|---|---:|
| reviewed candidates | 148 |
| manually accepted proposals | 36 |
| manually rejected candidates | 112 |
| accept rate | 24.3% |
| literal-support failures | 14 |
| accepted new-predicate proposals | 1 |

## Outcome by review group

| Primary group | Accepted | Rejected | Accept rate |
|---|---:|---:|---:|
| Anaphora-likely | 0 | 15 | 0.0% |
| Temporal-referent-likely | 0 | 1 | 0.0% |
| Generic-property-likely | 13 | 38 | 25.5% |
| Attachment-shape-flag | 0 | 1 | 0.0% |
| Clean / no-flag | 23 | 57 | 28.8% |

The labels concentrated clear failure modes: all 17 anaphora, temporal, and
attachment candidates were rejected. Generic-property is **not** a rejection
rule: 13 of 51 were accepted after review. Clean/no-flag is also not an
admission rule: 57 of 80 were rejected.

## Accepted proposal relations

All 36 accepted relations are selected by ID in
`manual_accepted_proposal_overlay.json`; their complete subject, predicate,
object, source sentence, evidence span, and official arXiv URL are retained in
`manual_review_candidates.json`.

Accepted IDs: `arxiv-001:0`, `arxiv-005:0`, `arxiv-009:1`, `arxiv-018:0`,
`arxiv-018:1`, `arxiv-018:3`, `arxiv-019:0`, `arxiv-021:0`, `arxiv-023:1`,
`arxiv-027:0`, `arxiv-042:0`, `arxiv-042:1`, `arxiv-043:1`, `arxiv-048:1`,
`arxiv-056:0`, `arxiv-056:1`, `arxiv-056:2`, `arxiv-066:0`, `arxiv-069:1`,
`arxiv-075:0`, `arxiv-084:0`, `arxiv-085:1`, `arxiv-088:0`, `arxiv-091:0`,
`arxiv-092:0`, `arxiv-093:0`, `arxiv-099:0`, `arxiv-101:0`, `arxiv-103:0`,
`arxiv-107:0`, `arxiv-110:0`, `arxiv-110:1`, `arxiv-111:0`, `arxiv-117:0`,
`arxiv-117:1`, and `arxiv-119:0`.

## Learning observations (not an automated rule)

- Repeated legitimate shape: named systems/methods with explicit `uses`,
  `supports`, `extends`, or `builds upon` relations to named techniques,
  architectures, or bounded methods.
- Appositional identifications and bounded quantitative facts also survived
  review (`detector technology` → `MKIDs`; GCR energy density → its measured
  value).
- Repeated rejections: unresolved pronouns/demonstratives, temporal values
  without a stable entity anchor, generic capabilities/properties, temporary
  survey/event mentions, and clause/attachment errors.
- Literal support alone was insufficient: 134 candidates had literal support,
  yet 98 were still rejected for non-clean or non-resolvable endpoints.

## Boundary

The overlay is proposal-only. Nothing was auto-admitted, the precision gate
did not run, serving memory did not change, and no commit was created.
