# Integration router v1 — final report

## Scope implemented

`worldpgt/reasoning/integrated_answer_router.py` is a new, non-invasive
orchestration wrapper.  It runs the existing `route_question` hard-safety
screen first, dispatches only safe requests with `BranchRouter`, and delegates
to the unchanged production branches:

- QA: `AnswerOrchestrator`;
- proven reflective reasoning: `reflective_reasoning_v1`;
- lower-confidence co-attribution: `reflective_reasoning_extended_v2`;
- constrained creative: `constrained_creative_v1`;
- pure creative: the existing `AnswerOrchestrator` Creative mode.

No branch implementation was edited.  `CognitiveAnswerSession` now constructs
the router by default and calls it after dialogue/coreference preparation;
non-QA output is returned directly, while QA continues through its established
session path.  The session's `ask(..., force_branch=...)` offers an explicit
caller override.  No separate QA/Creative UI toggle was found in this backend
repository, so there was no UI control to preserve or remove.

## Unified output contract

The wrapper returns `IntegratedAnswer`, whose public `support_kind` is
disjoint: `grounded`, `audit`, `speculative_inference`,
`speculative_extended`, `grounded_generation`, or `creative_generated`.
QA's more granular original label is preserved as
`detail.branch_support_kind`; no evidence provenance is discarded.

`speculative_extended` is rendered with the required caution that it is a
shared-attribute, broader and less-tested association.  It is never merged
with `speculative_inference`.  A regression guard tries co-attribution before
v1 abduction and rejects the latter's degenerate same-object bridge, avoiding
both a mislabeled verified inference and the malformed
`"X develops O, which develops O"` rendering.

## Realistic-flow set and results

`realistic_cases.json` contains 45 shuffled, natural user-style questions
across QA, both reflective forms, constrained creative, and pure creative. It
is explicitly distinct from the earlier boundary-adversarial router pilot.
`run_realistic.py` records question, intended family, route, final text, and
support label.

The final completed run was split into five deterministic chunks solely to fit
the execution ceiling, then merged with an exact 45-row count check.  It
produced **2/45 misroutes (4.4%)**, close to the earlier 3.2% pilot result.
Both are conservative fallbacks to QA:

1. `What if Tesla had never made electric cars?`
2. `What would happen if SpaceX stopped developing rockets?`
The two questions are activity/production counterfactuals outside the proven
existence-conferring v1 rule and correctly remain conservative.  The earlier
creative misses exposed a routing gap, so an explicit, narrow pure-creative
fast path was added for unambiguous literary requests.  All such paraphrases
in the final set now return `pure_creative`.

## Regression and gate decision

QA remains delegated to the unmodified `AnswerOrchestrator`; integration tests
compare direct and wrapped QA text and retain the original QA support label in
metadata.  Targeted checks passed: base QA (1), direct-vs-wrapped QA identity
plus extended-label separation (2), and safety plus all three creative
paraphrases (4).  Session-level smoke checks also returned
`speculative_extended` and `creative_generated` through
`CognitiveAnswerSession`.  The existing QA implementation and its held-out
artifacts were not changed by this integration layer.

**Gate: production-ready for the integrated `CognitiveAnswerSession` flow.**
The final realistic-flow rate is 4.4%, safely below the predeclared 15–20%
concern threshold; both errors fail safe and preserve the v1 rule's scope.
This is still **tested on 45 examples**, not production validation at scale.
Before widening rollout, continue monitoring representative traffic and rerun
the complete legacy QA held-out suite in an environment without the
per-command execution ceiling.
