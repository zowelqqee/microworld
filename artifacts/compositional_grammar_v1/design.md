# Compositional grammar v1

This is an isolated research path. It does not replace, call, or modify the current multi-evidence, paraphrase, fan-out, semantic-parser, or API routes.

## Operators

`AND(subject, predicates[])` returns one component per requested predicate group. It applies only when at least two predicate types have complete explicit support for the same normalized subject. An empty predicate list is the bounded interpretation of the structural prompt “two key relations”: it selects all predicate groups in the supplied evidence slice. Each object remains a distinct evidence reference, so fan-out is preserved. It audits if fewer than two groups are requested/available, any requested group is absent, or any component fails the shared hop-safety policy.

`CHAIN(subject, first_predicate, second_predicate)` returns a two-edge path only where the first edge’s object normalizes to the second edge’s subject. It reuses the shared `validate_hop_safety` policy and `HopEdge` provenance shape from `multihop_qa`, but does not reuse its enum question analyser or its predicate-specific planner. It audits on a missing first or second hop, or an unsafe hop.

Every answer plan carries per-edge `evidence_id` plus the source/stability/risk metadata. No partial plan is emitted: a missing or unsafe component is an audit.

## Candidate parsing

The parser is deliberately conservative: it recognizes the frozen explicit “For X, what are its P and Q relations?” marker by matching predicate names dynamically from supplied data, the structural “Tell me two key relations about X” marker, and a small canonical chain form. Parsing is an admission gate, not natural-language understanding; callers may construct `AndQuery` or `ChainQuery` directly.

## Out of scope

FILTER, OR, negation, COUNT, conflict resolution, arbitrary-depth chains, free-form predicate paraphrases, and production routing are explicitly future work. The first iteration only proves data-driven two-operator composition and evidence-preserving refusal.
