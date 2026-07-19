# Entity-seeding lane v1 — design checkpoint

Status: **DESIGN COMPLETE — pilot may start**
Date: 2026-07-19

## Problem and current architecture

The current relation-extraction stack resolves entity surfaces through
`worldpgt.relation_extraction_v2.entity_surface_index.EntitySurfaceIndex`.
The index is built read-only from the accepted overlay, the promoted/serving
overlay, the snapshot overlay, and (where configured) graph-backed node
surfaces. The API builds it during startup and reuses it for extraction,
question routing, dialogue resolution, and answer annotation. There is no
entity-admission method and no mutation path from a rejected extraction into
this index.

`node_quality_filter.py` is a separate pre-gate layer. It rejects authorial,
generic, event-like, list-derived, and unresolvable endpoint surfaces. The
existing precision validator is downstream and must remain unchanged. The
current `unresolvable_entity` result is the specific boundary this lane is
testing: a literal, named, new surface cannot pass the existing filter merely
because it is capitalized.

## Minimal criterion under test

For one candidate surface `S`, admit a **proposal-only entity seed** only if
all of the following hold:

1. `S` is not already resolvable in the current serving/index inputs.
2. The exact normalized surface (whitespace-normalized and compared
   case-insensitively; no alias or fuzzy merge) occurs in at least **two
   independent extractor runs**
   (`gpt-4o-mini`, Gemini 2.5 Flash, or Gemini 3.1 Flash-Lite), each in a
   distinct extracted relation record. Multiple mentions produced by one model
   run do not satisfy the count by themselves.
3. In every supporting record, the subject/object span is a literal
   case-insensitive substring of that record's stored source sentence. No
   title fallback, inferred abbreviation, nominal-list completion, or semantic
   rewrite counts as support.
4. The supporting relation passes the existing `assess_node_quality` decision
   when evaluated with a proposal-local, in-memory index containing only the
   exact literal candidate surfaces needed for that record. The production
   `node_quality_filter.py` code is not changed. This ephemeral index removes
   only the expected current-index failure (`unresolvable_entity`) so the
   existing authorial/generic/event/list checks still run.
5. The surface is manually verified as one distinct, legitimate entity rather
   than an authorial reference, generic concept, event, list artifact, phrase
   fragment, duplicate surface, or a merely descriptive property.

The pilot will report all qualifying and rejected candidates, including the
reason each candidate failed. A candidate can qualify on relation consensus
and literal support but still fail the final manual verification gate.

## Why start with this one criterion

The observed failure is entity-resolution coverage, not a demonstrated need
to relax relation precision. Cross-extractor agreement is a cheap, bounded
signal that can recover names such as `SciServer` and `AutoSlim` without
introducing a general-purpose resolver. Requiring literal spans prevents the
seed lane from turning a model's inferred or normalized phrase into a node.
Reusing the existing node-quality rules preserves the validated rejection
boundary for authorial, generic, event-like, and list-derived candidates.
Manual verification is retained because repeated model output over the same
source sentence is consensus evidence, not independent external truth.

This is intentionally stricter than “capitalized unknown string” and
intentionally incomplete as an admission policy. It tests whether a small,
high-precision set can be identified before any broader policy is designed.

## Proposal-only and integration boundary

The pilot and any later build may create an audit artifact containing
candidate entities and their source evidence. It must not modify accepted or
promoted overlays, serving memory, the existing precision validator, or the
node-quality filter. If the pilot passes, the formal module will sit at:

`LLM extraction → existing node-quality check → entity-seeding check →
unchanged precision gate → proposal-only overlay`

The seed lane will be read-only with respect to existing components and will
return structured decisions/proposals. No candidate will become resolvable in
serving until a separate, explicit promotion process exists; that process is
outside this task.

## Explicitly out of scope for v1

- alias, acronym, article, morphology, or fuzzy entity merging;
- external search, Wikidata/OpenAlex/arXiv corroboration, or embeddings;
- entity-type inference or relation canonicalization;
- changing validator thresholds or accepted relation vocabulary;
- auto-promotion, serving-overlay writes, or mutation of `EntitySurfaceIndex`;
- a complete admission policy for single mentions, one-model mentions,
  source diversity, temporal decay, or conflict resolution.

## Pilot gate

The pilot proceeds to formal build only if manual review finds a reasonable
small set (target order: 5–10, but not a hard quota) of legitimate distinct
new entities and **zero false positives** among the candidates that satisfy
the mechanical criterion. Any noise under this strict criterion is a FAIL and
stops the lane at the pilot report.
