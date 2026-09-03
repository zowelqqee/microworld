# Conditional-edge integration v1 — what shipped

The retrospective schema (`report.md`) is now wired into the runtime. This note
records what changed, the order it was done in, and how each piece stays
**dynamic** (no domain hardcoding) and **conservative** (nothing new
auto-admits). Branch: `feature/conditional-edges-and-class-subjects`.

Proof it works: the 16 conditional-consequence candidates that were **0/50
admissible** before this change are **16/16 admissible proposals** through the
integrated validator, and the elided cross-references (`365(a)` → `section
365(a)`) are recovered. See the end-to-end check reproduced below.

## What changed, in the agreed order

### #3 — Dynamic citation normalizer  ·  `relation_extraction_v2/citation_normalizer.py` (new)
Recovers the governing word that a statute elides across a reference list
("section 119, 365(a), 365(b)" → each is a *section*). **Dynamic:** the
governing word is read out of the evidence text, never from a table of section
numbers or titles; the only fixed knowledge is the closed set of English
subdivision nouns the Code uses as governing words. A surface that is not a bare
reference in a governed list is returned unchanged, so ordinary nodes are
untouched.

### #1 + #2 — Conditional edges + class-subject nodes
- `types.py`: `ExtractedRelationCandidate` gains optional `conditions`,
  `exceptions`, `object_alternatives`, `polarity`, `subject_kind`, plus
  `ConditionClause`. Defaults reproduce a plain edge exactly; `to_dict()` omits
  them at defaults, so existing overlays serialize byte-for-byte identically.
  `is_simple()` reports whether an edge is pre-extension.
- `node_quality_filter.py`: `classify_subject_node()` recognizes a class
  description ("whoever …", "any invention made …") vs a named entity.
  **Dynamic/structural:** the decision rests on determiner/quantifier position,
  a relative clause, and length — the cues a reader uses — not a keyword list.
- `relation_candidate_validator.py`: a structured edge is validated by *shape*
  — a short un-welded predicate, a verbatim-grounded object, and every
  condition/exception/alternative anchored to a literal evidence span — instead
  of by an allowlist of legal predicates (which would be the hardcoding this
  avoids). **Conservative:** a passing structured edge is a review-only proposal
  (`safe_for_overlay_delta = False`), never auto-admitted; simple edges follow
  the unchanged path.

### #4 — Structure-driven rendering + mandatory-guard invariant  ·  `reasoning/answer_plan_renderer.py`, `answer_behavior.py`
`EvidenceEdge` carries the conditional fields (parsed from the overlay dict).
`_render_conditional_claim()` builds the surface entirely from those fields —
scope → "For purposes of …", conditions → "provided that …", exceptions →
"except where …", negate polarity → "is not …". The **mandatory-guard
invariant** is the key safety property: a conditional edge can never be realized
by the plain path (which would silently drop its guards), and the renderer
raises rather than emit a rule missing any guard. This is the invariant the
pilot flagged as absent — conditions/exceptions are truth-conditional, not
optional hedging.

### #5 — Disjunctive consequences
`object_alternatives` holds "deemed X **or** subject to Y" as one edge rendered
"X or Y", instead of two sibling edges that would read as a conjunction — the
one case the retrospective schema could not represent.

## What is deliberately *not* done
- **Serving:** nothing is promoted. Conditional/class-subject edges are
  proposals; they do not enter serving memory.
- **SQL cache seam:** the persistent edge index has no conditional columns yet,
  so conditional edges must be served via the in-memory path until it is
  extended. Documented at `_edge_from_sql_row`; harmless today because these
  edges are not in serving memory.
- **Grammar polish:** predicate negation is rendered "is not X", correct for the
  stative statutory predicates seen; a general verb-negation realizer is future
  work.

## Tests
`worldpgt/tests/test_conditional_edges_v1.py` — 19 tests across all five pieces
(citation no-op/recovery, class vs entity, structured admit/quarantine,
class-subject grounding, disjunction, guard rendering, and the invariant
raising). Existing `relation_extraction_v2` and `reasoning` suites still pass;
the one failing and one erroring test in the repo fail identically on the base
branch (pre-existing, unrelated).

## End-to-end check (reproducible)
```
16 conditional-consequence edges (v1) -> admissible proposals: 16/16, quarantined: 0
citation normalizer recovers: ['section 365(a)', 'section 386(b)', 'section 121', 'section 365(c)']
```
