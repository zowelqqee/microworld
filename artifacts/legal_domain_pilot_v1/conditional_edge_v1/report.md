# Conditional-edge representation v1 — schema design and retrospective gate

**Status: design + retrospective test complete. Verdict: PASS (15/16), with
one named architectural limit.** Zero API calls, zero new extraction, no
production code touched, nothing promoted.

- Schema prototype: [`schema_prototype.py`](schema_prototype.py) (design artifact, not runtime code)
- Retrospective builder: [`build_conditional_edges.py`](build_conditional_edges.py)
- Rebuilt edges + per-candidate checks: [`conditional_edges.json`](conditional_edges.json)
- Coverage and collision analysis: [`verification.json`](verification.json)
- Source of the 16 candidates: [`../runs/usc35_chapter10_v1/manual_review_decisions.json`](../runs/usc35_chapter10_v1/manual_review_decisions.json)

## 1. Existing schema — the extension is minimal, no parallel system needed

`worldpgt/relation_extraction_v2/types.py` already establishes the exact
pattern required. `ExtractedRelationCandidate` carries one optional structured
field:

```python
evidence: Optional[RelationExtractionEvidence] = None
```

The proposal reuses that precedent and adds three more optional fields:

```python
conditions: list[ConditionClause] = []      # conjunction — all must hold
exceptions: list[ConditionClause] = []      # disjunction of defeaters
polarity:   Literal["affirm","negate"] = "affirm"
```

with

```python
@dataclass(frozen=True)
class ConditionClause:
    text: str
    evidence_span: str          # literal span, same discipline as every other edge
    kind: Literal["factual","scope"] = "factual"
```

**No new edge class, no parallel store.** Existing simple edges are unaffected:
empty lists plus `polarity="affirm"` reproduce today's semantics exactly, and
`to_dict()` omits the three keys at their defaults, so existing overlay JSON
stays byte-identical. `is_simple()` tells a consumer whether it is looking at a
pre-extension edge.

Provenance is reused rather than reinvented: the *provision* that states the
rule goes in the existing `source_url` / `stated_in` slot. This is what fixes
the legal pilot's `subject_attachment_false_assertion` hallucination — the
subject becomes the entity the rule is *about* ("a disclosure"), not the
citation.

### The `kind` field was discovered by the test, not designed in advance

The task specified three fields. The retrospective test showed that **10 of 16**
candidates carry a restriction that is neither a factual condition nor an
exception, but a limit on the *context of application*: "not prior art **under
subsection (a)(1)**", "**for purposes of** determining whether … is prior art",
"**for the purposes of this title**". Dropping these over-claims: a disclosure
excluded under (a)(1) may still be prior art under (a)(2).

These are encoded as conditions tagged `kind="scope"`. Truth is preserved with
the bare three-field schema; the tag only makes the distinction inspectable.
This is reported as a discovered refinement rather than presented as part of
the original design.

## 2. Semantics, stated so a planner can rely on them

| Construct | Representation |
|---|---|
| "A unless C" | `polarity=negate` + `conditions=[C]` |
| "A if C1 and C2" | `conditions=[C1, C2]` (list is conjunction) |
| "A if C1, or C2, or C3" | three **separate edges**, one condition each |
| "A, except X and Y" | `exceptions=[X, Y]` |
| "A if B does not apply" (alternative limbs) | condition citing the competing provision |

## 3. Retrospective test — all 16 candidates, rebuilt

Predicate word count is the direct measure of the welding failure mode.

| Candidate | Provision | Original defect | Old pred. | New predicate | cond | exc | polarity |
|---|---|---|---:|---|---:|---:|---|
| `usc35c10-010:1` | §100(i)(2) | clause_fragment | 5w | 3w `determined_by_deeming` | 1 | 1 | affirm |
| `usc35c10-012:1` | §101 | condition_stripped | 5w | 2w `may_obtain` | 1 | 0 | affirm |
| `usc35c10-013:1` | §102(a)(1) | condition_in_predicate | 12w | 2w `entitled_to` | 1 | 0 | **negate** |
| `usc35c10-014:2` | §102(a)(2) | consequence_omitted | 5w | 2w `entitled_to` | 1 | 0 | **negate** |
| `usc35c10-015:0` | §102(b)(1)(A) | **hallucination** | 3w | 4w `is_prior_art_to` | 2 | 0 | **negate** |
| `usc35c10-016:2` | §102(b)(1)(B) | condition_in_predicate | **44w** | 4w `is_prior_art_to` | 2 | 0 | **negate** |
| `usc35c10-017:1` | §102(b)(2)(A) | condition_in_predicate | 25w | 4w `is_prior_art_to` | 2 | 0 | **negate** |
| `usc35c10-018:1` | §102(b)(2)(B) | condition_in_predicate | 10w | 4w `is_prior_art_to` | 2 | 0 | **negate** |
| `usc35c10-019:1` | §102(b)(2)(C) | condition_in_predicate | 10w | 4w `is_prior_art_to` | 2 | 0 | **negate** |
| `usc35c10-020:1` | §102(c)(1) | condition_stripped | 5w | 4w `deemed_commonly_owned_with` | 4 | 0 | affirm |
| `usc35c10-023:3` | §102(d)(1) | **hallucination** | 11w | 4w `effectively_filed_as_of` | 3 | 0 | affirm |
| `usc35c10-024:9` | §102(d)(2) | condition_in_predicate | 37w | 4w `effectively_filed_as_of` | 3 | 0 | affirm |
| `usc35c10-025:2` | §103 | condition_in_predicate | **51w** | 4w `may_be_obtained_for` | 2 | 0 | **negate** |
| `usc35c10-025:3` | §103 | condition_in_predicate | 8w | 2w `negated_by` | 0 | 0 | **negate** |
| `usc35c10-026:1` | §105(a) | **exception_stripped** | 30w | 6w `considered_made_used_or_sold_in` | 1 | **2** | affirm |
| `usc35c10-027:1` | §105(b) | condition_in_predicate | 22w | 6w `considered_made_used_or_sold_in` | 2 | 0 | affirm |

Aggregate, machine-checked:

| Measure | Before | After |
|---|---:|---:|
| Longest predicate | **51 words** | **6 words** |
| Distinct predicates for 16 edges | 15 one-off clauses | **9 reusable** (`is_prior_art_to` ×5, `entitled_to` ×2, `effectively_filed_as_of` ×2, `considered_made_used_or_sold_in` ×2) |
| Condition clauses stored | 0 | 29 |
| Exception clauses stored | 0 | 3 |
| Edges carrying `negate` | n/a | 9 |
| Evidence spans verbatim | — | **100%** (all 32 clause spans literal substrings of their provision) |
| Unresolved contradictions | 7 pairs | **0** |
| Mean per-provision content coverage | 0.48 | **0.79** |

### (a) Welding removed — mechanically verified

`verify_edge()` fails any predicate longer than 6 words or containing a
conditional connective (`if`, `unless`, `except`, `shall`, `states that`, …).
All 16 pass. The 44-word and 51-word predicates became `is_prior_art_to` and
`may_be_obtained_for`.

The vocabulary is now **closed and reusable** — 9 predicates covering 16 edges,
with genuine reuse. That is the property `ALLOWED_RELATIONS` requires and the
flat extraction could not supply.

### (b) Exceptions can no longer be dropped silently — §105(a)

The flagship failure. The flat triple stated the outer-space rule while
dropping both exceptions, asserting it precisely for the two classes the
statute excludes. Rebuilt:

```
subject    any invention made, used or sold in outer space on a space object
           or component thereof under the jurisdiction or control of the United States
predicate  considered_made_used_or_sold_in          object  the United States
polarity   affirm
conditions [scope] "for the purposes of this title"
exceptions [1] "…specifically identified and otherwise provided for by an international
               agreement to which the United States is a party"
           [2] "…carried on the registry of a foreign state in accordance with the
               Convention on Registration of Objects Launched into Outer Space"
```

Content coverage 0.46 → 0.89. The drop is now structurally impossible to hide:
an empty `exceptions` list is visibly different from a populated one, and each
exception carries its own literal span. §105(b) — which conditionally
re-includes exception [2] — sits alongside it without contradiction.

### (c) Alternative limbs resolved — §100(i)(1) and §102(d)

`contradiction_report()` finds every pair sharing subject+predicate that would
collide if conditions were ignored. It found **7 such pairs, and 0 remain
unresolved.**

- **§100(i)(1)(A)/(B)** (the case named in the task): two `effective filing
  date --means-->` edges with different objects. Limb (A) carries the literal
  guard `"if subparagraph (B) does not apply"`; limb (B) is the default. The
  pair is a priority-ordered rule, not a contradiction. Evaluating it requires
  resolving "subparagraph (B)" to the sibling edge — the same cross-reference
  resolution the legal pilot already identified as needed.
- **§102(d)(1)/(2)** — identical shape, same resolution.
- **§102(b)(2)(A)/(B)/(C)** — same triple, three alternative sufficient
  conditions. Modelled as three edges, which is the correct disjunction:
  any one suffices.

### The coverage check caught an error I made

While representing §102(d), I dropped the qualifier "with respect to any
subject matter described in the patent or application." Union-coverage analysis
surfaced it (0.55 for a provision whose siblings scored 0.8+); it is now
encoded as a scope condition and coverage rose to 0.67/0.80.

Worth stating plainly: **the schema does not prevent omission — it makes
omission measurable.** That is the honest claim, and it is the claim the
verification tooling supports.

## 4. Gate decision

Pre-set threshold: **13+ of 16** represented without welding, dropping, or
contradiction.

**Result: 15/16 PASS, 1 FAIL. Gate PASSES.**

### The one failure — §102(c)(1), disjunctive consequence

`usc35c10-020:1` is counted as a **FAIL**. The statute's consequence is itself
disjunctive: subject matter "shall be deemed to have been owned by the same
person **or** subject to an obligation of assignment to the same person."
The rebuilt edge compresses both limbs into the single predicate name
`deemed_commonly_owned_with`. A reader of that edge cannot recover the
assignment-obligation limb.

The schema handles disjunction in *conditions* (separate edges) but **has no
representation for disjunction in the consequence**: two edges sharing a
condition set would be read conjunctively, which is wrong. This is a genuine,
named limit and it is the most valuable negative result here.

It is also the only candidate requiring **cross-provision assembly**: §102(c)'s
three conditions are joined by "and" across three separate provisions, so a
faithful edge needs all three. The schema stores that correctly (conditions are
a conjunction), and the flat triple hid the requirement entirely — but the
extraction pipeline, which works one provision at a time, cannot currently
produce it.

### Two further caveats on passing candidates

- **§101 — unevaluable condition.** "Subject to the conditions and
  requirements of this title" is now stored as a first-class condition with a
  literal span, so nothing is dropped. But it is an open-ended reference to an
  entire statutory title: **inspectable, not decidable.** The same applies to
  the §100 definitional exception "unless the context otherwise indicates."
  The schema makes such conditions visible; it does not make them checkable,
  and a planner must be able to say so rather than silently treating the edge
  as unconditional.
- **§103 — arity coercion.** "A patent for a claimed invention may not be
  obtained" is a *unary* proposition. Forcing it into a binary triple
  (`a patent --may_be_obtained_for--> a claimed invention`, negated) preserves
  truth but is a coercion. Some legal consequences are not relations between
  two entities.

### What still is not solved by this schema

The legal pilot's other findings are untouched and remain open:

- Legal subjects are **class descriptions**, not named entities ("whoever
  invents or discovers any new and useful process, machine, manufacture, or
  composition of matter"). The validator's entity checks
  (`_is_clean_entity_phrase`: ≤5 words, no commas, capitalised) reject these.
  A conditional edge does not help; a **class-subject node type** is a separate
  requirement.
- Citation nodes still need a deterministic normalizer (`365(a)` → `35 U.S.C.
  §365(a)`).
- Cross-reference *resolution* is now load-bearing: alternative-limb guards
  like "if subparagraph (B) does not apply" are only evaluable if the planner
  can follow the reference.

## 5. Renderer impact (assessment only — not implemented)

**The existing structure-driven framing mechanism is sufficient. The existing
rendering policy is not.**

The mechanism fits cleanly. `ContentBlock.kind` in
`worldpgt/reasoning/answer_behavior.py` is derived from graph structure only,
and `answer_plan_renderer._sentence_for` already special-cases one kind
(`uncertainty_note`) with a bespoke, structure-built sentence while everything
else flows through `_clause` / `_grounded_clause`. A new structure-derived kind
— `conditional_claim` — would follow that established pattern exactly, the same
way `reflective_reasoning_extended_v2.render_extended` mirrors it for
`speculative_extended`. No new subsystem.

Sketch, driven entirely by the fields:

| Structure | Surface |
|---|---|
| `affirm` + conditions | "{subject} {predicate} {object}, provided that {conditions joined by *and*}." |
| `negate` + conditions | "{subject} does not {predicate} {object} where {conditions}." |
| exceptions present | "… except where {exceptions joined by *or*}." |
| `kind="scope"` condition | leading clause: "For purposes of {scope}, …" |
| unevaluable condition | explicit: "…subject to a condition the graph stores but cannot check: '{text}'." |

**The policy gap is the real finding.** `uncertainty_note` is *additive*
framing — omitting it makes an answer less hedged, not false. Conditions and
exceptions are **truth-conditional**: an answer that renders
`considered_made_used_or_sold_in(invention, United States)` and drops its two
exceptions is simply wrong. Today nothing in the planner marks a block as
mandatory-to-render.

So a conditional edge needs an invariant the current architecture lacks: **an
edge with non-empty conditions or exceptions may not be selected as evidence
unless its guards are rendered with it.** Without that, the exception-dropping
failure fixed at the graph layer re-enters at the speech layer.

Two secondary notes: conditions run long (up to ~280 characters), against
`_MAX_QUOTED_SPAN_CHARS = 240`; and `polarity="negate"` must reach the surface
as negation, since a renderer that ignores the field inverts the law.

## 6. Verdict

A three-field optional extension of the existing edge — `conditions`,
`exceptions`, `polarity`, each clause carrying its own literal evidence span —
**correctly represents 15 of the 16 conditional-consequence candidates**,
eliminates the welded-predicate failure mode outright (51 words → 6), makes
silent exception-dropping structurally visible, and resolves all 7
alternative-limb contradictions. It needs no parallel edge class and breaks no
existing simple edge.

It does **not** solve disjunctive consequences, class-subject nodes, or
condition *decidability*, and it shifts a new requirement onto the renderer.

Integration into the extraction pipeline is the next step and is deliberately
**not** part of this task.

## 7. Guardrail

Design and retrospective analysis only. No model calls, no new extraction, no
writes outside this directory, no changes to
`worldpgt/relation_extraction_v2/` or any other runtime module, nothing
promoted or admitted. `schema_prototype.py` is a design artifact under
`artifacts/` and imports nothing from the runtime.
