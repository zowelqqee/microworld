# Legal-domain pilot v1 — pre-registration

Frozen **before** any model call on legal text. Nothing below was revised after
seeing extraction output.

## Research question

Can the existing explicit-graph + deterministic-planner architecture extract
and represent legal relations with the same precision-gate discipline already
used for factual QA — or do legal text's structural properties
(cross-referencing, conditional/exception clauses, hierarchical nesting)
constitute a failure mode that the current extraction approach does not have a
representation for?

## What is reused unchanged

| Component | Status |
|---|---|
| `worldpgt.relation_extraction_v2.node_quality_filter.filter_triples` | **Unchanged.** Not tuned for legal text. |
| `run_llm_manual_review_batch_v1.node_quality_triage` | **Unchanged**, imported directly. |
| `run_llm_manual_review_batch_v1.group_for_manual_review` | **Unchanged**, imported directly. |
| `run_llm_manual_review_batch_v1.call_gemini` | **Unchanged**, imported directly. |
| Proposal-local literal index | **Unchanged** construction. |
| Manual-review sheet shape and verdict vocabulary | **Unchanged.** |
| Model and rate limit | `gemini-3.1-flash-lite`, ≥5 s spacing, same as the arXiv targeted run. |

The node-quality filter is knowingly arXiv-tuned (its `_GENERIC_HEADS`
blocklist contains words like `method`, `process`-adjacent heads, `data`,
`information`, `system`). It is **not** being relaxed for this domain. If it
misfires on legal text, that misfire is a reportable result, not a bug to be
patched mid-pilot.

## What is new, and why

Two things only.

1. **Source acquisition and segmentation.** arXiv units are cue-bearing
   sentences; legal units are numbered provisions. Segmentation is
   deterministic from GPO markup (see `source_selection.md`). No LLM
   participates in segmentation.

2. **The extraction prompt.** Adapted from the frozen arXiv targeted prompt
   (`artifacts/llm_manual_review_v1/targeted_prompt_v1/prompt.md`), preserving
   its exact shape: a closed enumerated list of permitted relation types, an
   explicit DO-NOT list, and permission to return `[]` rather than force an
   extraction. Only the relation-type vocabulary is swapped from academic to
   legal.

## Acknowledged adaptation, declared in advance

Each unit is presented to the model with a deterministic citation header, for
example:

```
35 U.S.C. §102(b)(1)(A) — [stem] A disclosure made 1 year or less before …
shall not be prior art to the claimed invention under subsection (a)(1) if—
[provision] (A) the disclosure was made by the inventor …
```

The citation is composed from section and subdivision markers that literally
appear in the source document, and it makes provision-anchored subjects
(`35 U.S.C. §102(b)(1)(A)`) literal spans of the unit text — which is what the
unchanged filter's literal index requires.

This is a real adaptation and it is declared here so it cannot be presented
later as a neutral choice. To keep it auditable, every candidate additionally
records `literal_in_statutory_body`, computed against the statutory text
**with the citation header removed**. The report states both numbers.

## Frozen prompt

Profile id: `targeted_legal_provision_relations_v1`. Verbatim text in
`prompt.md`.

## Pre-registered gate

The gate is the same discipline applied to the arXiv extraction pilot.

**Primary gate — hallucination rate < 10%.**

A candidate is a **hallucination** if any of:
- the asserted relation is not stated by the unit text (content invented,
  imported from the model's own knowledge of patent law, or inferred beyond
  what is literally written);
- `evidence_span` is not a verbatim contiguous span of the unit text;
- the relation's *direction* or *polarity* is reversed relative to the text
  (e.g. asserting that a disclosure **is** prior art where the text says it
  **shall not be**);
- a cross-reference target is cited that the unit text does not cite.

Polarity and reference-target errors are counted as hallucinations, not as
ordinary rejects, because in legal text they invert the legal meaning.

**Secondary measure — manual acceptance rate**, reported for comparison
against the arXiv targeted-prompt result of 34/45 = 75.6%. This is a
comparison figure, not a gate; the arXiv run had a different unit type and a
single comparison is not conclusive.

### Verdict rule, fixed in advance

- **PASS → proceed to a broader legal pilot** if hallucination rate < 10%
  *and* the surviving accepted relations are representationally adequate —
  i.e. the accepted triples preserve the conditionality and polarity of the
  provisions they came from.
- **CONDITIONAL** if hallucination rate < 10% but accepted triples
  systematically drop conditions, exceptions, or polarity. The extraction is
  then honest about what it wrote but the *representation* is lossy, which is
  an architectural finding, not an extraction-quality finding.
- **STOP / redesign** if hallucination rate ≥ 10%.

A CONDITIONAL or STOP verdict is a legitimate outcome of this pilot. Nothing
in this artifact authorizes promotion, serving-memory writes, precision-gate
changes, or any product claim.
