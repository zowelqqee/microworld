# Legal-domain pilot v2 — pre-registration

Frozen **before** any model call on Chapter 41. Nothing below was revised after
seeing extraction output.

## Research question

Does the targeted legal-extraction approach validated on 35 U.S.C. ch. 10
(patentability: definitions + conditions) **transfer to a structurally
different chapter** — a criminal offense chapter dominated by the
**violation → penalty** relation type that ch. 10 could not test — at the same
hallucination gate, using the same unchanged node-quality filter?

And, secondarily: do the **conditional-edge schema** findings from
`conditional_edge_v1` recur here, i.e. are penalty consequences also
conditional/exception-bearing in a way the flat triple cannot hold?

## What is reused unchanged

| Component | Status |
|---|---|
| `node_quality_filter.filter_triples` | **Unchanged.** Still arXiv-tuned; not tuned for legal text. |
| `run_llm_manual_review_batch_v1`: `node_quality_triage`, `group_for_manual_review`, `call_gemini`, `load_dotenv`, `_review_markdown` | **Unchanged**, imported directly. |
| `run_legal_domain_pilot_v1.parse_sections` | **Unchanged**, imported directly (GPO markup is title-agnostic). |
| Model / rate limit | `gemini-3.1-flash-lite`, temperature 0, ≥5 s spacing — identical to v1. |
| Gate | Identical to v1 (below). |

## What is new in v2, and why

1. **Source**: Title 18 ch. 41 instead of Title 35 ch. 10 (see
   `source_selection.md`).

2. **Segmenter extension** — three narrow, general changes over v1's
   `segment_units`, confined to the pilot's own experiment script
   (`run_legal_domain_pilot_v2.py`); **no production module is modified**:
   - **Title parameter.** Citations read "18 U.S.C. §…" instead of a hardcoded
     "35 U.S.C.".
   - **Hanging-clause folding.** A trailing *unmarked* paragraph that follows
     an enumerated list (e.g. §879's "shall be fined … or imprisoned …" after
     (a)(1)–(a)(4)) is folded into its governing subsection stem, so the
     penalty is carried into each leaf rather than orphaned or mis-attached to
     the section preamble. This is essential: the penalty clause is the target
     relation of this pilot.
   - **4th hierarchy level (roman).** A lowercase-roman marker `(i)/(ii)`
     appearing under an uppercase `(A)/(B)` is recognised as a level-4
     subdivision, so §879(b)(1)(B)(i) segments correctly instead of producing a
     malformed citation.

   These changes only affect *segmentation*. No LLM participates in
   segmentation; the extraction prompt, filter, and gate are untouched.

3. **Prompt**: the same shape as v1's `targeted_legal_provision_relations_v1`,
   with the relation-type list extended by one type for the offense form —
   **penalty** ("violation of X results in penalty Y"). Full text in
   `prompt.md`, profile `targeted_legal_provision_relations_v2`.

## Frozen gate (identical discipline to v1)

**Primary gate — hallucination rate < 10%.** A candidate is a hallucination if:
- the asserted relation is not stated by the unit text (invented, imported from
  the model's own knowledge of criminal law, or inferred beyond the text);
- `evidence_span` is not a verbatim contiguous span of the unit text;
- the relation's direction or **polarity** is reversed relative to the text;
- a penalty is asserted that the text does not state (wrong term of years,
  wrong fine, or a penalty attached to the wrong offense);
- a cross-reference target is cited that the unit text does not cite.

Penalty-value errors and polarity errors count as hallucinations, not ordinary
rejects, because they misstate the law.

**Secondary measures**, reported for comparison, not gated: manual acceptance
rate (vs. v1's 76.9%), accept rate by relation type, and how many penalty
consequences are conditional (feeding the conditional-edge question).

### Verdict rule, fixed in advance

- **PASS (transfer confirmed)** if hallucination rate < 10% and the accepted
  relations — including penalty consequences — are representationally adequate.
- **CONDITIONAL** if hallucination rate < 10% but accepted triples
  systematically drop conditions, exceptions, or polarity (as in v1). This
  would show the conditional-edge gap is not specific to patentability.
- **STOP / redesign** if hallucination rate ≥ 10%, or if the penalty relation
  type specifically fails in a way definitions/conditions did not.

A CONDITIONAL or STOP verdict is a legitimate outcome. Nothing in this artifact
authorizes promotion, serving-memory writes, precision-gate changes, or any
product claim.
