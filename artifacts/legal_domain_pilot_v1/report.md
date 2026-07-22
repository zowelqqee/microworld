# Legal-domain pilot v1 — extraction results and gate verdict

**Status: pilot complete. Verdict: CONDITIONAL** (pre-registered outcome
category). Nothing was promoted, admitted, or served. No product claim is made
or implied.

- Pre-registration (frozen before any model call): [`pre_registration.md`](pre_registration.md)
- Source choice and its justification: [`source_selection.md`](source_selection.md)
- Frozen prompt: [`prompt.md`](prompt.md)
- Run artifacts: [`runs/usc35_chapter10_v1/`](runs/usc35_chapter10_v1)

## 1. What was actually run

| Field | Value |
|---|---|
| Source | 35 U.S.C. chapter 10 (Patentability of Inventions), U.S. Code 2023 ed., GPO |
| Coverage | **All 28 leaf provisions of the chapter.** No sampling, therefore no selection bias |
| Segmentation | Deterministic from GPO markup; no LLM involved |
| Model | `gemini-3.1-flash-lite`, temperature 0, ≥5 s spacing — same as the arXiv targeted run |
| Prompt | `targeted_legal_provision_relations_v1`, same shape as the arXiv targeted prompt |
| Node-quality filter | **Unchanged.** Not tuned for this domain |
| Cost | 13,682 input + 7,328 output tokens (0 thinking), ≈ $0.02 |

Reuse was the point: `call_gemini`, `node_quality_triage`,
`group_for_manual_review`, and `filter_triples` are imported from the existing
arXiv lane without modification. The only new code in
[`run_legal_domain_pilot_v1.py`](../../worldpgt/experiments/run_legal_domain_pilot_v1.py)
is source acquisition and provision segmentation.

## 2. Headline numbers

| Stage | Count |
|---|---:|
| source provisions | 28 |
| raw model triples | 72 |
| node-quality exclusions (unchanged filter) | 7 |
| manual-review candidates | 65 |
| manual verdicts recorded | 65 |
| **manually accepted** | **50** |
| manually rejected | 15 |

| Measure | Result | Gate |
|---|---:|---|
| **Hallucination rate (strict)** | **2/65 = 3.1%** (Wilson 95% CI 0.8–10.5%) | **PASS** (< 10%) |
| Hallucination rate over raw triples | 2/72 = 2.8% | PASS |
| Hallucination rate (lenient reading, §4.1) | 0/65 = 0.0% | PASS |
| Acceptance rate | 50/65 = **76.9%** (CI 65.4–85.5%) | comparison only |
| arXiv targeted-prompt comparison | 34/45 = 75.6% | statistically indistinguishable |
| Unchanged precision gate on the 50 accepts | **0 safe / 50 quarantined** | same as arXiv (0/36) |

The surface-level answer is "legal text behaves like arXiv text." **That
answer is misleading, and the rest of this report is why.**

## 3. The number that matters: accept rate by relation type

| Relation type | Accepted | Rate |
|---|---:|---:|
| Definition (`X is defined as Y`) | 13/13 | **100%** |
| Scope / applicability (`X applies to Y`) | 9/10 | 90% |
| Cross-reference (`X references Z`) | 19/26 | 73% |
| Conditional consequence (`condition → legal result`) | 9/16 | 56% |

The aggregate 76.9% is an average across four types whose behaviour is
completely different. Failure concentrates exactly where the research question
predicted it would: conditional and cross-referential structure.

## 4. Findings

### 4.1 Extraction is honest. Two hallucinations, both domain-specific.

Only two of 65 candidates assert something the provision text does not say,
and both are *legal* failure shapes rather than generic ones:

- `usc35c10-015:0` — subject-attachment error producing a false proposition:
  the edge reads "35 U.S.C. §102(b)(1)(A) **shall not be prior art** to the
  claimed invention." The bearer of "shall not be prior art" is *a
  disclosure*, not the provision, and the governing condition is absent
  entirely.
- `usc35c10-023:3` — defined-term substitution: the edge asserts "the
  **effective filing date** is …", but §102(d)(1) says the patent "shall be
  considered to have been **effectively filed**." "Effective filing date" is a
  separately defined term (§100(i)) meaning something different. Notably, the
  model got the identical construction *right* one paragraph later in
  §102(d)(2).

Under a lenient reading — counting only fabricated content or non-verbatim
spans, and treating both cases above as attachment/terminology defects rather
than hallucinations — the rate is 0/65. The gate passes either way; both
numbers are given so the count can be re-derived under either rule.

**On the primary pre-registered gate, legal extraction is at least as clean as
arXiv extraction.**

### 4.2 The unchanged node-quality filter was useless here — 0/7 precision

All 7 filter exclusions were `unresolvable_entity` on cross-reference targets
(`section 365(a)`, `section 386(b)`, `section 121`, …). Reviewed against the
text, **all 7 were substantively correct cross-references.** The filter
produced seven false negatives and zero true positives on this domain.

The cause is mechanical. The statute writes citation lists elliptically —
"a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b)" —
so only the *first* citation carries the word "section". When the model
normalised `365(a)` into `section 365(a)`, it stopped being a literal span and
the filter killed it. When the model instead emitted the bare `365(a)` (in
§102(d)(2)), the filter passed it and **manual review** rejected it as an
unresolvable node identity.

Either way the same 7 relations are lost. This is not an argument for
relaxing the filter; it is evidence that legal citation needs a deterministic
**citation normalizer** upstream of the filter. That is a bounded, solvable
engineering gap, not an architectural one.

### 4.3 The real finding: conditional rules produce zero usable edges

Sixteen candidates expressed a conditional legal rule. **None of them is
admissible**, for two mutually exclusive reasons:

| Outcome | Count | Why unusable |
|---|---:|---|
| True but unrepresentable | 9 | Verified correct, condition and polarity intact — because the entire conditional rule was welded into the *predicate string* |
| Representable but false | 7 | Ordinary triple shape — achieved by dropping the condition, the exception, or the consequence |

Nine accepted consequence edges look like this:

> subject `35 U.S.C. §102(b)(1)(B)`
> predicate `states that if the subject matter disclosed had, before such disclosure, been publicly disclosed by the inventor or a joint inventor or another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor, the disclosure shall not be`
> object `prior art to the claimed invention`

That is a 44-word predicate. Across the 50 accepted edges there are 15
distinct predicate surfaces; 9 of them are one-off clauses of up to 51 words
that will never recur. A predicate vocabulary cannot be built from them,
which is why the unchanged precision gate quarantined **50 of 50** accepted
edges as `ambiguous_relation`.

The seven that *do* fit the triple shape fit it by losing law:

- `usc35c10-020:1` asserts the §102(c) common-ownership deeming
  **unconditionally**, dropping "in applying the provisions of subsection
  (b)(2)(C) **if**—" and its three conditions.
- `usc35c10-026:1` states the §105(a) outer-space rule while dropping **both**
  express exceptions in the same sentence — so the edge asserts the rule
  precisely for the two classes the statute excludes.
- `usc35c10-012:1` grants a §101 patent entitlement while dropping "subject to
  the conditions and requirements of this title."

Each of those is a materially wrong statement of law that a manual reviewer
caught. In an arXiv graph, dropping a qualifier degrades an edge. In a legal
graph, dropping a qualifier **inverts** it.

### 4.4 A second structural gap: alternative limbs collide

§100(i)(1) defines "effective filing date" through two limbs: (A) applies *if
(B) does not apply*, and (B) applies otherwise. Extraction produced two edges:

```
effective filing date --means--> if subparagraph (B) does not apply, the actual filing date …
effective filing date --means--> the filing date of the earliest application … under section 119 …
```

Both are individually faithful, and both survived review. Together they are a
contradiction, because the graph has **no representation for "these two edges
are alternative limbs of one definition, ordered by a priority rule."** The
existing architecture would have to treat one as a conflict with the other.

This is not an extraction defect. Extraction did its job. It is a missing node
type.

### 4.5 What transfers cleanly

Definitions transfer with no adaptation at all: 13/13 accepted, all verbatim,
all in the shape the existing graph already stores (`term --means--> definition`).
Scope statements transfer at 9/10. Cross-references transfer at 73% with a
purely mechanical, fixable failure cause.

Every one of the 28 provisions yielded at least one accepted edge, and the
model never returned `[]` — legal text is far more relation-dense per unit
than arXiv abstracts, where the same targeted prompt suppressed yield.

## 5. Gate verdict

**CONDITIONAL** — the pre-registered middle outcome, reached on its
pre-registered terms.

> "**CONDITIONAL** if hallucination rate < 10% but accepted triples
> systematically drop conditions, exceptions, or polarity."

- The **hallucination gate passes** (3.1% strict, 0% lenient, both < 10%). The
  targeted arXiv extraction approach transfers to legal text without loss of
  extraction honesty, and the acceptance rate (76.9%) is statistically
  indistinguishable from the arXiv result (75.6%).
- The **representation gate does not pass.** Legal text's conditional and
  exception structure produces a failure mode that arXiv extraction never
  exhibited: the choice between an edge that is true but has no canonicalizable
  predicate, and an edge that has a clean predicate but states the law wrongly.
  Zero of 16 conditional-rule candidates are usable.

### Answering the research question directly

> Can the existing explicit-graph + deterministic-planner architecture extract
> and represent legal relations with the same precision-gate discipline used
> for factual QA — or do legal text's structural properties require a
> fundamentally different extraction approach?

**Extraction: yes, unchanged.** The same prompt discipline, the same model,
the same manual-review protocol, the same gate, and comparable numbers.

**Representation: no, not for one of the four relation types.** Definitions,
scope, and cross-references are representable in the existing graph today.
Conditional consequences are not — and conditional consequences are what
statutes are mostly made of. This is a missing representation, not a
tuning problem, and it will not be fixed by a better prompt.

Cross-referencing turned out to be the *easier* of the two predicted problems:
it failed for mechanical reasons (elliptical citation lists) that a
deterministic normalizer solves. Conditionality is the hard one.

## 6. What this pilot does not establish

- One chapter, one statute, one jurisdiction, one model, one run. Nothing here
  generalises to "legal text."
- This chapter has **no penalty provisions**, so the requested
  "violation of X results in penalty Y" relation was never tested in its
  penalty form (declared in advance in `source_selection.md`).
- The 28 units are short and self-contained by construction. Long provisions,
  incorporation by reference, and amendment-over-time were not tested at all.
- Manual review was performed against the verbatim provision text by the
  agent executing the pilot, not by a lawyer. Every verdict is recorded with
  its reason in `manual_review_decisions.json` and is independently
  re-checkable against the quoted text; a qualified reviewer may well decide
  some of the 15 rejects differently. Legal correctness of the accepted edges
  is *not* certified.
- **0 of 50 accepted edges are admissible today.** The unchanged precision
  gate quarantined all of them. Nothing here is closer to serving than the
  arXiv pilot was.

## 7. Next step, if any

The honest next step is **not** the QA interface sketched as the conditional
follow-on. Building a QA layer over this graph now would surface exactly the
edges that are true-but-unrepresentable or clean-but-wrong.

The blocking question is narrower and comes first: **does the explicit graph
need a conditional-edge representation — an edge that carries a condition set,
an exception set, and a polarity as first-class structure rather than as
predicate text?** That is a runtime design question, answerable on this same
28-provision corpus without any new extraction, because the extraction output
needed to test it already exists in `manual_review_decisions.json`.

If that representation is designed and the same 16 conditional candidates can
be expressed in it without loss, a QA pilot over this chapter becomes
meaningful. Until then it would be measuring the wrong thing.

**Update — that question has since been answered.** See
[`conditional_edge_v1/report.md`](conditional_edge_v1/report.md): a three-field
optional extension (`conditions`, `exceptions`, `polarity`) represents 15 of
the 16 conditional candidates, eliminating the welded-predicate mode (51 words
→ 6) and resolving all 7 alternative-limb contradictions. The one failure is
disjunctive *consequences* (§102(c)). Pipeline integration remains open, and a
QA pilot still should not precede it.

## 8. Guardrail

Proposal-only. No accepted-memory write, no promoted-overlay write, no
serving-memory write, no precision-gate change, no node-quality-filter change,
no commit to any serving path. The precision gate was run in read-only
assessment mode and admitted nothing.
