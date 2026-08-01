# Legal-domain pilot v2 — extraction results and gate verdict

**Status: pilot complete. Verdict: CONDITIONAL** (pre-registered category).
Transfer confirmed on the hallucination gate; the conditional-representation
gap from v1 recurs, in a new location. Nothing promoted, admitted, or served.

- Pre-registration (frozen before any model call): [`pre_registration.md`](pre_registration.md)
- Source choice and justification: [`source_selection.md`](source_selection.md)
- Frozen prompt: [`prompt.md`](prompt.md)
- Run artifacts: [`runs/usc18_chapter41_v1/`](runs/usc18_chapter41_v1)
- Companion: v1 [`../report.md`](../report.md) · conditional-edge schema [`../conditional_edge_v1/report.md`](../conditional_edge_v1/report.md)

## 1. What was run

| Field | Value |
|---|---|
| Source | 18 U.S.C. chapter 41 (Extortion and Threats), U.S. Code 2023 ed., GPO |
| Coverage | **All 28 leaf provisions.** No sampling, no selection bias |
| Why this chapter | Criminal-offense chapter — exercises the **violation → penalty** relation type v1 (patentability) could not test |
| Segmentation | Deterministic; v2 segmenter adds title parameter, hanging-clause folding, roman 4th level (see `pre_registration.md`). No LLM in segmentation |
| Model / prompt / filter / gate | Identical discipline to v1: `gemini-3.1-flash-lite`, temp 0, ≥5 s; unchanged node-quality filter; same gate |
| Cost | 17,051 in + 8,380 out tokens, ≈ $0.02 |

Reuse: `call_gemini`, `node_quality_triage`, `group_for_manual_review`,
`filter_triples`, and v1's `parse_sections` are all imported unchanged. New code
is confined to the pilot's own script — no production module touched.

## 2. Headline numbers

| Stage | Count |
|---|---:|
| provisions | 28 |
| raw model triples | 65 |
| node-quality exclusions (unchanged filter) | 7 |
| manual-review candidates | 58 |
| **manually accepted** | **49** |
| manually rejected | 9 |

| Measure | Result | Gate |
|---|---:|---|
| **Content hallucinations** (fabrication / wrong penalty value / wrong polarity / wrong target) | **0/58 = 0.0%** | **PASS** |
| Non-contiguous evidence spans (violate the verbatim-span rule, assert no false content) | 4/58 = 6.9% | — |
| **Full-gate reading** (counting span defects as failures) | **6.9%** (CI 2.7–16.4%) | **PASS** (< 10%) |
| Acceptance rate | 49/58 = **84.5%** (CI 73.1–91.6%) | comparison only |
| v1 acceptance comparison | 50/65 = 76.9% | v2 slightly higher |
| Unchanged precision gate on the 49 accepts | **0 safe / 49 quarantined** | same as v1 |

**On the primary gate, transfer is confirmed and clean.** Zero content
hallucinations — the penalty relation type did not introduce a fabrication
failure. Every one of the ~30 penalty values (terms of years, fines) was
transcribed exactly; not one was invented or altered.

The 4 non-contiguous spans are honest to report but assert no false content:
3 are `…`-joined conjoined-list definitions in §878(c) (subject and object both
literal), and 1 is an artifact of the pilot's **own** hanging-clause fold in
§879(a)(1). Both readings clear the gate.

## 3. Accept rate by relation type — the penalty type transfers best

| Relation type | Accepted | Rate |
|---|---:|---:|
| **Penalty** (`violation → penalty`) | 20/22 | **91%** |
| Cross-reference | 15/17 | 88% |
| Definition | 11/14 | 79% |
| Scope / applicability | 2/3 | 67% |
| Conditional consequence | 1/2 | 50% |

The relation type v1 could not test is the one that transfers **best**. The
canonical form — "Whoever [does X] shall be fined under this title or imprisoned
not more than N years, or both" — extracts cleanly and verbatim, with the
penalty value intact, in every clean offense section (§871, §873–877, §880).

## 4. Findings

### 4.1 The v1 conditional-representation gap recurs — relocated to the subject

This is the central result. In v1, conditional rules welded their condition
into a 44–51-word **predicate**. In v2 the same welding happens, but into the
**subject**: the offense conduct becomes a ~100-word subject clause.

> subject `Whoever knowingly and willfully deposits for conveyance in the mail … any threat to take the life of … the President of the United States … or the Vice President-elect`
> predicate `is_punishable_by`
> object `fined under this title or imprisoned not more than five years, or both`

**All 20 accepted penalty edges** carry the offense as a monolithic subject.
The predicate is clean (`is_punishable_by`), but the entire conditional
structure of the offense — the mental state ("knowingly and willfully"), the
act, the object of the threat, the intent element ("with intent to extort") —
is an unanalysable string. This is exactly what the unchanged precision gate
rejected: **0 of 49 accepts admitted**, 49/49 quarantined `ambiguous_relation`
+ `missing_explicit_evidence`, because a 100-word subject is not an entity node.

The conditional-edge schema (`../conditional_edge_v1`) is the right container,
and v2 shows its scope is broader than v1 implied: the offense elements belong
in the `conditions` list with the actor as subject and the penalty as object —

```
subject     an offender ("whoever")
predicate   is_punishable_by            object  fined … or imprisoned not more than five years, or both
conditions  [factual] "knowingly and willfully deposits … any threat to take the life of … the President"
            [factual] "with intent to extort" (for §875(b),(d) etc.)
```

Every specific v1 sub-finding also recurred here:

- **Exceptions dropped** — §878(a) "except that imprisonment for a threatened
  assault shall not exceed three years" was dropped from the base-penalty edge
  (rejected, as v1 §105(a) was), though the exception was separately extracted
  as its own unlinked edge (`014:4`).
- **Conditions stripped** — §878(d)'s jurisdiction consequence dropped all
  three governing conditions (rejected, as v1 §102(c)(1) was).
- **Scoped alternative limbs** — "immediate family" is defined three times in
  §879(b), each scoped to a different subsection ("with respect to subsection
  (a)(1)" vs "(a)(2) and (a)(3)"). This is the v1 §100(i)(1) alternative-limbs
  case, recurring — and the conditional-edge `kind="scope"` clause handles it.
- **Enhanced penalties** — §876(c)/(d) attach a higher penalty (10 yrs)
  conditional on "if … addressed to a United States judge …". The model
  captured this once well (condition as subject, `012:2`) and once badly (vague
  subject "the individual", condition dropped, `011:2` rejected).

### 4.2 The unchanged filter is actively harmful to the penalty form — it deleted whole offenses

v1 found the filter scored 0/7 on cross-references. v2 is worse: the filter's
`event_like_node` and `unresolvable_entity` rules reject the **offense-conduct
subject** of the penalty form, and it deleted penalties outright:

- **§872 (extortion by federal officers) — entirely lost.** All three of its
  candidates were filtered (`event_like_node`); §872 is one of only **two units
  in the chapter with zero surviving edges.**
- **§879(a)(1) and §879(a)(4) penalties** — filtered as `unresolvable_entity`.

So the chapter's headline relation — who commits an offense and what penalty
attaches — is precisely what the arXiv-tuned filter is built to discard. This
is not an argument to weaken the filter; it is direct evidence that the penalty
form needs a **class-subject node type** (an offender/conduct description is not
a named entity), a gap v1 already flagged and v2 makes unavoidable.

### 4.3 Cross-reference ellipsis recurs identically

§878(a)/(b) cite "section 112, 1116, or 1201"; only the first keeps the word
"section", so `1116`/`1201` were extracted as bare fragments and either
filtered or rejected — the exact mechanical failure from v1 that a deterministic
citation normalizer would fix.

### 4.4 What transferred with no friction

Definitions (11/14), including definition-by-cross-reference ("shall have the
same meanings as … section 1116(a)"), and internal cross-references (88%)
transferred as cleanly as in v1. Every provision except §872 and the
filter-stripped §879(a)(1) yielded at least one accepted edge (26/28), and the
model never returned `[]`.

## 5. Gate verdict

**CONDITIONAL** — the pre-registered middle outcome, on its pre-registered terms.

- **Hallucination gate PASSES** (0% content, 6.9% full-gate, both < 10%). The
  targeted approach **transfers** from patentability to criminal-offense text,
  and the previously-untested violation→penalty relation is in fact the
  **highest-accuracy** type (91%), with perfect penalty-value fidelity.
- **Representation gate does not pass.** The conditional structure of offenses
  is welded into the subject; 0/49 accepts are admissible under the unchanged
  precision gate. This is the same architectural gap v1 identified — now shown
  to be **not specific to patentability** but a general property of statutory
  text, and to live in the subject as readily as the predicate.

### Direct answer to the research question

The v1 approach transfers to a structurally different chapter without loss of
extraction honesty. It also **replicates v1's representation gap on independent
material**, which materially strengthens the v1 finding: the conditional-edge
schema is not a patch for one chapter's quirk but a response to a recurring
structural property of legal text. The schema's design (`conditions`,
`exceptions`, `polarity`, `kind="scope"`) covers every failure shape v2
surfaced — provided its subject can be a class description, which the current
entity model does not yet allow.

## 6. What this pilot does not establish

- Two chapters, two titles, one model, one run each. Still not "legal text".
- Manual review was performed against the verbatim text by the agent running
  the pilot, not a lawyer; every verdict is recorded with its reason in
  `manual_review_decisions.json` and is independently re-checkable.
- The v2 segmenter's hanging-clause fold, while general, was validated only on
  §879; it introduced one non-contiguous-span artifact (§879(a)(1)).
- **0 of 49 accepted edges are admissible today.** Nothing here is closer to
  serving than v1 was.
- Chapter 41 is a threat/extortion statute. The pilot extracts only structural
  relations (offense → penalty, definitions, cross-references) and quotes text
  solely as verbatim evidence spans; it reproduces no operational content.

## 7. Next step, if any

v2 does not change v1's recommended ordering, it reinforces it. The blocking
work remains **conditional-edge integration into the extraction pipeline**,
now with two concrete requirements that both pilots agree on:

1. a **class-subject node type** (an offender/conduct or a governed class is
   not a named entity), without which the penalty form cannot be admitted;
2. the conditional-edge schema wired in so offense elements land in
   `conditions` rather than a monolithic subject.

A QA layer still should not precede that work.

## 8. Guardrail

Proposal-only. No accepted-memory, promoted-overlay, or serving-memory write;
no precision-gate or node-quality-filter change; no commit to any serving path.
The precision gate ran in read-only assessment mode and admitted nothing. The
only code added is the pilot's own experiment script.
