# Targeted-prompt rejected-case analysis

## Scope and caveat

This analyzes the 11 manually rejected candidates from `arxiv_targeted_prompt_100_20260720`. Reviewer comments below are the supplied manual-review reasons. **n=11 is too small for a confirmed pattern or an automated fix.** This is a preliminary prompt-diagnostics observation only.

## Case-by-case classification

| ID | Extracted triple (short) | Reviewer comment | Classified reason |
|---|---|---|---|
| `arxiv-005:2` | ubiquitous computing applications → type of → research in CS | Applications are discussed within research areas, not stated as a type of research in CS. | **(c) Relation coercion:** contextual enumeration coerced into class membership. |
| `arxiv-028:1` | FSM → designed to provide → full student/activity clause | FSM is resolvable, but the object is a full purpose/activity clause rather than a reusable graph endpoint; the purpose relation is explicit. | **(b) Vague/non-node endpoint:** named subject, but purpose-clause object is not a graph node. |
| `arxiv-034:0` | FICS 2015 → instance of → workshop name | The parenthetical licenses an alias/abbreviation, not a distinct instance relation. | **(c) Relation coercion:** alias treated as instance-of. |
| `arxiv-034:1` | FICS 2015 → type of → satellite event of CSL 2015 | The source licenses `is satellite event of`, not `is a type of`. | **(c) Relation coercion:** specific event relation flattened into class membership. |
| `arxiv-042:0` | LIGO center → type of → computational infrastructure | `To enable this computational infrastructure` is purpose/context; it does not classify the center. | **(c) Attachment/scope:** purpose context converted to class membership. |
| `arxiv-054:0` | hit-and-run algorithm → type of → rounding | The source says rounding **uses** the algorithm; extraction reverses/coerces the relation. | **(c) Directional error:** `uses` converted to reversed type-of. |
| `arxiv-070:2` | DYNAPs → used to → improve performance/efficiency | The purpose applies to introducing the systems, not a stated use relation for DYNAPs. | **(c) Attachment/scope:** infinitival purpose attached to the wrong subject. |
| `arxiv-070:3` | Loihi → used to → improve performance/efficiency | The same purpose clause modifies system introduction, not an explicit Loihi usage relation. | **(c) Attachment/scope:** infinitival purpose attached to the wrong subject. |
| `arxiv-087:0` | OASIS → is a → suite of receivers | OASIS **has** a suite; it is not asserted to be that suite. | **(c) Possessive-scope error:** possession converted to identity/type. |
| `arxiv-087:1` | OASIS → allows for → observations | The suite of receivers, not OASIS directly, allows the observations. | **(c) Possessive-scope error:** capability assigned to possessor rather than possessed subject. |
| `arxiv-093:0` | MR SDS → type of → program structure | The source says MR SDS **supports** program structures; it does not classify MR SDS as one. | **(c) Relation coercion:** support relation converted to class membership. |

## Counts

| Top-level category | Count | Interpretation |
|---|---:|---|
| (a) Explicit generic/authorial forbidden statement | 0/11 | No evidence that direct `this paper` / generic-subject prohibition was ignored. |
| (b) Named endpoint but vague/broad/non-node object | 1/11 | FSM's full purpose/activity clause. |
| (c) Literal relation, direction, attachment, or subject-scope error | **10/11** | Repeated residual failure family. |
| (d) Other | 0/11 | — |

Six category-(c) cases coerce a non-target relation into `is a type of` or `is an instance of` (`005:2`, `034:0`, `034:1`, `042:0`, `054:0`, `093:0`). Four mis-attach a purpose or possessive clause (`070:2`, `070:3`, `087:0`, `087:1`).

## Preliminary diagnosis

The errors are not diverse irreducible noise in this small sample. They concentrate in one broader failure family: the model selects an allowed relation shape but does not preserve the source's exact relation direction, attachment, or semantic type. This is a literal-entailment failure, not a direct violation of the authorial/generic-subject `DO NOT` instructions.

It is plausible—but unconfirmed—that this family is partially reducible by a small prompt refinement. The data does not justify changing production prompt behavior now.

## Proposed test-only prompt addendum

For a future independent batch, test this one additional constraint without rewriting the prompt:

> Before output, verify that the **exact predicate and direction** are stated for the proposed subject and object. Do not convert aliases/parentheticals, possession, `uses`/`supports`, satellite-event relations, or purpose/capability clauses into `is a type of`, `is an instance of`, or system-property triples. If the exact relation is not stated, return `[]`.

This addendum targets all 10 category-(c) cases, but its effectiveness must be measured on a new disjoint manual-reviewed batch. It is not a production change, an automated admission rule, or evidence that all residual errors are prompt-fixable.

## Guardrail

No extraction was run and no prompt was changed for this analysis. Nothing was auto-admitted or promoted; precision gate and serving memory remain unchanged.
