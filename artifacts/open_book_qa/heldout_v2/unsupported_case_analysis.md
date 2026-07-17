# Held-out v2 unsupported-answer analysis

## Scope and method

The deterministic evaluator marks a response unsupported when it mentions a
known benchmark object that is neither expected for that case nor present in
that case's supplied evidence contexts. This analysis uses the first recorded
response for each paraphrase case; the five benchmark repeats are identical in
their selected plan edges. Exactly two of 20 paraphrase cases were flagged
(10%).

## Classification

Both cases are **(b) a true support-gate / plan-expansion bug**, not a
resolver/predicate-mapping error. The question entity and predicate were
correctly resolved as `used_for`, and the first selected edge was exactly the
expected edge. The planner then admitted a second edge through a non-exact
lexical attachment (`contained` or `soft`). That edge is real proposal evidence
about a different entity, but it is not evidence for the focused question and
is absent from the benchmark case's evidence set. The safety boundary should
have stopped the expansion after the exact direct claim.

## Case: Adobe Presenter

- **Case ID:** `obqa-02c190c26412b8c1-paraphrase-33a6c87b`
- **Question:** `For what application is Adobe Presenter employed?`
- **Expected object:** `presentation software`
- **Returned objects:** `presentation software`; additionally `screencast`
- **Decision / support kind:** `answer` / `evidence_backed_answer_plan`
- **Why the extra object is unsupported for this case:** `screencast` belongs
  to `Adobe Presenter Video Express`, a different relation edge and an absent
  case context. It is not an alias of `presentation software`.

### Trace

1. The direct, supported block selected
   `edge:adobe presenter|used_for|presentation software` with exact target
   attachment and score `0.875`.
2. The plan then selected
   `edge:adobe presenter video express|used_for|screencast` as a
   `sibling_elaboration`, attached by `contained` to target `adobe presenter`,
   score `0.600`.
3. No audit fired because the existing gate regarded any graph-connected,
   evidence-backed second block as supported. It did not require that every
   block in a focused predicate answer attach exactly to the resolved target.

## Case: Adobe Presenter Video Express

- **Case ID:** `obqa-a8f792d2fb8fac86-paraphrase-5726baa5`
- **Question:** `For what application is Adobe Presenter Video Express employed?`
- **Expected object:** `screencast`
- **Returned objects:** `screencast`; additionally `presentation software`
- **Decision / support kind:** `answer` / `evidence_backed_answer_plan`
- **Why the extra object is unsupported for this case:** `presentation software`
  belongs to `Adobe Presenter`, a different relation edge and an absent case
  context. It is not an alias of `screencast`.

### Trace

1. The direct, supported block selected
   `edge:adobe presenter video express|used_for|screencast` with exact target
   attachment and score `0.9167`.
2. The plan then selected
   `edge:adobe presenter|used_for|presentation software` as a
   `sibling_elaboration`, attached by `soft` matching to target
   `adobe presenter video express`, score `0.4433`.
3. Again, the plan-level support check accepted an indirectly attached second
   fact solely because it was graph evidence. A focused lookup needs a stricter
   attachment requirement before rendering an expansion.

## Fix criterion

For a query with an explicit predicate filter, a rendered direct answer may
only contain blocks that attach **exactly** to a resolved target. Non-exact
attachments remain available for open synthesis and genuine graph-chain
explanations, but cannot extend a focused relation lookup. This is a general
support rule, not a case-specific subject or phrase list.
