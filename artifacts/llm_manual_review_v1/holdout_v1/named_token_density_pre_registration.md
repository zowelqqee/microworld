# Pre-registration — named-token-density holdout test

Written before any human verdict is present in
`runs/arxiv_holdout_74_20260720/manual_review.md`.

This is a separate, broader test of the retrospective underlying signal. It
does not alter, relax, replace, or retrospectively reinterpret H1/H2.

## Frozen metric

For every candidate, `named_token_count` is the total number of tokens across
`subject` and `object` which:

1. match `[A-Za-z][A-Za-z0-9-]*`;
2. start with an uppercase letter; and
3. are not closed-class/common words such as `The`, `This`, `It`, `And`, or
   prepositions.

This is the same capitalized non-common token metric used in
`../pattern_analysis_v1/report.md`.

## Frozen threshold rule

`named_token_count >= 2` is the only threshold rule in this validation. It is
applied to all 81 candidates, independent of primary review group, predicate,
source phrase, H1, or H2.

## Evaluation commitment after full human review

1. Report the mean named-token count separately for manual ACCEPT and REJECT
   across all 81 candidates.
2. Report the same means within `generic_property_likely` and `clean_no_flag`
   where both verdict classes exist, so they can be compared with the original
   retrospective group means (ACCEPT 1.85 vs REJECT 0.97; ACCEPT 1.70 vs
   REJECT 1.05, respectively).
3. Report threshold precision, recall, TP, FP, FN, and TN for `>=2`.
4. Do not change the token definition or threshold after reviewing outcomes.
5. No automatic admission follows from any result; this remains an
   auto-priority / evidence-collection signal only.
