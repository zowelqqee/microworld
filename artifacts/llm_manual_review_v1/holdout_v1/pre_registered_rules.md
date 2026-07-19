# Pre-registered H1/H2 rules — holdout v1

Written: 2026-07-20, **before selecting, inspecting, extracting from, or
reviewing holdout material**.

This file is a commitment device. H1 and H2 below are copied unchanged from
`../pattern_analysis_v1/report.md`. They must be applied mechanically to the
holdout candidates before any human verdict is recorded. No threshold, token
class, predicate, source-phrase, group restriction, or Boolean combination may
be changed based on holdout observations.

## H1 — technical `uses` shape (clean/no-flag only)

A candidate matches H1 **iff** all of the following are true:

1. `primary_review_group == "clean_no_flag"`;
2. `predicate == "uses"`; and
3. `object` contains at least one structural technical marker:
   - a capitalized non-common token;
   - an acronym;
   - a digit; or
   - a hyphenated token.

For this rule, a capitalized non-common token is an alphabetic token beginning
with a capital letter other than closed-class/common tokens such as `The`,
`This`, `It`, `And`, and prepositions. An acronym is two or more uppercase
letters, optionally with digits or hyphens.

## H2 — enumerated `supports` shape (generic-property only)

A candidate matches H2 **iff** all of the following are true:

1. `primary_review_group == "generic_property_likely"`;
2. `predicate == "supports"`; and
3. the literal `source_text` contains `tasks such as` **or** `applications
   such as` (case-insensitive).

## Evaluation commitment

- Report H1, H2, and `H1 OR H2` separately.
- Primary measure: precision among selected candidates after complete manual
  review of the whole holdout batch.
- Retrospective reference is 95.0% precision (19/20) for `H1 OR H2`.
- A holdout precision near 80% or above is evidence of a useful auto-priority
  signal only; it is never automatic admission.
- A lower holdout result will be reported as an honest overfitting finding.
- No candidate may be promoted, no serving memory may change, and no commit
  may be created in this validation lane.
