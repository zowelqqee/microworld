# Held-out v1: blocked before run

## Status

No held-out benchmark was run. `run_count` is zero and neither MicroWorld nor
Qwen result files exist for this control set.

The main v2 dataset was read only to make an opaque set of `relation_ids` and
`evidence_ids`; its questions, expected answers, templates, rejected candidates,
and failure-analysis artifacts were not used to select held-out cases.

## Disjointness audit

After programmatically excluding every subject-predicate group containing a
main-v2 relation/evidence ID, 331 clean groups remain. They belong to 331
different subjects: there are zero subjects with two clean predicates.

Therefore neither the required 10 explicit nor 10 implicit multi-evidence
questions can be formed on the current overlay without reusing a main-v2
relation or artificially expanding the overlay. Creating an implicit question
with an arbitrary hidden pair would recreate the ambiguity this control was
meant to remove.

## Relation-density audit

The missing resource is not generic text volume. The leakage-free pool has a
flat relation-density distribution: 331 subjects have one predicate group,
and zero have two or more. Multi-evidence evaluation needs two independently
grounded predicate groups for the same subject.

The proposal-pump audit found 313 prior candidates for these same subjects:
259 relation candidates were quarantined as
`open_web_relation_requires_source_specific_extractor`, and 54 candidates
were rejected as `open_web_subject_not_source_title`. Those are useful
acquisition leads, not evidence to promote: the current public-source lane
intentionally quarantines generic relation extraction from abstracts until a
source-specific extractor validates it.

This composed evaluation overlay is sourced from the evidence-grounded
open-web campaigns, not from the local Wikipedia snapshot lane. Of its 331
targets, only 2 match one of the 100 local snapshot titles, so a local-only
Wikipedia re-extraction cannot materially repair this particular held-out
pool.

The next acquisition pass should target these existing subjects and add only
new, independently evidenced predicate types. Progress is measured by the
post-exclusion `relation_density_distribution` in `dataset_summary.json`, not
by the number of fetched pages.

## Decision

The held-out run is intentionally not started. This is a limitation of the
available disjoint relation pool, not a MicroWorld or Qwen score. A valid run
requires a new independently collected evidence overlay, after which this
same builder can create the 20 paraphrase + 10 explicit + 10 implicit split and
run it exactly once.
