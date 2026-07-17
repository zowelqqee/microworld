# OpenAlex topic/citation lane — final report

## Scope and boundary

This run used the official OpenAlex Works API only. It began from 7 unique quarantined OpenAlex relations and 7 DOI seeds, then fetched each seed work and at most one named cited work per seed. The run made 13 API requests. All outputs remain proposal-only: accepted memory and the serving overlay were not modified.

## Gate result

| Stage | Count |
|---|---:|
| Raw candidates | 12 |
| Passed source gate | 12 |
| Passed v1 + v2 precision gates | 4 |
| Entities with >=1 accepted relation | 2 |
| Entities with >=2 accepted predicate groups | 2 |

Rejection/quarantine reasons: `{"generic_relation_bad_subject": 8}`.

## Predicate-type composition

The precision-accepted multi-predicate composition is:

- `has_topic+references_work`: 2

Accepted entities:

- `The Economics of Superstars` (`https://openalex.org/W2119710370`): `has_topic + references_work`
- `Rethinking the Orality-Literacy Paradigm in Musicology` (`https://openalex.org/W2126248690`): `has_topic + references_work`

This is structurally different from Crossref's `created_by + published_by`: it combines an OpenAlex topic classification (`has_topic`) with a citation-graph relation (`references_work`). It is therefore not a second source for the same author/publisher fact pattern.

## Overlap checks

- Pre-Crossref main target-subject overlap: 0 / 331.
- Main relation-ID overlap: 0.
- Promoted Crossref DOI entity overlap: 0.

## Recommendation

The lane demonstrates a structurally distinct predicate pair but is too small for a meaningful generalization held-out stratum. Keep it proposal-only and seek more OpenAlex work records or a citation-graph source before evaluating.
