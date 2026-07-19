# arXiv predicate analysis — retrospective

## Scope and reproducibility boundary

No network fetch, new source record, gate change, or promotion was performed. The analysis reads the stored arXiv allowlist and stored source records and deterministically replays the already-reported source-specific support check in memory so that the report's aggregate counts can be decomposed. The saved full-run report records 85 unique quarantine relations. The current archive traversal returns 86 source+URL-distinct allowlist rows; it nevertheless reproduces the reported `28 extracted / 25 gate accepted` totals. This is a one-row input-drift discrepancy in the historic quarantine traversal, not a changed gate result; it should be retained as an audit caveat.

## Predicate inventory

The current stored allowlist has five predicate types:

| Predicate | Allowlist rows |
| --- | ---: |
| `uses` | 37 |
| `enables` | 30 |
| `supports` | 12 |
| `provides` | 4 |
| `located_in` | 3 |

Thus arXiv is not authorship-only: neither `created_by` nor another authorship predicate occurred. It also does not expose a broad semantic range in this lane. Its parser's supported cue vocabulary is structurally narrow (`uses`, `enables`, `supports`, `provides`, `works_by`, `located_in`), and the stored 85/86-row pool realizes only five of those. There are no extracted methodology, findings, citation, or topic predicates as distinct schema types in this run.

The stricter supported-and-atomic subset narrows further: 28 rows were extracted (`uses`: 24, `enables`: 2, `supports`: 2). The existing gate accepted 25 (`uses`: 23, `enables`: 2).

## Accepted duplicates

Every accepted row duplicated the **same predicate group** already present for its subject; there is no case of an accepted arXiv predicate duplicating a different predicate name.

| Subject | Extracted predicate | Existing group already present |
| --- | --- | --- |
| Another approach | uses | uses |
| AutoGrad | uses | uses |
| AutoSlim | uses | uses |
| Classical physics | uses | uses |
| CYSEC | uses | uses |
| DABFT | uses | uses |
| Declarative modeling | uses | uses |
| DPoEV | uses | uses |
| EDCIM | uses | uses |
| FrontierMath | uses | uses |
| Fuzzy logic | uses | uses |
| Gradient flow decomposition | uses | uses |
| Hamiltonian reshaping | uses | uses |
| HotFuzz | uses | uses |
| MAVIS | uses | uses |
| Multimodal remote sensing | enables | enables |
| Otherwise the paper should | uses | uses |
| Previous work | uses | uses |
| PyOptInterface | uses | uses |
| Quantum neural networks | uses | uses |
| REGAI | uses | uses |
| SciServer | uses | uses |
| SHAZAM | uses | uses |
| TrackOR | uses | uses |
| WiCoM | enables | enables |

## The other three extracted rows

The arithmetic is `28 extracted = 25 accepted duplicates + 3 gate rejects`, not 25 duplicates plus three accepted non-duplicates:

| Subject | Predicate | Gate outcome |
| --- | --- | --- |
| Artificial intelligence | supports | `generic_relation_bad_subject` |
| Finite simulation | supports | `generic_relation_bad_subject` |
| KRAKENS | uses | `concept_relation_bad_object` |

## Conclusion

The evidence does **not** support an authorship-only source limitation. It supports a narrower conclusion: this arXiv lane, as implemented, produces a small action/capability vocabulary dominated by `uses`; the supported subset is even more concentrated (25/28 are `uses` or `enables`). The complete duplicate outcome is partly sample-specific — these 47 subjects already had those groups — but the source/extractor envelope is also genuinely narrow, so it is not evidence that arXiv will automatically contribute methodology, finding, citation, or topic predicates.

Trying arXiv on a different subject subset is plausible only if targets are selected to lack `uses`/`enables` and have clean named subjects plus bounded atomic objects. It is unlikely to unlock broad predicate diversity without a separately designed and validated extractor for additional abstract-specific relation types; that would be new extraction work and is outside this retrospective analysis.
