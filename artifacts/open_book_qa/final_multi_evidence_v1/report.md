# Final mixed-lane multi-evidence report

## Protocol

Frozen dataset; warm-up 50; seeded order 42; five measurement repeats; one execution per system. No run-time fixes or reruns were made.

## MicroWorld vs Qwen

| System | Category | Accuracy | Provenance | Unsupported |
|---|---|---:|---:|---:|
| MicroWorld explicit graph runtime | multi_evidence_explicit | 0.900 | 1.0 | 0.000 |
| MicroWorld explicit graph runtime | multi_evidence_implicit | 0.900 | 0.8666666666666667 | 0.100 |
| Qwen2.5-0.5B-Instruct 4-bit | multi_evidence_explicit | 0.700 | n/a | 0.000 |
| Qwen2.5-0.5B-Instruct 4-bit | multi_evidence_implicit | 0.667 | n/a | 0.000 |

## Lane breakdown

| System | Category | Lane | Accuracy | Provenance | Unsupported |
|---|---|---|---:|---:|---:|
| microworld | multi_evidence_explicit | crossref | 0.957 | 1.0 | 0.000 |
| microworld | multi_evidence_explicit | openalex | 1.000 | 1.0 | 0.000 |
| microworld | multi_evidence_explicit | wikidata | 0.667 | 1.0 | 0.000 |
| microworld | multi_evidence_implicit | crossref | 1.000 | 0.8695652173913043 | 0.130 |
| microworld | multi_evidence_implicit | openalex | 1.000 | 1.0 | 0.000 |
| microworld | multi_evidence_implicit | wikidata | 0.500 | 0.8333333333333334 | 0.000 |
| qwen | multi_evidence_explicit | crossref | 0.783 | n/a | 0.000 |
| qwen | multi_evidence_explicit | openalex | 1.000 | n/a | 0.000 |
| qwen | multi_evidence_explicit | wikidata | 0.333 | n/a | 0.000 |
| qwen | multi_evidence_implicit | crossref | 0.739 | n/a | 0.000 |
| qwen | multi_evidence_implicit | openalex | 0.000 | n/a | 0.000 |
| qwen | multi_evidence_implicit | wikidata | 0.500 | n/a | 0.000 |

## Conclusion

A 100% multi-evidence result did not hold on this mixed-predicate held-out set (MicroWorld category mean: 0.900). See the lane table for the failing stratum if any.
