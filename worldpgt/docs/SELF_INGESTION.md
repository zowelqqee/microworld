# Wikipedia Self-Ingestion v1

Wikipedia Self-Ingestion v1 is a safe offline self-feeding pipeline for local
Wikipedia-like documents. It proposes an overlay delta and verifies that the
proposal does not break controlled QA regressions.

It does not apply knowledge to trusted accepted memory.

```text
local Wikipedia-like docs
-> wiki-like pages
-> unchanged WikiIngestionV2
-> unchanged WikiCandidateOverlayBuilder
-> delta / duplicate / conflict / quarantine classification
-> proposed overlay delta
-> separate dry-run overlay
-> QA, adversarial, and cross-page regression gates
```

## Current Dry-run Status

| Metric | Value |
|---|---:|
| sources_total | 14 |
| URL sources rejected | 1 |
| documents_read | 14 |
| read_errors | 0 |
| candidates_total | 39 |
| new_candidates | 28 |
| duplicate_existing | 8 |
| conflicts | 2 |
| overlay_delta_items | 27 |
| quarantined_total | 9 |
| rejected_total | 4 |
| dry_run_overlay_items | 310 |
| safe_to_apply_overlay_delta | true |
| safe_for_general_runtime | false |

Overlay delta items:

| Type | Count |
|---|---:|
| entity | 5 |
| definition | 5 |
| relation | 3 |
| weak links | 14 |

The dry-run overlay is 310 items: 283 existing overlay items plus 27 proposed
delta items.

## Regression Gate

Dry-run QA against the 310-item overlay is green:

| Gate | Status |
|---|---:|
| Entity QA v1 | 28/28 |
| Entity QA expansion | 111/111 |
| Adversarial Entity QA | 68/68 |
| Cross-page Entity QA | 71/71 |

## Quarantine Reasons

The self-ingestion pass quarantines or rejects unsafe candidates instead of
promoting them:

| Reason | Count |
|---|---:|
| current_fact_without_as_of | 2 |
| weak_link_promoted_to_fact | 1 |
| inverted_relation | 1 |
| private_or_sensitive_data | 1 |
| unsupported_universal_claim | 1 |
| volatile_requires_source | 1 |
| entity_type_mismatch | 1 |
| conflicts_existing_fact | 1 |

## Safety Boundary

Self-ingestion v1:

- reads local files only
- rejects URL sources
- makes no Wikipedia/API calls
- never writes raw text directly into accepted memory
- keeps the dry-run overlay separate from the accepted wiki overlay
- leaves `accepted_knowledge_memory_v1.json` unchanged
- leaves `sense_memory.py` unchanged
- leaves ingestion extraction and overlay builder semantics unchanged
- does not lower thresholds or weaken validators
- keeps `safe_for_general_runtime=false`

## Next Step

The next planned step is Promote Overlay Delta v1: validate and promote the
safe self-ingestion overlay delta into a separate promoted overlay artifact,
without modifying trusted accepted memory or the current accepted overlay.
