# Expanded independent CHAIN validation

## Protocol and ceiling

This is measurement-only. It scanned the composed current serving graph at `artifacts/ios_demo_v2/extended_serving_overlay.json`: 985 explicit relation IDs, representing the documented main, Crossref, Wikidata, and OpenAlex cohorts. A candidate is `A --p1--> B --p2--> C`, found solely by normalized object-to-subject identity. The grammar and both planners were not changed.

The scan found 44 safe paths across 15 unique starting subjects and 18 separately audit-expected paths. It then excluded every relation/evidence ID in all existing open-book datasets, `independent_multi_evidence_v1`, and the prior `beyond_old_enum` capability set (581 IDs total). The resulting fresh maximum is 7 safe paths across 6 starting subjects plus 2 audit-expected paths across one subject. Only 5 of the seven safe paths are non-degenerate (`C != A`); the remaining two are the reverse `New Energy Finance <-> Michael Liebreich` loop and are labelled in `cases.json` rather than silently treated as normal natural-language chains.

Therefore this graph cannot supply the requested n=25–30 independent CHAIN validation under the required zero-overlap constraint. The honest usable non-degenerate ceiling is **n=5**, not a deficiency of the grammar. Reaching the target requires graph/data scaling, which is outside this measurement-only task.

## Results

| Stratum | n | Existing multi-evidence exact | Grammar exact / correctly audited |
| --- | ---: | ---: | ---: |
| Fresh answer-expected paths | 7 | 0.143 (1/7) | 1.000 (7/7) |
| Fresh non-degenerate answer paths | 5 | reported in full traces | 5/5; descriptive only |
| Fresh audit-expected paths | 2 | not an accuracy denominator | 1.000 correctly audited (2/2) |

The old result is produced through the unchanged `answer_behavior.build_answer_plan`, scoped to the old multi-evidence planner and not the separate multihop subsystem. The grammar uses the unchanged CHAIN operator and shared safety validator. `results.json` retains selected evidence IDs and plan/audit traces for each case. The candidate construction filters overlap before either path runs; all frozen new cases consequently have zero overlap by relation/evidence ID with the specified prior material.

## Comparison with the prior small result

The previous capability result reported CHAIN 0.60 in its small mixed capability sample, including safety refusals as a distinct design outcome. This fresh zero-overlap slice is grammar 1.00 on 7 answer-expected paths and 1.00 correctly audited on 2 temporal/current-sensitive paths. It does not establish that 0.60 “rose” to 1.00: n=7 (and only n=5 non-degenerate) is even less adequate for a stable rate estimate. It instead confirms the direction of the original result and, more importantly, documents the present data ceiling.
