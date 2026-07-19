# Compositional grammar v1 — final report

## Result

The isolated `AND`/`CHAIN` grammar is viable as an experiment, but is not ready to replace the established production path. It should remain side by side while a real retrieval-level A/B is added.

## Parity check

The new runner evaluates each frozen case only against its own frozen evidence slice. This measures composition, refusal, and provenance; it does **not** measure entity resolution or production retrieval.

| Set | New grammar cases | Answer accuracy | Exact evidence provenance | Unsupported claims |
| --- | ---: | ---: | ---: | ---: |
| heldout_v2 multi-evidence | 20 | 1.00 | 1.00 | 0.00 |
| heldout_v3 multi-evidence | 20 | 1.00 | 1.00 | 0.00 |
| fan-out stress | 11 | 1.00 | 1.00 | 0.00 |
| independent_paraphrase_v1 | 0 multi-evidence cases | n/a | n/a | n/a |

The published post-fix baseline reports 1.00 answer accuracy/object recall on its mixed-lane and fan-out evaluations. On the overlapping frozen composition task, this experiment matches that result. The independent artifact is paraphrase-only, so calling it a grammar comparison would be misleading.

## New capability

`capability_cases.json` records 14 executable synthetic cases: twelve `AND` combinations spanning predicates not encoded as one existing hardcoded pair, plus two `CHAIN` forms. `test_compositional_grammar_v1.py` executes them as a parameterized suite and additionally checks fan-out, missing support, object-to-subject joining, and unsafe-hop audits. New predicate types join `AND` without an operator-code change: only evidence and, when natural-language phrasing is desired, a lexical cue are needed.

## Honest assessment and recommendation

The operator layer is small and clearer than a growing query-pair enum; the gain is real for structured callers and newly pumped predicate combinations. Its current parser is intentionally conservative and still needs lexical cues for older paraphrased prompts. More importantly, the benchmark bypasses live retrieval by design. Therefore replacing the old route would overstate the evidence.

Recommendation: keep both paths parallel. Use this grammar for structured/experimental A/B calls, add a live-overlay retrieval benchmark (including CHAIN cases), then consider routing only if it preserves the existing end-to-end metrics and audit behavior.

## New capability validation

`beyond_old_enum_cases.json` is a new frozen 10-case set from the main accepted/promoted overlays and Crossref/Wikidata/OpenAlex promoted slices. It contains five three-predicate `AND` requests and five two-hop `CHAIN` requests. `beyond_old_enum_results.json` contains the complete plan trace for both systems. The comparison invokes the unchanged `answer_behavior.build_answer_plan` directly for the old multi-evidence planner, with no API/dialogue layer and no multihop route; that is the relevant old path for this question.

| System | All 10 exact | AND (5) | CHAIN (5) |
| --- | ---: | ---: | ---: |
| Existing multi-evidence planner | 0.40 | 0.80 | 0.00 |
| Grammar v1 | 0.80 | 1.00 | 0.60 |

The source inspection changes one part of the initial premise: the existing planner is already predicate-agnostic for explicit `AND` (`predicate_filter` is a set), so three-predicate AND is **not** proof that it has a hardcoded pair enum. Four of five AND traces therefore succeed on the old planner. The one failure is `Large Language Model`: its bounded four-block plan omits part of the requested fan-out, while grammar retains every evidence member of each requested predicate group.

The clear new capability is CHAIN in the scope requested here. For example, `Elon Musk --founded--> SpaceX --develops--> rockets` is answered by grammar with both edge IDs. The old multi-evidence trace selects unrelated direct neighbourhood edges of Elon Musk and never constructs the object-to-subject join; it is not a CHAIN plan. The same occurs for the SpaceX, Neuralink, and Blue Origin cases. Two grammar CHAIN cases (`leader_of`) correctly audit because the shared safety policy marks that current-sensitive relation unsafe without a snapshot/as-of value. This is a safety refusal, not a success claim.

This is significant evidence of expanded compositional capability specifically for structured two-hop composition versus the old multi-evidence planner. It is not a claim that grammar supersedes the separate existing `multihop_qa` subsystem, which was deliberately disabled from this comparison.

### Independent multi-evidence v1

`independent_multi_evidence_v1_cases.json` adds ten AND cases whose subjects are asserted disjoint from every existing `artifacts/open_book_qa/**/dataset.jsonl` subject. The grammar result is 0.70 exact with three safety audits (rather than fabricated answers); the full per-case provenance is in `independent_multi_evidence_v1_results.json`. It closes the former N=0 gap, but is a small curated independence check, not a population-level estimate.
