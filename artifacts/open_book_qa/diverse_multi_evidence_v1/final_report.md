# Diverse multi-evidence planner repair

## Diagnosis

The frozen mixed-lane run had 6 failed unique cases. Intent decomposition and retrieval completeness tied at 3/6 each (50.0%). This pass selects only intent decomposition: it is a general parser/planner contract, while the tied retrieval class requires a separate fan-out policy.

| Failure class | Cases |
|---|---:|
| intent_decomposition | 3 |
| predicate_identification | 0 |
| retrieval_one_of_two | 3 |
| relevance_selection | 0 |

## Change

The semantic parser now represents structural two-fact requests as `multi_fact`, including both implicit cardinality forms and explicit coordinated-relation forms. The evidence planner receives a cardinality contract: before it has two distinct predicate groups, it cannot spend a second block on another object from the first group, and a fresh predicate remains valuable even when both facts share an object. Exact graph attachment and all evidence/audit gates are unchanged.

## Results

| Dataset | Before | After |
|---|---:|---:|
| Original frozen mixed-lane set (n=60, five repeats) | 0.900 | 0.950 |
| Relation-level novel validation (n=18, five repeats) | n/a | 0.944 |

Original explicit after: 0.933; implicit after: 0.967. 
Validation explicit: 0.889; implicit: 1.000.

## Remaining limitation

The repair intentionally addresses only intent decomposition. The original retrieval-completeness failures remain: when one required predicate group has more supported objects than the bounded answer plan can render, the planner may omit an object. The validation set is relation-level isolated from the original frozen set but has only five unused promoted subject neighbourhoods, so it is evidence of parser/planner generalization across novel relation pairs, not a broad new subject-distribution estimate.
