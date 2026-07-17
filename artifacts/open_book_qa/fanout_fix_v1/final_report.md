# Fan-out completeness repair

## Root cause

The loss was in **plan construction**, not retrieval or rendering. In all three diagnosed frozen cases, the exact subject adjacency contained every required edge. The old planner emitted one block per edge, then stopped at four blocks or rejected a remaining object as low novelty. The renderer could only realize that incomplete plan.

The planner now creates one exact multi-object slot for complementary edges with the same attached subject and predicate. Each object retains its own evidence ID and source span in `object_slots`; a learned conflict still remains an uncertainty block. The renderer uses the existing deterministic list realizer to surface all objects in that slot.

## Diagnosed cases

| Subject | Ground truth objects by predicate | Returned before | Stop |
|---|---|---|---|
| Energy Drink Consumption Among Papuan Athletes | {'created_by': 6, 'published_by': 1} | {'created_by': 3, 'published_by': 1} | max_blocks_reached |
| SPSS | {'developed_by': 2, 'runs_on': 4} | {'developed_by': 2, 'runs_on': 3} | max_blocks_reached |
| LAMMPS | {'developed_by': 1, 'runs_on': 3} | {'developed_by': 1, 'runs_on': 2} | next best step (edge:lammps|runs_on|unix-like operating system) scored 0.229, under the continue threshold — answer value would not grow |

## Results

| Dataset | Accuracy before → after | Object recall before → after | Provenance after | Unsupported after |
|---|---:|---:|---:|---:|
| Frozen mixed-lane held-out (60 cases, 5 repeats) | 0.950 → 1.000 | 0.986 → 1.000 | 0.950 | 0.050 |
| Failure-disjoint fan-out stress set (11 cases, 5 repeats) | not previously measured → 1.000 | not previously measured → 1.000 | 1.000 | 0.000 |

The second row deliberately has no invented pre-fix number: it is a newly assembled, failure-disjoint stress set. Its post-fix result is the deciding generalization check for multi-object slots; its scope limitation is recorded in `validation.json`. The frozen-set provenance/unsupported values stay at 0.950/0.050 because five pre-existing off-target expansion cases remain outside this isolated fan-out change; the fan-out-specific set itself is 1.000/0.000.

## Cumulative multi-evidence validation series

| Stage | Result | Validation status |
|---|---:|---|
| Initial multi-evidence baseline | 0% | dataset-specific baseline |
| Early repaired dataset | 26% | dataset-specific |
| Dataset-specific post-repair | 93% | dataset-specific |
| Held-out v2, narrow predicate pair | 100% | held-out validated, narrow pair |
| Frozen mixed-lane set | 90% | held-out validated, diverse predicate pairs |
| After intent decomposition | 95% | same frozen held-out set; before fan-out repair |
| After fan-out repair | 100.0% | same frozen held-out set; all prior fan-out failures recovered |
| Failure-disjoint fan-out stress set | 100.0% | focused stress validation; not relation-level novel |

The 100% rows are scoped results, not an open-domain accuracy claim. This finishes the current multi-evidence validation series; no new data lane is opened by this experiment.
