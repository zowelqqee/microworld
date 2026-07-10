# Narrative reasoning implementation

## Architecture implemented

The experiment now supports this bounded optional path:

```text
prompt -> SceneGoal -> ScenePlan -> WorldState -> hypotheses -> beam search
       -> selected SentencePlan -> trigram/morphology/novelty realization
       -> realized_facts only -> committed WorldState
```

`--reasoning` enables the new path. The existing stateless path remains the
default, so existing CLI behavior and all production QA behavior stay intact.
Each candidate is evaluated copy-on-write. A fragment with no trustworthy
finite verb receives an explicit stateless fallback and cannot create a world
fact. Candidate selection now also removes a sentence carrying an out-of-scene
proper name whenever a safe realizable candidate exists.

Interactive narrative runs use a fresh route seed when `--seed` is omitted.
The seed changes bounded concept-field walks, learned transition markers, and
the selected speech-graph edge. Passing `--seed stable` makes the same route
replayable for audits. Two-token intransitive edges (for example
`Воланд остановился`) are valid events, so the planner is no longer limited to
the small set of exact subject–verb–object trigrams.

Within one engine session, unseeded repeated requests also reject recently
emitted identical paragraphs and resample up to eight valid routes. This gives
interactive generation diversity while keeping an explicit seed exactly
replayable.

Narrative ingestion accepts either one UTF-8 `.txt` file or a directory of
uploaded works:

```bash
python3 cli.py ingest-narrative --source corpus/uploaded
```

The artifact keeps exact speech edges and emits bounded
`universal_speech_frames`: action/object relations observed with multiple
subjects and safe oblique-case objects. These frames are a fallback for a
sparse character, so adding works increases coverage without treating every
inverted prose fragment as a fact.

Reasoning-mode realization now compresses a scene structurally: an opening
state is followed by a separate setup action, then by bounded two-clause
consequences (`… сделал попытку и …`). The public WorldState still receives
each realized action separately; sentence compression never merges facts.

## Files

- `poemcore/world_state.py`: temporal facts, entity state, frame persistence,
  retractions, proof steps, deterministic replay, and narrative violations.
- `poemcore/narrative_reasoning.py`: hypothesis lifecycle and named scores.
- `poemcore/plan_search.py`: deterministic bounded beam search.
- `poemcore/narrative.py`: optional bridge and realized-facts commit channel.
- `eval/trace_world_state.py`, `eval/eval_world_consistency.py`, and
  `eval/qa_compatibility.py`: manual trace, A/B evaluation, and QA report.

## Reuse and deviations

The implementation reuses production shapes: immutable fact provenance follows
`InferredFact`; form/test/accept/reject follows `thought_loop`; bounded
plan/audit behavior follows `path_planner`; entity tracking follows dialogue
state. `poemcore/transitions.py` also reuses production's JSON
`rule_interpreter` unchanged for the depth-1, timeless QA adapter; narrative
temporal rules remain intentionally separate.

The design was updated because surface realization only exposed role hits, not
semantic triples. `realized_facts` is the minimal bridge. This also exposed a
real corpus limitation: lightweight action detection can treat a name as a
verb. The factual bridge rejects such predicates.

## A/B evidence

`python3 eval/eval_world_consistency.py` on the five required ten-sentence
prompts produced these means:

| Metric | Stateless | WorldState |
| --- | ---: | ---: |
| Sentences | 10.0 | 10.0 |
| Unintroduced-entity rate | 0.00 | 0.00 |
| Event continuity | 0.022 | 1.000 |
| Realized-fact coverage | 0.00 | 0.48 |
| Sentence completion | 1.00 | 1.00 |
| Novelty | 1.00 | 1.00 |
| Deterministic replay | 1.00 | 1.00 |
| Runtime for five prompts | 1.361 s | 0.366 s |

All three adversarial transitions were rejected: unintroduced Behemoth,
teleportation, and bilocation. The report prints raw examples. The active-
subject candidate creates a genuine bounded choice for beam search, producing
the measured continuity gain without surface post-processing.

Reasoning-mode surface now uses only the selected event fragment or an explicit
non-event pause, with safe `он/она` continuation references. The remaining
known surface limitation is Russian object-case agreement in some learned
spans. The action/object guard is now artifact-driven: `ingest-narrative`
rebuilds `noun_like` from preposition and predicate-object evidence and emits
`verb_lexicon` with corpus-derived standalone, dative, and transitive action
sets. No production reasoning vocabulary is manually enumerated in
`narrative.py`; a new corpus can expand or shrink these sets on the next
ingest. Unknown cases conservatively fall back to a non-event pause.

## QA compatibility

`python3 eval/qa_compatibility.py` confirms production inference on a fixed
ownership-transitivity fixture and reports `compatibility_proof_succeeded:
true`: the same inferred triple and proof rule enter the generalized
hypothesis lifecycle as a depth-1 timeless configuration. No production QA
replacement was made.

## Test results

- `python3 -m unittest discover -s . -p 'test_*.py' -v`: **40 passed**.
- `python3 -m pytest worldpgt/tests/test_inference_engine.py -q`: **50 passed**.
- A broad runtime-root unittest discovery still has one pre-existing unrelated
  import failure in `test_reddit_community_context` because its experiment
  module is absent from this checkout; it was not modified by this work.

## Reproduction

```bash
cd poetry_lab
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 eval/trace_world_state.py
python3 cli.py narrate 'Напиши 10 предложений о Понтии Пилате' --sentences 10 --reasoning --trace
python3 eval/eval_world_consistency.py
python3 eval/qa_compatibility.py
```
