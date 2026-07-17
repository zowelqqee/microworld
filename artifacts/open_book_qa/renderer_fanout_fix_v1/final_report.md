# Renderer fan-out fix v1 — results

Goal: eliminate the 0.25 unsupported-claim rate on the independent_v1 held-out
paraphrase set, without losing its 0.875 answer accuracy and without disturbing
the profile/synthesis bundling behaviour or the already-published numbers.

## Localisation — planner, not renderer; and not the bundling mechanism

The task framed the bug as a renderer fan-out through the subject-locative
*bundling* mechanism. Tracing the four flagged cases showed a different,
upstream cause.

Each flagged answer carried a **second content block of a foreign predicate**,
emitted by the **planner** (`build_answer_plan` in
`worldpgt/reasoning/answer_behavior.py`) as a `sibling_elaboration` block — not
by the renderer, and not through `subject_locative_bundle` (whose own test
suite shows no involvement here):

| Question | Asked | Extra block the planner added |
| --- | --- | --- |
| By whom was Gyulaj Hunting Hungary set up? | founded_by | headquartered_in → Tamási |
| Who is the owner of Google Sheets? | owned_by | developed_by → Google (first block!) |
| For which purposes is Google Sheets intended? | used_for | (many-valued used_for — see below) |
| Where does Gujarat Vidyapith keep its head office? | headquartered_in | founded_by → Mahatma Gandhi |

The planner already has the correct guard: `build_answer_plan` accepts a
`predicate_filter`, and line ~1241 refuses any edge whose predicate is outside
it. The guard was simply **never armed** for these questions, because the
semantic parser returned `relation_intent = None` and
`relation_intents_from_text` returned an empty set — so `predicate_filter` was
`None` and every predicate of the subject was eligible.

Two distinct reasons the intent did not resolve:

1. **Missing cues.** `set up`, `owner`, `purpose`/`purposes`, and `intended
   for` were absent from `RELATION_KEYWORD_MAP`. The paraphrase question named
   its relation in words the keyword map did not carry.
2. **A nested-span false match.** "head office" resolved to `leader_of`,
   because the bare `head` leadership cue matched *inside* the phrase "head
   **office**". `relation_intents_from_text` collected every keyword hit,
   including fragments of longer phrases.

## The fix that shipped — Fix B only

`worldpgt/relation_extraction_v2/relation_policy.py`, two parts:

- **Added cues** to `RELATION_KEYWORD_MAP`: `set up`→founded_by,
  `owner`→owned_by, `head office`/`head offices`→headquartered_in,
  `purpose`/`purposes`/`intended for`→used_for.
- **Nested-span suppression** in `relation_intents_from_text`: matches are
  collected longest-keyword-first, and a shorter keyword whose span sits inside
  an already-claimed longer one is dropped as a phrase fragment. This mirrors
  the longest-match rule `relation_intent_from_text` already applied. Keywords
  matching *disjoint* spans still each contribute, so a genuinely coordinated
  question ("Who founded X and where is X headquartered?") still names both
  relations.

With the intent resolved, the planner's existing `predicate_filter` guard arms
itself and the foreign sibling block is refused at line ~1241. **No planner or
renderer code was changed.** The keyword-map change invalidates the per-verb
embedding cache hash, so `worldpgt/artifacts/embedding_*` regenerated on first
load (expected, non-behavioural).

## The fix that was reverted — Fix A

A `single_relation_scope` flag was prototyped on `build_answer_plan`: for a
focused question whose intent could not be named, lock the plan onto the first
selected relation group. It was **reverted** because the ablation showed it
*loses* answer accuracy and disturbs a published number:

| Paraphrase set | before | Fix B (cues) | Fix B + Fix A (gate) |
| --- | ---: | ---: | ---: |
| independent_v1 | 0.875 / 0.250 | **0.875 / 0.062** | 0.688 / 0.062 |
| heldout_v2 | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| heldout_v3 | 1.000 / 0.050 | 1.000 / 0.050 | 0.950 / 0.000 |
| main_dataset | 0.940 / 0.100 | 0.940 / 0.100 | 0.920 / 0.100 |

(format: answer accuracy / unsupported-claim rate)

Fix A converted answers that were *accidentally* correct — the right fact had
arrived as a sibling block precisely because the intent was unfiltered — into
honest audits (three cases on independent_v1 where the correct object was a
sibling of an unresolved-intent plan; one on the **published** heldout_v3). It
added nothing to the unsupported rate that Fix B had not already removed
(0.062 in both columns), while costing accuracy and touching a shipped result.
Per the project's own discipline ("if a change worsens accuracy, revert it"),
Fix A was dropped.

## Results — Fix B, every set

| Set / category | before | after (Fix B) |
| --- | --- | --- |
| independent_v1 / paraphrase | 0.875 / **0.250** | 0.875 / **0.062** |
| independent_v1 / negative | 1.000 / 0.000 | 1.000 / 0.000 |
| heldout_v2 / paraphrase | 1.000 / 0.000 | 1.000 / 0.000 |
| heldout_v2 / multi-evidence (both) | 1.000 / 0.000 | 1.000 / 0.000 |
| heldout_v3 / paraphrase | 1.000 / 0.050 | 1.000 / 0.050 |
| heldout_v3 / multi-evidence (both) | 1.000 / 0.000 | 1.000 / 0.000 |
| main_dataset / direct | 0.980 / 0.090 | 0.980 / 0.090 |
| main_dataset / paraphrase | 0.940 / 0.100 | 0.940 / 0.100 |
| main_dataset / negative | 1.000 / 0.000 | 1.000 / 0.000 |
| main_dataset / multi-evidence | 0.760 / 0.680 | 0.760 / 0.680 |

Only **independent_v1 unsupported moved: 0.250 → 0.062**, with accuracy held at
0.875. Every other set — including the published heldout_v2/v3 and the main
250-case dataset — is bit-for-bit identical. Three of the four fan-out cases
are fully resolved.

## The remaining 0.062 is not fan-out

The one still-flagged case is `For which purposes is Google Sheets intended?`.
The graph carries **two** `used_for` edges for Google Sheets ("collaborative
real-time editor" and "office suite"); the planner correctly renders the
many-valued group, and the dataset's expected set names only one of the two.
This is a *same-predicate* many-valued group plus a dataset expected-set
artifact — not a foreign-predicate fan-out. Fix A did not remove it either
(0.062 in both ablation columns) and should not: refusing a second object of
the *requested* relation would break legitimate multi-valued answers, which the
task explicitly protected. Left as-is and noted, not chased.

## Profile / synthesis bundling — confirmed intact

Direct API check (single frozen graph, both fixes' final state):

| Question | query_type | blocks | predicates |
| --- | --- | --- | --- |
| Tell me about Google Sheets | open_synthesis | 2 | developed_by, used_for |
| What do you know about Gujarat Vidyapith? | open_synthesis | 2 | founded_by, headquartered_in |
| Tell me two facts about Google Sheets | multi_fact | 3 | developed_by, owned_by, used_for |
| By whom was Gyulaj Hunting Hungary set up? | lookup | 1 | founded_by |
| Who is the owner of Google Sheets? | lookup | 1 | owned_by |

Profile and cardinality requests still gather multiple distinct relations;
focused lookups now return exactly the asked relation. Because the shipped fix
only resolves intents and never disables elaboration, bundling was never at
risk — the profile check simply confirms it.

## Tests

Changed: `worldpgt/relation_extraction_v2/relation_policy.py` (logic),
`worldpgt/tests/test_relation_policy_and_patterns.py` (new
`TestIntentCueCoverage`: cue resolution, `head office` nested-span suppression,
bare-`head`→leader_of preserved, disjoint coordinated cues both survive).
Regenerated: `worldpgt/artifacts/embedding_*` (cache).

Passing suites (parser / planner / renderer / bundling): `test_relation_policy_and_patterns`,
`test_answer_behavior_v1`, `test_semantic_question_parser`,
`test_predicate_centroid_index`, `test_subject_locative_bundle_v1`,
`test_open_book_qa_heldout_v1`, `test_open_book_qa_comparison_v1` — **177 passed**.

Pre-existing failures, unchanged by this work (verified identical by stashing
the change and re-running on HEAD):

- `test_multihop_qa_v1`: 1 failed + 9 errors — artifact-dependent, present on
  clean HEAD.
- `test_answer_behavior_api_v1::test_answer_plan_blocks_are_evidence_traceable_over_experimental_graph`:
  1 failed on clean HEAD.

These are part of the known 14 failed + 14 errors baseline recorded in the
previous session; they are not regressions from this change.

## Reproduce

Runs in this directory: `before/` (baseline), `fixB_only/` (shipped), `final/`
(shipped, re-run without the monkeypatch to confirm `fixB_only == final`).
`microworld_results.jsonl` is the shipped independent_v1 run.
