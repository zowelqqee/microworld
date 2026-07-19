# Deterministic node-quality filter v1 — design and pilot gate

## Scope

This is a model-agnostic, rule-based **pre-gate** layer for raw LLM triples. It
does not change `relation_candidate_validator.py`, relation policy, serving
overlays, or accepted memory. It runs before the existing precision gate and
rejects nodes which are unsuitable to present to that gate.

The pilot reuses all three same-day extraction outputs: OpenAI `gpt-4o-mini`,
Gemini 2.5 Flash, and Gemini 3.1 Flash-Lite. It contains 101 raw triples total,
not an invented benchmark.

## Existing infrastructure reused

- `EntitySurfaceIndex` is instantiated from the same accepted, promoted, and
  snapshot overlays used by the serving stack. A capitalized string is not
  treated as a durable node merely because it looks name-like.
- The existing `relation_candidate_validator` remains the only precision gate.
  It is run unchanged both before and after the new layer.
- Existing extractor helpers already reject bare pronouns/demonstratives and
  malformed spans. They are intentionally not modified; this layer adds the
  LLM-specific authorial, event, list-context, and exact-resolution checks.

## Declarative rule set

| Rule | Deterministic condition | Example rejected |
|---|---|---|
| `authorial_self_reference` | Subject is `we`, `this paper`, `this study`, `our approach`, or an analogous proposed-work phrase | `this paper presents NB2Slides` |
| `generic_abstract_node` | A generic noun head has no named qualifier | `models`, `statistical uncertainties` |
| `event_like_node` | Node is a gerund/infinitival/event construction | `identifying vulnerabilities` |
| `list_derived_context` | Extracted node occurs in a comma-list after a colon in its evidence sentence | `Nav-Q is used for …` |
| `unresolvable_entity` | Endpoint does not exactly resolve through the current serving `EntitySurfaceIndex` | `AutoSlim`, `SciServer`, `Random Forest classification` |

Rules accumulate reasons rather than hide a secondary failure. A unit test also
shows that two resolved named nodes (`SciServer`, `SkyServer`) pass the layer;
the all-rejected pilot result is about current graph coverage, not a rule that
blindly rejects proper names.

## Before/after results

| Run | Raw triples | Raw existing-gate admitted | Pre-filter kept | Filter + existing-gate admitted |
|---|---:|---:|---:|---:|
| OpenAI `gpt-4o-mini` | 33 | 0 | 0 | 0 |
| Gemini 2.5 Flash | 36 | 0 | 0 | 0 |
| Gemini 3.1 Flash-Lite | 32 | 0 | 0 | 0 |
| **Total** | **101** | **0** | **0** | **0** |

Pre-filter reason counts across the 101 candidates (a candidate can carry more
than one reason): `unresolvable_entity` 101, `generic_abstract_node` 40,
`authorial_self_reference` 26, `event_like_node` 10, and
`list_derived_context` 9. Full row-level outputs are in `results.json`.

## Interpretation and gate verdict

The layer accurately catches every named failure class, but it does **not** meet
the requested admission-improvement gate. The pre-filter's exact-resolution
rule rejects even the manually clean proposals (`SciServer builds upon
SkyServer`, `AutoSlim uses Random Forest classification`) because neither
endpoint is in the current serving entity index. Consequently the after-filter
admission rate is undefined (0 survivors), not a misleading claimed 30%.

**Verdict: honest stop.** Do not broaden LLM extraction or promote any triple.
The node-quality problem is deeper than generic/authorial/list rules: the next
separate design task is a deterministic **entity-resolution / candidate-node
seeding lane** for named, source-grounded unknown surfaces, followed by the
same unchanged precision gate. That lane must be evaluated independently; this
filter must not be weakened to make the current metric look better.
