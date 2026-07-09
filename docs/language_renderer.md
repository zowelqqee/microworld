# Language Renderer

This document covers Microworld's controlled language generation layer. The
renderer is intentionally downstream of semantic support: it may vary phrasing,
style, and explanation shape, but it does not decide what is true.

## Text Generation Experiment

Microworld is testing a non-neural approach to language generation over
verified semantic support.

The working principle is:

```text
facts are not generated
speech is generated
```

Instead of predicting the next token from neural weights, the experimental
speech layer can choose the next allowed speech unit from explicit state:

- the user's semantic question
- the current semantic entity
- the verified relations and facts already selected by the planner
- what the answer has already said
- the requested answer style
- deterministic safety and support checks

The goal is LLM-like surface behavior without moving truth into an opaque
model. The generated wording may vary, but every factual claim still has to
come from accepted memory, an accepted overlay, or a clearly labelled
proposal/snapshot source. This is an experiment in controlled text generation,
not open-domain language modeling and not a neural model replacement.

## Speech And Reasoning Layer

The current breakthrough is not that Microworld "knows everything." It does
not. The stronger result is architectural: semantic support, reasoning,
dialogue, and speech are now separate enough to test independently.

```text
semantic memory rows
  -> semantic speech plan
  -> explicit reasoning trace
  -> action plan: answer / answer_with_gap / audit / no
  -> semantic language renderer
  -> surface validator + benchmark metrics
```

The reasoning layer operates over an already-built speech plan. It does not
query raw text, invent facts, or decide truth from phrasing. Its job is to make
the semantic answer decision inspectable:

- detect whether the user is asking for a profile, relation, path, or mechanism
- decompose the task into subgoals
- check whether required evidence roles exist
- name missing evidence, especially mechanism gaps
- choose an action such as `answer`, `answer_with_gap`, or `ask_clarification`
- forbid unsupported claims from entering speech

The speech layer then turns that bounded reasoning state into ordinary English.
It can say a useful partial answer such as "I can identify Starlink, but I do
not yet have the mechanism" without pretending it knows how Starlink works.

Recent phrase-graph changes keep that layer deterministic while making the
surface less brittle:

- `word_types` learns a reverse index from definition words to canonical entity
  types during the existing overlay-training pass; this is corpus-induced
  support, not a hardcoded taxonomy or embedding lookup.
- `fragment_variant(node, seed)` chooses among eligible phrase fragments with a
  deterministic frequency-weighted hash seed, so repeated runs are stable while
  wording can vary where the corpus supports it.
- Fragment eligibility is grammar-gated: only finite verb forms participate in
  `It {fragment}` variants, so learned but ungrammatical fragments do not reach
  the renderer.
- Consecutive capability facts such as `develops`, `produces`, and `operates`
  render as one combined sentence instead of several short repeated sentences.
- Facts that describe the same rendered predicate from opposite directions,
  such as forward `founded_by` plus inverse `founded`, are merged before
  rendering, so co-founder facts read as one founded-by clause.

The corpus is still small and predicate coverage is uneven. For example,
`introduced` is recognized as a historical release/date predicate in the
extraction and relation-policy layers, but the current promoted overlay does
not make that a broad, well-covered speech capability by itself.

## Important Modules

| Layer | Code | Role |
|---|---|---|
| Assistant orchestrator | `worldpgt/assistant_surface/answer_orchestrator.py` | routes requests, chooses memory/search/community path, attaches traces |
| Style normalizer | `worldpgt/assistant_surface/answer_style.py` | handles brief/simple/detailed style requests without changing facts |
| Speech planner | `worldpgt/entity_qa/semantic_speech_planner.py` | turns supported semantic facts into roles such as definition, activity, purpose, mechanism |
| Reasoning engine | `worldpgt/cognition/reasoning_engine.py` | builds explicit reasoning trace and action plan |
| Thought loop | `worldpgt/cognition/thought_loop.py` | rejects unsupported direct mechanism answers and accepts gap fallback |
| Deliberation/support guard | `worldpgt/cognition/deliberation_engine.py`, `support_guard.py` | prevents unsupported conclusions |
| Decision speech | `worldpgt/cognition/decision_surface.py` | human-facing phrasing for gaps, thin profiles, and clarification |
| Symbolic speech renderer | `worldpgt/entity_qa/symbolic_text_generator.py` | emits bounded English from the speech plan |
| Phrase transition store | `worldpgt/cognition/phrase_graph.py` | stores deterministic phrase fragments and transitions for rendering |
| Surface selection | `worldpgt/cognition/surface_selection.py` | rejects debug-like/repetitive variants and chooses cleaner speech |
| Semantic thought state | `worldpgt/cognition/semantic_thought_graph.py` | represents task, evidence, gap, and pattern state for cognitive moves |

This is why `How does Starlink work?` can honestly answer with a gap: the
system has enough facts to identify Starlink and its service, but no admitted
mechanism evidence role. The answer is useful because it separates "what I
know" from "what I do not know."

## Answer Styles

The renderer supports lightweight style hints. These do not change facts; they
only change selection and phrasing.

```text
коротко про SpaceX
самое важное про Tesla
простыми словами How does Starlink work?
подробнее про Elon Musk
```

Example:

```text
Q: коротко про SpaceX
A: SpaceX is an aerospace manufacturer and space transportation company. It
   develops rockets, spacecraft, and launch vehicles.
```

## Speech Quality Benchmark Contract

`benchmark_speech_quality_v1.py` measures the answer surface, not factual
coverage. It treats the semantic planner as an explicit-memory lookup and
checks whether speech stays natural, honest about gaps, non-repetitive, and
free of debug/internal wording.

It records row-level diagnostics:

```text
question
decision / route / support_kind / source_system
answer_text
latency_ms
debug_like
repetitive
honest_gap
decision_mismatch
missing_required_text
flags
```

Current suites:

| Suite | Purpose | Questions | Result |
|---|---|---:|---:|
| `smoke` | fast contract check | 12 | green |
| `large` | broad speech/reasoning baseline | 50 | 50 / 50 |
| `stress` | deterministic load/stability suite | 1,000 | 1,000 / 1,000 |

Stress category coverage:

| Category | Passed |
|---|---:|
| profile | 304 / 304 |
| direct_relation | 162 / 162 |
| mechanism_gap | 114 / 114 |
| adversarial | 72 / 72 |
| missing_or_current | 72 / 72 |
| thin_profile | 57 / 57 |
| style_control | 57 / 57 |
| connection | 54 / 54 |
| private_info | 54 / 54 |
| unsupported_universal | 54 / 54 |

## Conclusion

The renderer is a controlled speech layer over admitted semantic support. That
separation is what lets Microworld improve wording, style, and explanation
quality without treating language fluency as factual evidence.
