# Dialogue Context

This document describes Microworld's dialogue context layer. The layer exists
to make follow-up questions inspectable: a multi-turn question is reduced to the
same kind of semantic query as a single-turn question, with explicit reference
resolution and auditable state transitions.

## Role In The Runtime

Dialogue context is a core architectural layer, not a hidden chat log. It is
explicit semantic state over canonical entities, answer roles, surfaced
relations, topics, and reference bindings. The layer exists so a multi-turn
question can be reduced to the same kind of inspectable semantic query as a
single-turn question.

This state is not model memory. It contains pointers into known entities and
turn records, not new facts. It never writes accepted memory, never promotes an
overlay row, and never makes a trusted claim true. A reference such as `it`,
`he`, `that company`, `the founder`, or `the other one` is resolved over
semantic entities and dialogue roles, not over nearby text.

## Deterministic Resolver Path

The current dialogue path is intentionally deterministic:

- `DialogueState` is the only session memory for the v2 layer.
- `TurnRecord` is the only input that mutates `DialogueState`.
- `resolve_question(question, state, index)` is a pure resolver step before
  semantic parsing.
- Candidate selection uses hard type gates, integer salience scores, and a
  required margin.
- Every slot resolution carries candidates, score breakdowns, strategy, margin,
  and outcome.
- Every decision has an explanation: integer candidate scores plus the
  resolution trace that produced them.
- If any required slot is ambiguous or missing, the whole question audits.
- Topic shifts are explicit `topic_op` changes such as `("set", "Tesla")`.
- State is serializable with `to_dict()` / `from_dict()` and replayable from
  the committed turn records.
- The language renderer receives an already-resolved question or an audit. It
  does not decide reference identity.

The important boundary is semantic: dialogue context may select which existing
entity a later question refers to, but it may not create a fact about that
entity. When the resolver needs role evidence, it uses a narrow read-only
semantic role lookup only to choose among already-known dialogue candidates.
That lookup is trace-marked; it is not a write path.

## Example Dialogue

```text
Q1: Tell me about SpaceX.
State: topic_op=("set", "SpaceX")
A1: SpaceX is an aerospace manufacturer and space transportation company.

Q2: Who founded it?
Resolution:
  slot "it" -> SpaceX
  candidates:
    SpaceX total=191
      active_topic=100, last_answer_entity=40, last_question_subject=30,
      user_named=15, mentions=6
  margin: single typed candidate
  strategy: salience
A2: SpaceX was founded by Elon Musk.
State:
  surfaced relation: (SpaceX, founded_by, Elon Musk)
  role: Elon Musk entered as founded_by(SpaceX)
  active topic remains SpaceX

Q3: Tell me about Elon Musk.
State: topic_op=("set", "Elon Musk")
A3: Elon Musk is a businessman and entrepreneur.

Q4: What else did he found?
Resolution:
  slot "he" -> Elon Musk
  candidates:
    Elon Musk total=191
      active_topic=100, last_answer_entity=40, last_question_subject=30,
      user_named=15, mentions=6
  margin: single typed candidate
  strategy: salience
  exclusion: already surfaced SpaceX for founded_by
A4: Elon Musk founded Tesla, Neuralink, The Boring Company, xAI, Zip2, and Big Green.

Q5: What about Tesla?
Resolution:
  topic_shift "Tesla" -> Tesla
  reformulated question: Tell me about Tesla.
State: topic_op=("set", "Tesla"), previous_topic="Elon Musk"
A5: Tesla is an automotive and clean energy company.

Q6: Who founded it?
Resolution:
  slot "it" -> Tesla
  candidates:
    Tesla total=191
      active_topic=100, last_answer_entity=40, last_question_subject=30,
      user_named=15, mentions=6
  margin: single typed candidate
  strategy: salience
A6: Tesla was founded by Elon Musk, Martin Eberhard, Marc Tarpenning, JB
    Straubel, and Ian Wright.
```

The chain is `SpaceX -> it -> Elon Musk -> he -> Tesla -> it`, but the topic
does not move through hidden memory. It changes only when a committed turn says
so: first `SpaceX`, then `Elon Musk`, then `Tesla`.

## Ambiguity Handling

Ambiguity produces an audit rather than a best guess:

```text
Q: Tell me about OpenAI.
A: OpenAI was founded by Sam Altman, Greg Brockman, Ilya Sutskever,
   John Schulman, Wojciech Zaremba, and Elon Musk.

Q: What did he found?
Resolution:
  slot "he" -> unresolved
  candidates:
    Elon Musk total=43
      last_answer_entity=40, mentions=3
    Greg Brockman total=43
      last_answer_entity=40, mentions=3
    Ilya Sutskever total=43
      last_answer_entity=40, mentions=3
    John Schulman total=43
      last_answer_entity=40, mentions=3
    Sam Altman total=43
      last_answer_entity=40, mentions=3
    Wojciech Zaremba total=43
      last_answer_entity=40, mentions=3
  margin: 0, below required threshold
A: audit. unresolved_dialogue_reference
```

## Migration And Benchmarking

The migration path is explicit. `MICROWORLD_DIALOGUE_V2=shadow` runs the new
`DialogueState` resolver and commit logic in parallel with the older serving
behavior so divergences are logged before the new path drives answers. The
deterministic dialogue benchmark checks zero false resolutions, stable trace
output across repeated runs, and `DialogueState.replay(records) == live_state`
after each session. Compatibility tests keep single-turn QA byte-identical when
no dialogue slot is present, so adding dialogue context does not change the
ordinary one-question path.

Latest requested validation on this trimmed runtime copy, measured on
2026-07-09:

```text
python3 -m worldpgt.benchmarks.dialogue_benchmark
21 / 21 sessions passed; 138 resolver calls; mean 240.6 us/call
```

## Conclusion

Dialogue context is useful only because it stays bounded. It can bind `he`,
`it`, or `that company` to an explicit entity under a traceable policy, but it
does not create facts, promote memory, or let wording decide identity.
