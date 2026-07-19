# Constrained-creative generation v1 — design

**Status: plan only. No code, no generation, no change to the existing Creative mode.**
This is an isolated research path awaiting review before any implementation. It does not
replace, call, or modify `worldpgt/cognition/creative_generator.py`, the QA support gate,
or any production route. When built it will live under a new `constrained_creative`
module and its own experiment runner, parallel to the existing paths — the same way
`compositional_grammar_v1` was kept parallel to the multi-evidence planner.

---

## 1. Research question

Following the framing style of the whitepaper's factual-QA question, stated so that it is
**measurable rather than a matter of taste**:

> Can constrained-creative generation — text that must stay grounded in a **bounded,
> explicitly supplied set of N facts** about a subject, using all of them and asserting
> nothing beyond them, while still reading as connected prose — **outperform an LLM's free
> generation on factual-constraint adherence** (fact inclusion, fact fidelity, and
> non-hallucination), *while the LLM is expected to outperform on generation
> fluency/naturalness*?

Explicit non-goal: this is **not** "who writes better prose." Prose quality is subjective
and unwinnable as stated, and we do not have human eval. The claim under test is narrow and
objective: **who better respects an explicit inclusion/exclusion constraint while producing
readable text.** Fluency is measured only by a labelled proxy and is expected to favour the
LLM — that asymmetry is part of the hypothesis, not a result to hide.

This slots between the two existing modes and is distinct from both:
- **Pure factual QA** — grounded, but no creative freedom (answers a specific question).
- **Current Creative mode** — creative freedom, but *no grounding constraint* (its gate
  rejects recitation, and it never guarantees which facts appear).
Constrained-creative asks for **both** controlled recombination **and** explicit grounding.

---

## 2. What already exists vs. what is new

### Reviewed existing infrastructure
- `worldpgt/cognition/creative_generator.py` — order-1/order-2 (bigram/trigram) word
  transition tables, opener stats, seeded deterministic weighted pick
  (`seeded_weighted_pick`), and a **4-gram novelty gate** (`_allows_next`,
  `contains_seen_4gram`) that enforces *recombine, never recite*. Trains from local prose
  / a literary corpus. `_boost_field` gives a **soft, probabilistic** bias toward seed
  concepts via spreading activation.
- `poetry_lab/poemcore/novelty.py` — documents the *inverted support gate*: QA rejects
  output **not** grounded; the creative gate rejects output that **reproduces** the corpus.
- QA side: `support_guard` (the precision/support gate) rejects any claim not backed by
  accepted memory; the whitepaper/benchmarks report this as *unsupported-claim rate*.
- Comparison harness: `worldpgt/experiments/final_multi_evidence_qwen_checkpoint.py` and
  the `artifacts/open_book_qa/**` datasets, with the frozen baseline
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (evidence-only prompt, resumable
  checkpointing).

### Why the existing generator is not sufficient as-is
The current creative path optimises the **opposite polarity** of what we need:

| | Existing Creative mode | Constrained-creative (new) |
|---|---|---|
| Gate polarity | reject *recitation* (4-gram) | require *inclusion of N facts*, reject *anything extra* |
| Concept steering | soft probabilistic boost | hard inclusion constraint |
| Grounding | none (labelled non-factual) | must use only graph facts about X |

Crucially, the 4-gram novelty gate actively works *against* fact inclusion: a fact often
lexicalizes as a multi-word span that may itself be a seen 4-gram, and the gate would steer
the walk away from it. So the mechanism cannot simply be reused with the boost turned up —
the gate's job changes.

### (a) Reused as-is
- `seeded_weighted_pick` — deterministic replayable choice primitive.
- `CreativeModel` transition/opener tables and `build_creative_model` — but used only for
  **connective tissue** between fact clauses, not to invent the facts.
- The **language renderer** (existing lexicalization of `(subject, predicate, object)`
  triples) — to turn each required fact into a surface clause.
- The Qwen harness + `open_book_qa` datasets/subject lists — for the comparison and for an
  independence check.
- The order-2 (trigram) tables double as the **grammaticality proxy** (see §3).

### (b) New — a post-hoc constraint layer
The core new mechanism is a **constraint-checking layer applied to generated text** — the
creative analogue of the support/precision gate, but **post-hoc on the output** rather than
**pre-hoc on extracted relations**:
1. **Constraint spec type** — `{subject, required_facts: [(p, o)...], N}`, drawn from the
   explicit graph overlay only.
2. **Fact-set selector** — pull a bounded set of N accepted/promoted facts about subject X
   from the overlay (reusing the same overlay the multi-evidence planner reads).
3. **Constrained generator** — hybrid: lexicalize each required triple via the renderer,
   then use the transition tables to join clauses; a re-roll/repair loop retries when the
   post-hoc checker fails. (Deliberately conservative first cut; template-with-connectives,
   not free recombination.)
4. **Post-hoc constraint verifier** (the heart of the new work) — scores any text (ours or
   the LLM's) against the constraint spec: fact inclusion, fidelity, and hallucination
   (§3). This verifier is **shared and applied identically to both systems** so the
   comparison is symmetric.

---

## 3. Metrics (design only — not computed here)

All objective and computable without human eval. Each row states how it maps to an existing
concept so the numbers are interpretable against prior tracks.

- **Fact inclusion rate** — of the N required facts, how many appear literally (matched by
  normalized subject/object surface + predicate lexicalization, allowing the renderer's
  known paraphrases). Analogue of object recall.
- **Fact fidelity** — of facts that appear, how many appear *correctly* (object attached to
  the right predicate/subject, not merely the object token floating somewhere). Guards
  against "mentioned but distorted." Analogue of precision-of-the-included.
- **Hallucination rate** — fraction of asserted content beyond the supplied fact set
  (extra named entities / relations not in `required_facts`). Direct analogue of the QA
  **unsupported-claim rate** from the support gate, moved post-hoc onto generated text.
  This is the single metric where the hypothesis predicts MicroWorld's structural advantage.
- **Readability / fluency proxy** — labelled explicitly as a *proxy, not human-validated
  quality*: fraction of 3-word windows in the output that are attested in the corpus
  order-2/trigram tables (the same spirit as poetry_lab's 3-word-window grammaticality
  metric). Reported as "proxy fluency," expected to favour the LLM. We will **not** claim
  it measures prose quality.

Reporting rule (carried over from prior tracks): no single composite "goodness" score;
report the constraint-adherence metrics and the proxy-fluency metric separately, so the
predicted trade-off (MicroWorld ↑ adherence, LLM ↑ fluency) is visible rather than averaged
away.

---

## 4. Fair comparison protocol with the LLM

Symmetry is the whole point; every asymmetry is a confound.

- **Same model / scale** — `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, the frozen baseline
  already used in `open_book_qa`, same checkpointed harness.
- **Same task input** — identical constraint spec. The LLM prompt is the natural-language
  form of the same spec: *"Write a short piece about X using only these facts: A, B, C. Use
  all of them. Do not add anything not listed."* MicroWorld receives the structured spec.
- **Same scorer** — the §3 post-hoc verifier runs **unchanged on both outputs**. Inclusion,
  fidelity, and hallucination are judged by the same normalization and the same fact set.
  Proxy fluency uses the same corpus tables for both.
- **Same subjects, with an independence check** — reuse `open_book_qa` subjects for the main
  set, plus a curated set of subjects asserted disjoint from any training/eval subject
  (mirroring `independent_multi_evidence_v1`), so results are not an in-sample artifact.
- **Honest refusal is a valid outcome** — if the graph lacks N clean facts about X,
  MicroWorld audits rather than fabricates; that is recorded as an audit, not a loss,
  exactly as in the compositional-grammar track.

---

## 5. Honest effort estimate (implementation, not this design)

Scope comparable to `compositional_grammar_v1`'s first iteration. Realistic estimate for a
**first runnable iteration**, assuming reuse of the renderer, transition tables, and Qwen
harness:

| Component | Est. |
|---|---|
| Constraint spec type + fact-set selector from overlay | 0.5 day |
| Constrained generator (lexicalize + connective join + repair loop) | 1.5–2 days |
| Post-hoc constraint verifier (inclusion / fidelity / hallucination) | 1.5 days — the hard part; normalization + fidelity (right-predicate attachment) is where most bugs will be |
| Proxy-fluency metric wired to order-2 tables | 0.5 day |
| Frozen case set + independence subset + LLM prompt template | 1 day |
| Experiment runner + parity/symmetry checks + report | 1 day |
| Tests (parametrized suite mirroring `test_compositional_grammar_v1.py`) | 1 day |

**Total: ~7–8 focused days** for a first iteration that produces a defensible A/B table.
Main risk, and where the estimate could grow: the verifier's fidelity check (distinguishing
"object present" from "object correctly attached to its predicate") is genuinely harder than
QA object-recall and may need iteration on normalization; and the constrained generator's
fluency may be poor enough that the proxy-fluency gap is uninteresting — in which case the
honest report is "adherence advantage confirmed, fluency gap large and expected," not a
softened composite.

---

## 6. Out of scope (v1)

Multi-subject pieces, long-form (>~5 sentences), stylistic control, free recombination
under constraint (v1 is template-with-connectives), human fluency eval, conflict resolution
among facts, and any production routing. v1 exists only to prove that **explicit
inclusion/exclusion constraints can be enforced post-hoc on generated text and compared
symmetrically against an LLM** — and to measure the predicted adherence-vs-fluency
trade-off. Await review before implementation.
