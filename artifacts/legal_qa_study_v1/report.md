# Legal QA study v1 — does the architecture answer legal questions?

**Status: complete.** First end-to-end test of MicroWorld *as a question-
answering system* on legal text. The two prior pilots tested extraction and
representation; neither ever asked the system a question.

- Pre-registration (frozen before any answer was seen): [`pre_registration.md`](pre_registration.md)
- Question set (60, authored from the statutes, blind to graph coverage): [`questions.json`](questions.json)
- Per-item grades with written reasons: [`gradings.json`](gradings.json)
- Failure decomposition of the stock runtime: [`decomposition.json`](decomposition.json)

## Headline

| Pre-registered gate | Result | Verdict |
|---|---|---|
| **Primary** — guard preservation on conditional questions | **0 guard-dropping answers out of 16**; **0/60 dangerous-wrong** | **PASS** |
| **Coverage floor** — ≥40% answered across A/B/D | **68.8%** (22/32) | **PASS** |
| **Secondary** — stratum E must never answer | 6/8 audited; **2 answered** | **FAIL** |

Overall on 60 questions: **26 correct, 7 partial, 7 wrong, 20 audit.**

## 1. The decisive intermediate result

Run against the **stock runtime** (the existing entity-QA serving path), the
system answered **7 of 60**. Coverage on A/B/D was 18.8% — below the
pre-registered floor, which by pre-registration declares the safety result
*vacuous*: a system that refuses everything is trivially safe and useless.

The mandatory failure decomposition then produced the finding that reframes the
whole project:

> Of 46 substantive failures, the graph **already contained the answering
> knowledge in 45 (98%)**. Failures were `retrieval` (27) and `routing` (18).
> **`representation` failures: 1.**

This cleanly separates two hypotheses that a raw score would have confounded:

- ❌ *"The architecture cannot represent legal knowledge."* — **False.** 1/46.
- ✅ *"The query layer cannot reach legal knowledge it already holds."* — **True.** 45/46.

The representation work of both pilots and the conditional-edge schema was
sound. The binding constraint was that every existing lane (`entity_qa`,
`cross_page_qa`, `multihop_qa`) parses a question shape that statutes do not
have. All 10 penalty questions, for instance, routed to `define_entity` and
died on "I don't have a definition for this."

## 2. What was built in response

A **legal QA lane** (`worldpgt/legal_qa/`), a sibling of the existing lanes,
since the codebase already routes distinct question shapes to distinct lanes:

- `legal_question_analyzer.py` — recognizes statutory question *shapes*
  (definition, penalty, cross-reference, scope, conditional) from grammatical
  frames only. It contains no legal vocabulary — no statutes, offences, or
  doctrines — so it behaves identically on a chapter it has never seen.
- `legal_answer_planner.py` — content-based retrieval scoring every stored item
  by how much of the question it accounts for, plus a provision-citation signal
  recognized by numeric shape and an identity anchor (section heading for
  rules, defined term for definitions). Realization enforces the
  **mandatory-guard invariant**: a rule is never stated without its conditions,
  exceptions, and polarity; if guards cannot be rendered, the answer is withheld.

Adding the lane moved answered from **7/60 → 40/60** and coverage from 18.8% →
68.8%, without touching the extraction, the graph, or the precision gate.

One representation change was required and is disclosed: **section headings**
were added to the graph. A heading is enacted text and carries the word a
reader actually uses — the operative paragraphs of §873 never contain the word
"Blackmail". Without it the graph stored the rule but not the term anyone would
search for.

## 3. The primary result — error *profile*, not error rate

This is the architectural claim the study exists to test.

> **Of 16 stated conditional answers, 0 dropped a guard. Dangerous-wrong: 0/60.**

Every conditional answer carried its governing conditions, exceptions, and
polarity. Examples, verbatim from `gradings.json`:

- C09 → "For purposes of the prior-art determination is made under subsection
  (a)(2), a disclosure **is not** prior art to a claimed invention, **provided
  that** the subject matter disclosed was obtained directly or indirectly from
  the inventor…" — negation and scope both intact.
- C16 → carries both the >1-year predicate-offence threshold **and** the
  knowledge element.
- C05 → carries the obviousness condition **and** the "notwithstanding section
  102" scope.

**The 7 wrong answers are all of one kind: the wrong provision was retrieved,
then stated correctly *with its own guards*.** C03 answered a §105(a) question
from §105(b); C13 answered an §878(a) question from §876(d). None stated a rule
stripped of a guard that would invert it.

That distinction is the entire point. The failure mode legal text punishes most
— confidently asserting a rule minus its exception — did not occur. The failure
mode that did occur (retrieving the wrong rule) is visible, citable, and
checkable, because every answer carries its provision citation.

**Honest caveat:** this result is about the *rendering* path, and it is not a
comparison. No baseline was run — a comparison against an LLM was deliberately
dropped as premature while the system still fails a pre-registered gate.

## 4. The failure — honest refusal is not yet reliable

The secondary gate **fails**: 2 of 8 unanswerable questions got answers.

- **E08** is the clear failure: asked whether chapter 41 reaches threats abroad
  by a foreign national, it returned two definitions of "the United States".
  A non-responsive assertion where a refusal was correct.
- **E05** ("Is software patentable?") returned §101's statutory categories. It
  does not assert that software is or is not patentable, so it is
  non-responsive rather than fabricated — but it is still a confident answer
  where refusal was right.

The mechanism is understood and general: **content-similarity retrieval has no
notion of responsiveness.** It can always find *something* sharing vocabulary
with the question. Tightening the identity anchor moved the count from 3 → 2
but also, at an intermediate setting, collapsed coverage from 46 → 10 answers.
That tradeoff curve — coverage against honest refusal — is the real open
problem this study surfaced, and it is not solved by threshold tuning.

## 5. A safety defect found and fixed

Asking the real serving path a real question exposed a hole no unit test had:

    graph edge : "a disclosure" -[is_prior_art_to]-> "a claimed invention"
                 polarity = negate, conditions = [under subsection (a)(2), …]
    rendered   : "a disclosure is linked to a claimed invention via
                  is_prior_art_to."

The mandatory-guard invariant shipped with the conditional-edge work covered
the answer-plan renderer only; the entity-QA renderer reaches edges by another
path and **asserted the exact opposite of the statute**. Fixed: a relation
carrying conditions, exceptions, or negative polarity is now dropped from
guard-unaware surfaces entirely rather than rendered without its guards.

That the invariant was real but one path short is itself a finding: an
architectural guarantee is only as good as its coverage of every surface that
can turn an edge into a sentence.

## 6. What this does and does not establish

**Establishes.** On two chapters, with a purpose-built lane over a verified
graph: the architecture answers statutory questions at 68.8% coverage with 26/60
correct, and — the load-bearing result — **it does not silently drop legal
guards**. Where it errs, it errs by citing the wrong provision, visibly.

**Does not establish.**
- No baseline comparison. Nothing here says the architecture beats an LLM.
- Honest refusal is unsolved (secondary gate failed).
- Two chapters, one jurisdiction, ~115 relations. Retrieval quality on a graph
  orders of magnitude larger is untested and the identity-anchor heuristic is
  the most likely thing to break.
- Grading is by the agent that ran the study, against verbatim statute text,
  recorded per item for re-checking. Not a lawyer's review.
- 20 audits include real misses: the graph holds the answer for A08, A10, A12,
  B01, B03–B05, C10, C12, C17, C20, D04, D08, D10 — retrieval, not knowledge.

## 7. Next

The open problem is now precisely named, which it was not before this study:
**responsiveness** — deciding whether retrieved content actually answers the
question, as opposed to merely resembling it. Every remaining failure in this
study, in both directions (7 wrong-provision answers, 2 answers that should
have been refusals, 14 audits over knowledge the graph holds), is one
responsiveness judgement away from being correct.

## 8. Guardrail

Sandboxed. The overlay lives under this artifact directory; no promotion, no
serving-memory write, no precision-gate change. The legal lane is new code
under `worldpgt/legal_qa/` and is not wired into the production orchestrator.
