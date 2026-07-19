# Reflective reasoning core v2 — extension design + gate pilot

**Type: design + hard-gate pilot (same discipline as the day's prior four).**
**Verdict: MIXED (both outcomes anticipated by the task).**
- **Pattern A — co-attribution: PROCEED**, narrow and gated, as a NEW lower-confidence class
  `speculative_extended` (kept architecturally separate from the proven `speculative_verified`).
- **Pattern B — property-transfer by analogy: HONEST STOP** (structurally unsound; no filter
  rescues it — the informed-reflection outcome).

Does not touch the two proven rules (`founding-counterfactual`, `2-hop why-might`). Their result
(11/11 defensible today) is preserved and **must not be diluted** by the new, weaker mechanism —
see §5, where the confidence levels are reported separately.

Artifacts: [`pilot_extension_enumerator.py`](pilot_extension_enumerator.py),
[`pilot_extension_traces.json`](pilot_extension_traces.json),
[`pilot_extension_score.py`](pilot_extension_score.py),
[`pilot_extension_scored.json`](pilot_extension_scored.json).

---

## 0. Architectural framing (three distinct confidence levels)

This is **not** "closing the open-domain gap." It is extending the proven reflective mechanism to
a broader, explicitly-named composition class, at an explicitly *lower* confidence. Three levels
that must remain **separately labelled and separately measured** (never merged):

| Level | `support_kind` | Meaning | Status |
|---|---|---|---|
| Grounded | `grounded` | direct QA path | unchanged |
| Verified speculation | `speculative_inference` | proven rules (founding-counterfactual, 2-hop why-might) | 11/11 defensible today |
| **Extended speculation** | **`speculative_extended`** | **broader composition beyond the proven patterns** | **NEW — this pilot** |

`speculative_extended` must render and score **separately** from `speculative_inference`, so the
weaker new mechanism cannot inflate — or be credited with — the strong proven result.

---

## 1. What the proven rules constrain, and why the extensions are THESE

Reading `reflective_reasoning_v1.py`, the two proven rules are narrow by three structural locks:
- **single predicate class per rule** (counterfactual: existence-conferring only; abduction: any,
  but via a path),
- **fixed hop distance** (counterfactual: 1 focal + its object's facts; abduction: exactly 2),
- **entity-type constraint** (counterfactual object must itself be a graph entity).

Data-driven candidate selection (not "try more"):
- Deep entity-chains are **sparse**: only 15/276 objects are themselves subjects, and the
  typed-containment 2-hop chains turned out **identical to the existing why-might bridges** — so
  "3-hop" / "typed containment" would be *redundant*, not an extension. **Rejected before pilot.**
- But **32 objects have ≥2 incoming subjects** — rich SHARED-ATTRIBUTE structure. That supports
  two genuinely-new, non-redundant patterns:
  - **Pattern A — co-attribution:** `X --pred--> O` and `Y --pred--> O` ⇒ "X and Y might be
    related; both `pred` `O`." (peer-linking via a shared object — a relation the existing rules
    never build).
  - **Pattern B — property-transfer (analogy):** X and Y share O, and `Y --p--> Z` (which X
    lacks) ⇒ "X might also `p` `Z`." (analogical leap — deliberately tested to keep or reject).

---

## 2. Pilot — naive first, then filter

### Pattern A (co-attribution)

| Stage | Pairs | Notes |
|---|---:|---|
| Naive (any shared object) | 382 | dominated by spurious cliques |
| Filter v1 (same content predicate) | 363 | barely helps — the cliques use one predicate |
| **Filter v2 (KINSHIP predicates only)** | **29** | excludes 341 distribution pairs |

The decisive refinement: **distribution predicates create spurious cliques.** `published_by`
Oxford University Press links 12+ unrelated books (Trade Remedies ~ Quantum Fields ~ Arts
Education), pairing them all as "related" — a non-sequitur. The filter keeps only KINSHIP
predicates (`develops, produces, created_by, founded, developed_by, provides`), where a shared
object implies genuine similarity, and excludes `published_by, publishes, located_in, uses,
leader_of, known_for, estimated_net_worth`.

**Manual defensibility of the 29 kinship pairs: 29/29 defensible as WEAK associations.** Examples:
- SpaceX & Blue Origin (both develop rockets) — aerospace peers ✓
- SpaceX & NASA (both develop spacecraft) — peers ✓
- xAI & OpenAI (both develop LLMs) — competitors ✓
- Martin Eberhard & Marc Tarpenning (both founded Tesla) — co-founders ✓
- 10 Jerry Kaplan AI-book pairs; 6 Michael G. Raymer quantum-book pairs (same author, same field) ✓
- Tesla & Tesla Energy (both produce battery storage) — related divisions ✓

All are defensible *as a similarity/peer relation* — but this is **genuinely weaker** than a
verified why-might: "both develop rockets" asserts kinship, not a derived causal fact. Hence
`speculative_extended`, not `speculative_inference`.

### Pattern B (property-transfer)

| Stage | Count | Defensible (manual sample of 15) |
|---|---:|---:|
| Naive | 886 | ~0 |

Every sampled transfer is a non-sequitur: "Martin Eberhard might have **founded SpaceX** because
he shares Tesla with Musk"; "Eberhard might have **net worth US$1.1 trillion**." Sharing one
attribute (co-founding Tesla) licenses transferring **nothing** else. The few that are
coincidentally true ("Musk might be known_for Tesla") are true for other reasons, not by the
inference — coincidental truth is not a defensible inference.

**No structural filter can rescue Pattern B.** The unsoundness is in the inference *form*, not in
candidate selection: to know which attributes transfer you would need causal/type knowledge the
graph does not encode — exactly the wall informed-reflection hit. **HONEST STOP.**

---

## 3. `speculative_extended` framing (structure-driven, like `uncertainty_note`)

The renderer must make the lower confidence explicit *from plan structure*, not as a vague
"be careful." Proposed phrasing for a co-attribution step:

> "**X** and **Y** are not directly linked in the verified relations; they are connected here only
> because both **⟨predicate⟩ ⟨shared object⟩**. Treat this as a broader, less-tested inference
> than the system's core reasoning — a similarity, not a stored or directly-derived fact."

This mirrors how `uncertainty_note` blocks already render structure-derived caution ("the evidence
diverges … neither reading is treated as settled"). The caution is generated from the block's
kind + the shared-object premise, never guessed from text.

---

## 4. Verdict

- **Pattern A — PROCEED**, exactly as narrow and gated as the prior two rules: ship the KINSHIP
  filter (never the naive rule), label output `speculative_extended`, render with the §3 caution.
  It adds a real capability (peer/kinship association) at an honestly-lower confidence.
- **Pattern B — STOP.** Documented negative result; analogical property-transfer is unsound over
  this graph and no filter fixes the form. This is the mixed outcome the task allowed: one new
  pattern in, one out.

---

## 5. Metrics — confidence levels reported SEPARATELY (never merged)

| Level | Cases | Defensible | Confidence |
|---|---:|---:|---|
| `speculative_inference` (proven, today — UNCHANGED) | 11 admitted | **11 / 11 (100%)** | verified |
| `speculative_extended` (Pattern A — NEW) | 29 kinship pairs | **29 / 29 defensible-as-weak-association** | **extended (lower)** |
| Pattern B (rejected) | — | ~0 / 15 sampled | n/a — stopped |

**These rows must not be summed.** The proven 11/11 is a strong, tight result about *derived*
speculation; the 29/29 is a weaker result about *similarity* speculation and is only meaningful
under the explicit `speculative_extended` caution. Merging them into one "40/40" number would
misrepresent both — the whole point of keeping the levels architecturally distinct.

### Scope guard
29 kinship pairs over one 276-edge overlay; "defensible as a weak association" is a manual,
auditable judgment (`pilot_extension_scored.json`), deliberately a *lower* bar than the
verified-speculation defensibility bar. Do not read Pattern A's 29/29 as equivalent in strength
to the proven 11/11 — that separation is the design.

---

## 6. Recommended next step

Build Pattern A as a **separate** `speculative_extended` path (new file, not touching
`reflective_reasoning_v1.py`), with the §3 framing and its own metric bucket. Leave Pattern B
documented as a negative result. Await review of the confidence-level architecture (§0) before
any integration into `answer_session`.
