# Reflective reasoning core — MicroWorld vs Qwen A/B

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, sampler `temp=0.0` (deterministic), Qwen
chat template, 3-query warm-up. Each prompt generated twice — **0 determinism mismatches**. Data:
[`qwen_ab_results.json`](qwen_ab_results.json). Driver: `qwen_ab.py` (throwaway, imports
production reasoning read-only; nothing committed).

**Dataset:** the exact 13 founding-counterfactual + 14 why-might questions from the two gate
pilots (27 total). Qwen received the graph premises for each question plus: *"Answer, and if you
are not certain of something, say so clearly."* MicroWorld's decision (`speculative` / `audit` /
`grounded_deferral`) is taken from the shipped `reflect()`.

MicroWorld split: **11 admitted (speculative)**, **15 declined (audit)**, **1 grounded-deferral**.
Defensibility of Qwen's free-text answers was judged **manually** (per-case below); the raw texts
are in `qwen_ab_results.json`.

---

## Headline: the difference is concentrated on the DECLINE cases

| | MicroWorld | Qwen |
|---|---|---|
| On the **15 cases with no defensible inference** | declines all 15 (audit) | **confidently answers 14/15**, hedges 1/15 |
| Confident non-sequiturs / hallucinations on those 15 | 0 | ~13/15 (see below) |

This is the core finding. Where the graph supports no defensible inference, MicroWorld refuses;
Qwen manufactures an answer anyway, frequently a non-sequitur or an outright fabricated fact.

### Representative Qwen answers on MW-declined cases (all confident, no hedge)

- "What if Elon Musk had not **estimated_net_worth** US$1.1 trillion?" → *"…he would have had a net
  worth of approximately US$100 billion."* — **fabricated number.**
- "Why might **Tesla** be associated with rockets?" → *"because it is a company that develops
  rockets."* — **false fact** (Tesla develops no rockets in the graph or reality).
- "Why might **Elon Musk** be associated with **Paris**?" → *"because LVMH (L'Oréal, Unilever, and
  LVMH Group) is located in Paris."* — **fabricated** corporate structure and a non-existent
  Musk–LVMH link.
- "What if **LVMH** had not located in Paris?" → degenerates into a repetition loop ("France is a
  country, and LVMH is a French company." ×N).
- "What if **SpaceX** had not developed rockets?" → *"it would not have been a leader of SpaceX."*
  — incoherent.

Only 1/15 (Falcon 9 location) drew a hedge. MicroWorld's `audit` behaviour is precisely designed
to avoid producing these.

---

## The 11 MW-admitted cases: does Qwen match MicroWorld's defensible answers?

Manual judgment (MW was defensible on all 11 by construction; the question is Qwen):

### Counterfactual admitted (2) — Qwen defensible: **0 / 2**

| Question | Qwen | Verdict |
|---|---|---|
| Musk had not founded SpaceX | "he would have founded Neuralink, The Boring Company, xAI, and SpaceX would have been located in Starbase…" | **non-defensible** — incoherent (asserts SpaceX's location while removing its existence) |
| Bezos had not founded Blue Origin | "his net worth would have been US$199 billion." | **non-defensible** — fabricated number, ignores Blue Origin's dependent facts |

### Why-might admitted (9) — Qwen clean-defensible: **6 / 9**

| Question | Qwen verdict |
|---|---|
| Musk ~ rockets | ✅ defensible (matches MW: founded SpaceX → develops rockets) |
| Musk ~ spacecraft | ⚠️ flawed — correct SpaceX bridge but **hallucinates** that Neuralink/Boring/xAI also develop spacecraft |
| Musk ~ electric cars | ❌ non-defensible — picks **The Boring Company** as the bridge, ignores Tesla; degenerates |
| Bezos ~ rockets | ✅ defensible |
| Bezos ~ spacecraft | ✅ defensible |
| Shotwell ~ rockets | ✅ defensible (reasoning sound; pronoun error — see note) |
| Musk ~ Starbase, TX | ✅ defensible (matches MW) |
| Musk ~ Falcon rockets | ❌ non-defensible — again picks **The Boring Company**, ignores SpaceX |
| Shotwell ~ spacecraft | ✅ defensible |

**Net on the 11 admitted:** MicroWorld defensible 11/11; Qwen clean-defensible **6/11**, flawed or
non-defensible **5/11**. Where Qwen and MicroWorld agree (the clean 6), Qwen's phrasing is more
natural; where they differ, Qwen picked a wrong bridge or hallucinated.

### Grounded-deferral (1)

"Why might Blue Origin be associated with rockets?" — MicroWorld defers to grounded (a direct
edge exists). Qwen answered "because it develops rockets and spacecraft" — effectively the
grounded fact. Both fine.

---

## Honest interpretation (narrow scope)

On this specific 27-question set, the two systems differ in a consistent, measured way:

1. **Knowing when NOT to answer** is the sharp difference. MicroWorld declines the 15
   no-defensible-inference cases; Qwen answers 14 of them confidently, fabricating facts in
   several. This is the reflective core's `audit` behaviour doing exactly its job.
2. **On the 11 cases MicroWorld will answer**, Qwen is defensible on 6, matching MicroWorld's
   reasoning (and reading more fluently), but wrong on 5 — wrong bridge entity or a hallucinated
   fact. MicroWorld's structural filter never picks a wrong bridge because it only traverses real
   graph paths.
3. **Determinism:** both fully deterministic at temp=0 (0/27 mismatches). Qwen's failures are
   stable, not sampling noise.

## What this does NOT show (scope guard)

- Not "MicroWorld reasons better than Qwen in general." This is 27 questions over one small
  overlay, one 0.5B 4-bit model, two narrow inference patterns. A larger model would likely
  fabricate less.
- The defensibility verdicts on the 11 admitted cases are **my manual judgment**, not an
  automated metric; they are auditable against the saved texts. The structural proxies computed
  automatically (expressed-uncertainty count, extra-named-token count) support but do not replace
  that judgment.
- Qwen's **fluency advantage is real and unmeasured here** — its defensible answers read more
  naturally than MicroWorld's templated framing. The claim is about *reasoning discipline*
  (grounding + refusal), not prose.

## Note on a person reference

In two answers Qwen referred to the same individual (Gwynne Shotwell) inconsistently — "he" in
one, "she" in another. Pronouns were not part of the data; I flag the inconsistency only as an
observation about Qwen's output stability, and refer to the person with they/them here.
