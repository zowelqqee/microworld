# Constrained-creative — MicroWorld vs Qwen A/B

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, sampler `temp=0.0` (deterministic), Qwen
chat template, 3-query warm-up. Each prompt generated twice — **0 determinism mismatches** (as
expected at temp=0). Data: [`qwen_ab_results.json`](qwen_ab_results.json); Qwen text drop-in for
the runner: [`qwen_outputs.json`](qwen_outputs.json). Driver: `qwen_ab.py` (throwaway, imports
production reasoning read-only; nothing committed, no production code changed).

**Dataset:** all 27 overlay subjects with ≥3 facts (no cherry-picking), N=3 facts each. Same
constrained prompt for both sides ("Write a short piece about X using only these facts: …. Use
all of them. Do not add anything not listed."). **Same verifier scores both.**

## Result

| Metric | MicroWorld | Qwen | Notes |
|---|---:|---:|---|
| Inclusion rate | **1.000** | **0.691** | Qwen drops ~31% of required facts |
| Fidelity rate | 0.926* | 0.482 | *MW's true value is ~1.0 — see limitation 1 |
| Hallucination-token rate | **0.000** | **0.496** | ~half of Qwen's content is unlisted |
| Proxy fluency | 0.000 | 0.000 | **uninformative here — see limitation 2** |

## Honest interpretation (narrow scope)

On the **measured constraint-adherence claim**, MicroWorld's template wins decisively on this
set: it includes every fact and asserts nothing outside the fact set, by construction. Qwen,
given the identical "use only these facts, add nothing" instruction, exhibits **both** failure
modes the constraint targets:

- **Drops specifics.** e.g. Blue Origin: Qwen wrote "headquartered in the United States,
  specifically in the state of Washington" and omitted the given "Kent, Washington" (inclusion
  0.667 on that case).
- **Adds unlisted content.** e.g. Bloomberg News: "a leading international news organization";
  Arts Education Research: "a multidisciplinary approach … Their collaborative efforts resulted
  in a comprehensive study …". These are real additions beyond the facts, not paraphrase noise
  (confirmed by inspecting the outputs), so the ~0.50 hallucination rate is meaningful.

This is exactly the trade the design hypothesized, with one important correction (limitation 2).

## Limitations — read before trusting the numbers

1. **MicroWorld fidelity 0.926 is a verifier artifact, not a real defect.** The fidelity check
   splits text into sentences to test object↔subject attachment; it mis-splits on objects
   containing periods (initials like "I.A. Narkyevich", "I.V. Pavlushkov"). All 3 sub-1.0 MW
   cases (`Mathematics`, `E Groups`, `SPSS`) are this artifact. MicroWorld's generator includes
   every fact correctly by construction — **true MW fidelity ≈ 1.0.** (The same splitter affects
   Qwen symmetrically, but Qwen's fidelity is genuinely low for other reasons.)

2. **Proxy fluency is uninformative — it does NOT show MicroWorld is as fluent as Qwen.** Both
   score 0.000 because the corpus-trigram proxy (built from the literary/wiki corpus) covers
   almost none of the graph vocabulary (proper nouns) *or* either system's phrasing. Reading the
   outputs, **Qwen's prose is plainly more natural** than MicroWorld's template ("In addition, it
   develops spacecraft."). The design predicted the LLM would win fluency; this run **cannot
   confirm or quantify that** because the metric floors out. Honest statement: MicroWorld wins
   measured constraint adherence; the fluency comparison is **unmeasured**, and qualitative
   reading favours Qwen.

3. **Verifier is surface-token based.** Inclusion/hallucination are proxies. A paraphrase
   ("builds" for "develops") could read as a missing fact; spot-checks show this did not drive
   Qwen's inclusion misses here (they were genuine omissions), but the proxy is not
   paraphrase-robust and a paraphrase-aware verifier is future work.

## Scope guard

This is 27 subjects from one overlay, N=3, one 0.5B model. The finding is: **on this constrained
fact-adherence task, MicroWorld's grounded template maintains the inclusion/exclusion constraint
and this Qwen model does not.** It is **not** evidence about prose quality, larger models, or
generation ability in general.
