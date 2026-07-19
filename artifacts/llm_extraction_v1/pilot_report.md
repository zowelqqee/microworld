# Gemini 2.5 Flash stateless extraction pilot

## Boundary and material

Gemini was used only as a stateless text-to-JSON extractor: 15 stored arXiv
sentences from the same construction families used in the grammar pilot. It
does not participate in the reasoning or answer path. No triple was promoted to
accepted, promoted, or serving memory; no production precision-gate logic was
changed. Raw responses and per-call usage are in `raw_responses.json`; the
human review is recorded in `manual_review.md`.

## Manual-review results

| Measure | Result |
|---|---:|
| Calls / source sentences | 15 / 15 |
| Extracted triples | 35 |
| Literal-support hallucinations | 0 / 35 (0.0%) |
| Generic/self-reference endpoint rejects (`we`, `this paper`, `this study`) | 8 / 35 (22.9%) |
| Accepted or accepted-after-normalization | 26 / 35 (74.3%) |
| Clean, literal, new-predicate candidates | 17 / 35 (48.6%) |
| Grammar pilot net-new accepted groups on the same construction family | 0 |

The manual review accepts normalization where wording is merely morphological
or surface variation (`has been serving` → `serves`, `allowing` → `allow`,
`providing` → `provides`, `are for` → `used_for`). It rejects paper/author
self-reference, unresolved pronouns, and the incorrect attachment “Random
Forest classification prunes transitions.”

## What worked and what did not

Gemini extracted an `is based on` relation from “a novel architecture based on
Quantum Artificial Intelligence,” a noun-phrase/ACL form for which the grammar
pilot observed 19 cues and 0 triples. It also recovered literal relations such
as SciServer builds upon/extends SkyServer and AutoSlim uses/prunes. This is a
real construction-recognition gain over the grammar pilot.

The structural quality problem remains: authorial subjects and generic objects
are common, and raw predicates need canonicalization. The existing unchanged
validator would also quarantine arbitrary new predicate labels as
`ambiguous_relation`; this pilot therefore establishes extraction potential,
not a bypass or replacement for the precision gate.

## Cost

Measured API usage: 1,466 input tokens, 2,119 visible output tokens, and
25,432 thinking tokens (29,017 total). Using Gemini 2.5 Flash standard pricing
for this token band—$0.30 / million input tokens and $2.50 / million output
tokens, with thinking treated as output—the measured pilot cost is about
**$0.069**. At the same observed mean usage, 1,000 similar sentences would be
approximately **$4.62**, before retries or any higher-tier pricing. Pricing is
external and must be rechecked before a broader run.

## Gate verdict: proceed, narrowly

The hallucination gate passes (0.0% < 10%) and clean new candidates are
materially above the grammar pilot's 0 net-new result. Proceed only to a
broader **proposal-only** pilot that adds deterministic canonicalization and
then sends candidates through the unchanged precision gate. Do not promote any
candidate or change the gate on this evidence alone.
