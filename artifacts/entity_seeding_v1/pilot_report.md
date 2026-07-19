# Entity-seeding lane v1 — pilot gate report

Status: **FAIL — stop before formal build**
Date: 2026-07-19
Input: the three stored arXiv LLM runs used by the validated node-quality
pilot.

## Gate decision

**FAIL.** The mechanical criterion found 15 repeated, literal, proposal-local
surfaces. Manual review found only 3 clean distinct new entities and 12 false
positives/noise candidates. Therefore the strict criterion does not achieve
zero false positives, and the formal build (Stage 3) is not authorized by this
gate.

No entity-seeding module, pipeline integration, overlay write, or regression
run was performed after this FAIL. The existing node-quality filter and
precision validator were not modified.

## Corpus and counting method

The inputs are the exact raw response files already used by
`node_quality_filter_v1`:

| Extractor run | Raw triples | Source rows |
|---|---:|---:|
| OpenAI `gpt-4o-mini` | 33 | 15 |
| Gemini 2.5 Flash | 36 | 15 |
| Gemini 3.1 Flash-Lite | 32 | 15 |
| **Total** | **101** | **45 model/source rows** |

Each triple contributes its subject and object endpoint: 202 endpoint mentions
in total. After whitespace normalization and exact case-insensitive grouping,
there are 82 unique endpoint surfaces. Forty-two surfaces occur in at least
two distinct extractor runs. Existing production-index resolution remains
unchanged: the baseline node-quality pilot records 101/101
`unresolvable_entity` outcomes.

For this pilot only, each supporting relation was checked with an ephemeral
in-memory index containing the exact literal endpoint surfaces in that
relation. This removes only the expected current-index failure so the
unchanged `assess_node_quality` rules still check authorial, generic,
event-like, and list-derived structure. It does not write an overlay or alter
serving resolution. Literal support means the extracted endpoint surface is a
case-insensitive contiguous substring of the stored source sentence.

The independence count is deliberately cross-run: two distinct extractor
runs are required; repeated rows from one model do not count twice. No alias,
article stripping, fuzzy merge, title fallback, or external corroboration was
used.

## Mechanical candidates and manual verification

These are all 15 surfaces satisfying the mechanical part of the design:
minimum two extractor runs, at least two supporting relation records, literal
span support, and an accepted decision from the unchanged node-quality rules
under the proposal-local index.

| Exact candidate surface | Mentions | Runs | Good records | Supporting source rows | Manual verdict | Reason |
|---|---:|---:|---:|---|---|---|
| `AutoSlim` | 8 | 3 | 8 | pilot-09 | **ACCEPT** | Named system; distinct and literal. |
| `debiased estimators` | 7 | 3 | 5 | pilot-12 | **REJECT** | Generic methodological class, not a distinct named entity. |
| `Existing AML threat evaluation approaches` | 5 | 3 | 5 | pilot-07 | **REJECT** | Generic family of approaches. |
| `low-impact transitions` | 5 | 3 | 5 | pilot-09 | **REJECT** | Descriptive graph/artifact phrase. |
| `Random Forest classification` | 5 | 3 | 5 | pilot-09 | **REJECT** | Literal method description in this sentence, not a distinct named node surface. |
| `SciServer` | 5 | 3 | 3 | pilot-03 | **ACCEPT** | Named system; distinct and literal. |
| `REGAI` | 4 | 3 | 4 | pilot-04 | **ACCEPT** | Named system; distinct and literal. |
| `asymptotically Gaussian` | 3 | 3 | 3 | pilot-12 | **REJECT** | Mathematical property, not an entity. |
| `technical attack robustness` | 3 | 3 | 3 | pilot-07 | **REJECT** | Generic property/concept. |
| `the SkyServer system of server-side tools` | 5 | 2 | 5 | pilot-03 | **REJECT** | Underlying SkyServer is legitimate, but this exact candidate is a verbose referring phrase; accepting it requires alias/descriptor normalization outside v1. |
| `automata graph density` | 2 | 2 | 2 | pilot-09 | **REJECT** | Generic graph property. |
| `CT of such clouds` | 2 | 2 | 2 | pilot-10 | **REJECT** | Contextual/descriptive fragment, not a named entity. |
| `hypothesis testing for linear combinations of precision matrix entries across populations` | 2 | 2 | 2 | pilot-12 | **REJECT** | Generic activity description. |
| `quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles` | 2 | 2 | 2 | pilot-13 | **REJECT** | Long descriptive phrase/list-context residue. |
| `the performance of both classical LLMs and RAG-based LLM techniques` | 2 | 2 | 2 | pilot-04 | **REJECT** | Descriptive performance object, not an entity. |

The three accepted rows are all legitimate named systems, but 12/15
mechanically qualifying rows are noise. In particular, repeated model output
does not prevent generic noun phrases and properties from passing the current
node-quality rules when a proposal-local index makes them resolvable.

## Full repeated-surface inventory

For completeness, the following is the complete inventory of all 42 exact
surfaces occurring in at least two extractor runs. `good` counts literal
supporting records accepted by the unchanged node-quality rules under the
proposal-local index. A surface is mechanically qualifying only when
`good_runs >= 2`.

| Surface | Mentions | Runs | Good records | Good runs |
|---|---:|---:|---:|---:|
| `we` | 12 | 3 | 0 | 0 |
| `AutoSlim` | 8 | 3 | 8 | 3 |
| `debiased estimators` | 7 | 3 | 5 | 3 |
| `NB2Slides` | 7 | 3 | 1 | 1 |
| `Existing AML threat evaluation approaches` | 5 | 3 | 5 | 3 |
| `low-impact transitions` | 5 | 3 | 5 | 3 |
| `Random Forest classification` | 5 | 3 | 5 | 3 |
| `SciServer` | 5 | 3 | 3 | 2 |
| `a novel tutor model for adaptive training` | 4 | 3 | 1 | 1 |
| `REGAI` | 4 | 3 | 4 | 3 |
| `this paper` | 4 | 3 | 0 | 0 |
| `adversaries to achieve either remote code execution or information disclosure` | 3 | 3 | 0 | 0 |
| `asymptotically Gaussian` | 3 | 3 | 3 | 3 |
| `Contemporary fuzz testing techniques` | 3 | 3 | 0 | 0 |
| `fundamental challenges in autonomous vehicles navigation` | 3 | 3 | 0 | 0 |
| `GAs` | 3 | 3 | 0 | 0 |
| `identifying memory corruption vulnerabilities` | 3 | 3 | 0 | 0 |
| `memory corruption vulnerabilities` | 3 | 3 | 0 | 0 |
| `models` | 3 | 3 | 0 | 0 |
| `multimodal sensor fusion` | 3 | 3 | 0 | 0 |
| `Nav-Q` | 3 | 3 | 0 | 0 |
| `post-quantum cryptographic protocols` | 3 | 3 | 0 | 0 |
| `probabilistic reasoning` | 3 | 3 | 0 | 0 |
| `Quantum Neural Networks` | 3 | 3 | 0 | 0 |
| `Quantum reinforcement learning for navigation policy optimization` | 3 | 3 | 0 | 0 |
| `secure communication` | 3 | 3 | 0 | 0 |
| `technical attack robustness` | 3 | 3 | 3 | 3 |
| `the crucial need for fairness and explainability in AI applications within healthcare` | 3 | 3 | 0 | 0 |
| `This study` | 3 | 3 | 0 | 0 |
| `users to compose presentations of their data science work` | 3 | 3 | 0 | 0 |
| `the SkyServer system of server-side tools` | 5 | 2 | 5 | 2 |
| `the proposed framework` | 4 | 2 | 0 | 0 |
| `a learning-based model (ProbCT)` | 2 | 2 | 0 | 0 |
| `automata graph density` | 2 | 2 | 2 | 2 |
| `both methods` | 2 | 2 | 0 | 0 |
| `CT of such clouds` | 2 | 2 | 2 | 2 |
| `high-fidelity wave packets` | 2 | 2 | 1 | 1 |
| `hypothesis testing for linear combinations of precision matrix entries across populations` | 2 | 2 | 2 | 2 |
| `quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles` | 2 | 2 | 2 | 2 |
| `quantum performance and future proof security` | 2 | 2 | 0 | 0 |
| `statistical uncertainties` | 2 | 2 | 0 | 0 |
| `the performance of both classical LLMs and RAG-based LLM techniques` | 2 | 2 | 2 | 2 |

The remaining 40 unique surfaces occur in fewer than two extractor runs and
therefore fail the first mechanical criterion. The source rows and every
triple remain available without transformation in:

- `artifacts/llm_extraction_v1/openai/pilot_v1/raw_responses.json`
- `artifacts/llm_extraction_v1/runs/pilot_5s_v2/raw_responses.json`
- `artifacts/llm_extraction_v1/runs/pilot_3_1_flash_lite_v1/raw_responses.json`
- `artifacts/llm_extraction_v1/node_quality_filter_v1/results.json`

## Interpretation and stop condition

This is a useful negative result. Cross-model repetition recovers the intended
new systems (`SciServer`, `AutoSlim`, and `REGAI`), but the same signal also
repeats generic concepts, method descriptions, properties, and verbose
fragments. Literal support is necessary but insufficient, and the existing
node-quality filter was not designed to distinguish every generic concept
from a newly named graph node once resolution is provisionally available.

The lane therefore stops at Stage 2. No Stage 3 formal module or integration
should be built from this corpus. Nothing was promoted to serving memory,
nothing was written to accepted/promoted overlays, and nothing was committed.
