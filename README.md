# Microworld

Microworld is an experimental explicit-memory, graph, trust, policy, and audit
based reasoning and QA system. It explores whether useful controlled QA,
memory, reasoning, and knowledge ingestion can be built without neural weights,
backpropagation, fine-tuning, GPT-style next-token rendering, embeddings, GPU,
or network calls.

The project is intentionally research-oriented. It does not claim that symbolic
graphs are generally superior to neural networks. It explores a complementary
path: compact explicit memory and trust learning for graph reasoning, where
behavior can be audited, compressed, transferred, and corrected without updating
neural weights.

LLMs learn to speak and world understanding emerges as a side effect.
Microworld tries to build explicit world memory first, then use language as an
interface to that world.

Current test status:

```text
python3 -m pytest -q  ->  2030 passed
```

## Navigation

* [Why This Exists](#why-this-exists)
* [Current Status](#current-status)
* [Current worldpgt research status](#current-worldpgt-research-status)
* [Current Architecture](#current-architecture)
* [Latest Experimental Results](#latest-experimental-results)
  * [Current Controlled QA Results](#current-controlled-qa-results)
  * [Human Audit Baseline](#human-audit-baseline)
  * [Audit-Driven Trust Learning](#audit-driven-trust-learning)
  * [Feedback Compression](#feedback-compression)
  * [Suppression Audit](#suppression-audit)
  * [Quality-Aware Suppression](#quality-aware-suppression)
  * [Microworld-style Name/Surname Generation](#microworld-style-namesurname-generation)
  * [Latest Name/Surname Generation Results](#latest-namesurname-generation-results)
  * [Makemore vs Microworld Benchmark](#makemore-vs-microworld-benchmark)
  * [Efficiency Benchmark](#efficiency-benchmark)
  * [RAM / RSS Benchmark](#ram--rss-benchmark)
  * [Careful Research Claim](#careful-research-claim)
  * [worldpgt QA Layer](#worldpgt-qa-layer)
* [What Was Learned](#what-was-learned)
* [Running](#running)
* [Documentation](#documentation)
* [Limitations](#limitations)
* [Next Steps](#next-steps)
* [Status](#status)

## Current Status

Microworld currently demonstrates a lightweight, deterministic, auditable QA
architecture over explicit memory and isolated knowledge overlays. It is strong
on controlled benchmark domains, explicit memory, source-aware facts, safe
abstention/audit, and low runtime cost. It is limited by narrow scope, curated
inputs, rule/curriculum-based analyzers, and surface renderer quality.

Microworld currently demonstrates a narrow but useful property: on controlled
explicit-memory QA benchmarks, it can answer when the supporting path is present
and audit when the answer would require unsupported inference, weak-link
promotion, current/live data, or volatile facts.

Across the current controlled QA benchmark layers, worldpgt handles 350 prompts
with 0 wrong decisions: 219 answered and 131 safely audited. Answer precision is
1.0 on each benchmark family. These results are scoped to supported controlled
domains and isolated overlays; they are not open-domain or live-fact claims.

The current Python implementation runs these small controlled benchmark batches
in about 0.06-0.14 seconds with roughly 24 MB peak RSS. These are single-run
local measurements and should be treated as order-of-magnitude efficiency
indicators, not final benchmark claims.

## Current worldpgt research status

| Layer                     | Status      |
| ------------------------- | ----------- |
| Main QA                   | 48/48       |
| Generalization QA         | 24/24       |
| Entity QA v1              | 28/28       |
| Entity QA expansion       | 111/111     |
| Adversarial Entity QA     | 68/68       |
| Cross-page Entity QA      | 71/71       |
| Self-ingestion dry-run QA | all green   |
| Promote Overlay Delta v1  | 310-item promoted overlay, regressions green |
| Full suite                | 2030 passed |

Promote Overlay Delta v1 — 310-item promoted overlay, regressions green.

The current 50-page wiki overlay is an isolated memory artifact, not trusted
accepted memory. It contains 283 overlay items: 50 entities, 50 definitions, 53
relations, 126 weak contextual links, and 4 source-qualified volatile facts.
The overlay remains `safe_for_general_runtime=false`.

Wikipedia Self-Ingestion v1 is a safe offline dry-run pipeline. It reads local
Wikipedia-like documents, converts them to wiki-like pages, reuses unchanged
wiki ingestion and overlay builders, classifies deltas/conflicts/quarantine,
proposes an overlay delta, and runs deterministic regression gates. It does not
write raw text directly into accepted memory, and the dry-run overlay is
separate from the accepted wiki overlay.

Promote Overlay Delta v1 validates the self-ingestion delta and promotes only
safe items into a separate promoted overlay artifact. The base accepted wiki
overlay is preserved unchanged.

Promotion artifacts:

```text
worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.json
worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.meta.json
worldpgt/experiments/self_ingestion_v1/promotion/promotion_report.json
worldpgt/experiments/self_ingestion_v1/promotion/promotion_validation.json
worldpgt/experiments/self_ingestion_v1/promotion/promotion_regression_summary.json
```

Important overlay distinction:

* `accepted_wiki_memory_overlay_v1.json` - 283 items, not touched.
* `promoted_wiki_memory_overlay_v1.json` - 310 items, separate promoted overlay.
* `safe_for_general_runtime=false`.

## Why This Exists

Modern AI systems are powerful, but much of their knowledge and behavior is
stored implicitly inside model weights or long context histories. That makes it
hard to inspect why a behavior changed, compress feedback into durable memory,
or debug errors at the level of relations, nodes, and policies.

Microworld asks a narrower question:

```text
Can some memory, reasoning, and learning behavior be achieved more efficiently
and more transparently than by simply growing neural models wider, expanding
context, or relying on backpropagation?
```

The current answer is partially yes. For graph-based symbolic reasoning tasks,
Microworld shows that audit feedback can be compressed into a tiny explicit
trust state, that the state can transfer to unseen data, and that behavior can
change without retraining a neural model.

## Current Architecture

Knowledge is represented as explicit graph relations:

```text
source --relation_type--> target
```

The current system includes:

* graph memory
* ConceptNet import
* pattern discovery
* transitive reasoning
* mixed-pattern reasoning
* structural similarity
* concept discovery
* relation proposal
* hub penalty
* relation trust
* node quality
* relation drift
* relation blacklist
* audit pipeline
* audit-driven trust learning
* trust transfer experiment
* feedback compression benchmark
* suppression audit
* quality-aware suppression policy
* controlled continuation benchmark
* surface repair layer
* prompt-tail compatibility gate
* audit reason mining
* coverage mode
* coverage gap curriculum
* knowledge ingestion pipeline
* accepted memory provider
* answer planner QA
* generalized question analyzer
* Wikipedia/Wikidata-style ingestion v2
* wiki candidate memory overlay v1
* entity QA benchmark v1
* name/surname generation
* audit-driven bad-pattern mining
* makemore-vs-Microworld efficiency benchmark
* RAM/RSS benchmark
* full pipeline demo

The main reasoning path is:

```text
graph memory
-> pattern discovery
-> prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final decision
```

The key design choice is separation of concerns:

* trust memory estimates whether relation/rule families have been reliable
* decision policy decides what to suppress or keep
* normalization should repair source and target spelling/canonicalization issues

The current entity-knowledge QA path is intentionally isolated:

```text
curated source pages
-> deterministic ingestion candidates
-> isolated wiki candidate memory overlay
-> overlay provider
-> entity question analyzer
-> entity answer planner
-> entity answer renderer
-> validator / audit
-> controlled benchmark
```

## Latest Experimental Results

### Current Controlled QA Results

| Benchmark | Items | Correct | Wrong | Answered | Audited | Precision | Time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main accepted-memory QA | 48 | 48 | 0 | 42 | 6 | 1.0 | ~0.070 s | ~24.73 MB |
| Generalization QA | 24 | 24 | 0 | 19 | 5 | 1.0 | ~0.140 s | ~24.61 MB |
| Entity QA over wiki overlay | 28 | 28 | 0 | 23 | 5 | 1.0 | ~0.060 s | ~23.61 MB |
| Entity QA expansion | 111 | 111 | 0 | 79 | 32 | 1.0 | - | - |
| Adversarial Entity QA | 68 | 68 | 0 | 6 | 62 | 1.0 | - | - |
| Cross-page Entity QA | 71 | 71 | 0 | 50 | 21 | 1.0 | ~0.160 s | ~24 MB |
| Wiki ingestion v2 | 283 candidates | review errors: 0 | - | - | - | - | ~0.060 s | ~23.8 MB |
| Wiki overlay v1 | 283 overlay items | skipped: 0 | - | - | - | - | ~0.050 s | ~23.7 MB |
| Self-ingestion v1 dry run | 310-item dry-run overlay | dry-run regressions green | - | - | - | - | ~0.210 s | ~25.8 MB |
| Promote Overlay Delta v1 | 310 promoted overlay items | 27/27 delta accepted | rejected 0 | blocked 0 | regressions green | full suite 2030 passed | - | - |

Accepted-memory QA uses
`worldpgt/experiments/accepted_knowledge_memory_v1.json`: 221 accepted items
(163 fact items, 58 pattern items), 6 ambiguous terms, 12 senses, and 104
positive cues. Supported intents are `define_sense`, `classify_context`,
`explain_cue`, `distinguish_senses`, and `unknown_or_ambiguous`.

Wikipedia/Wikidata-style ingestion v2 is deterministic, offline, local-fixture
only, and candidate-generation only. It does not modify accepted memory or
runtime memory. Wiki candidate overlay v1 remains isolated from general runtime
memory; `safe_for_general_runtime` is false and
`safe_for_entity_qa_overlay` is true. Weak contextual links are never promoted
to stable facts, and source-qualified volatile facts remain source-qualified
with `requires_recheck=true`.

### Human Audit Baseline

Manual audit of ConceptNet-derived graph predictions:

```text
reviewed: 104
useful overall: 78.8%
made_of useful: 86.2%
part_of useful: 76.7%
is_a useful: 75.6%
mixed reasoning useful: 76.7%
```

`useful` means the prediction was labeled correct or plausible. These are small
exploratory audits, not formal benchmark claims.

### Audit-Driven Trust Learning

Microworld can turn manual audit feedback into a compact trust profile. That
profile changes behavior on unseen TEST data without backpropagation.

Trust transfer experiment:

```text
baseline accepted: 195
learned accepted: 99
suppressed: 96
```

Interpretation:

```text
feedback -> trust profile -> changed behavior on unseen split
```

This confirms that Microworld can learn behavioral preferences from audit
feedback without updating neural weights.

### Feedback Compression

Feedback compression benchmark with 10,000 audit rows:

```text
raw audit history: ~500,360 tokens
trust state: ~313 tokens
compression: ~1598.6x
```

This is one of the strongest current results. It shows an alternative to simply
expanding context, memory logs, or model size: large feedback histories can be
compressed into a tiny, explicit, inspectable state.

### Suppression Audit

The first suppression rule was intentionally simple:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual audit of that naive rule:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

The learned trust signal changed behavior, but it was too aggressive as the
final suppression decision. It suppressed many useful predictions.

Delta calibration did not solve the issue. Useful and harmful suppressions had
similar confidence drops, so confidence delta alone did not separate bad
predictions from useful ones.

### Quality-Aware Suppression

A separate suppression policy layer was added:

```text
graph prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final suppression
```

Quality-aware v1 exported 12 rows. Manual audit found:

```text
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was:

```text
talbe --made_of--> wood
```

That prediction is almost certainly useful. The problem is a source typo for
`table`, so it belongs in normalization/canonicalization rather than
suppression.

Quality-aware v2 no longer lets source noise trigger suppression. Target noise
still triggers suppression. The output contained 11 rows, all with:

```text
target = oxegen
```

The `talbe --made_of--> wood` row disappeared.

Current interpretation:

* bad target -> suppress
* bad source -> normalize later
* bad relation or pattern -> lower trust
* clean prediction with a source typo -> keep after normalization

### Microworld-style Name/Surname Generation

A makemore-like character generation experiment, implemented as an explicit
graph transition system rather than a neural network. It tests whether feedback
can improve generation through a compact per-transition *trust* profile instead
of weight updates and backpropagation.

**Input dataset.** The experiment works with any one-name-per-line text file —
given names, family names, or a mixed personal-name list. The provided
`data/names.txt` contains a mix of both; the generator makes no assumption
about name type. The goal is not "real surname realism" specifically, but
testing graph-based character generation, audit feedback compression, and
explicit trust learning without neural weights or backpropagation.

Each name is treated as a START-padded sequence of character transitions; for
n-gram order 2 the name `ABRAMIDZE` becomes:

```text
<START><START> -> a
<START>a       -> b
ab             -> r
br             -> a
...
ze             -> <END>
```

The pipeline is the same shape as the rest of Microworld — generate, audit,
compress feedback, regenerate, compare:

```text
name list (given names / surnames / mixed)
-> character-transition graph (counts only)
-> weighted graph walk
-> quality policy (vowel balance, clusters, length, punctuation, duplicates)
-> manual audit (good / bad / unclear)
-> compact trust profile
   (transition trust + shape trust + mined bad-pattern trust)
-> regenerate with learned trust and adjusted quality scores
-> baseline vs learned comparison
```

There are no weights and no backprop anywhere: generation is a counted random
walk, and "learning" is a small JSON of per-transition and per-shape
multipliers. Transition trust biases the next-character walk. Shape trust and
mined bad-pattern trust are applied after the explicit quality policy as
`adjusted_quality_score`, so weak patterns such as overlong glued names, short
fragments, or bad-heavy audited n-grams become visible and can optionally be
filtered during generation. The quality policy is intentionally not
Anglo-centric and does not require classic surname endings — common Russian,
Georgian, Armenian and European endings (`ov`, `ova`, `dze`, `shvili`, `yan`,
`ian`, `sky`, …) are treated as *positive signals* that relax some checks, but
their absence is not penalised. Given-name-like outputs such as `eleanor`,
`eldrick`, and `ebraheem` score as high quality.

Manual audit labels for generated names:

* `good`: plausible generated name
* `unclear`: possible but weak, strange, or uncertain
* `bad`: clear glued name, fragment, common word, brand, typo-like output, or
  poor readability

Examples that should normally be labelled `bad`: `aasyahuvallen`,
`kahlianevital`, `jaileignatty`, `ezeridhavina`, `kamarisselmir`,
`fioreniylanie`, `momodenciszeki`, `qweslienna`, `gateuillis`, `all`, `march`,
`avito`, `kha`, `gen`, `ter`, `kyn`, `yia`.

This is **not** meant as a general neural-generator replacement. The point is
interpretability, auditability, compact feedback learning, and explicit control:
every transition, score, and trust nudge is inspectable, and feedback is stored
as a few hundred bytes of multipliers rather than a weight matrix.

#### Latest Name/Surname Generation Results

Microworld now demonstrates audit-driven learning in both reasoning suppression
and small-scale character generation. For generation, the system uses counted
character transitions, explicit trust profiles, quality diagnostics, and
audit-mined bad-pattern trust. Human audit feedback is converted into explicit
pattern trust, improving this benchmark without neural weights or
backpropagation.

Baseline fresh generation:

```text
good_rate: 0.45
bad_rate: 0.19
unclear_rate: 0.36
generation_precision: 0.7031
```

Positive and weak-negative trust learning changed the output distribution but
did not improve precision:

```text
learned v1:
good_rate: 0.53
bad_rate: 0.29
unclear_rate: 0.18
generation_precision: 0.6463

learned v4 with hand-written diagnostics:
good_rate: 0.51
bad_rate: 0.25
unclear_rate: 0.24
generation_precision: 0.6711
```

Early conclusion:

* transition trust alone changed the output distribution but did not improve
  precision
* hand-written shape diagnostics helped locally but did not generalize enough
* bad outputs still often received high `quality_score` from generic reasons
  such as `looks like a plausible name`, `reasonable_length`, and
  `balanced_vowels`

Audit-mined bad-pattern trust improved the audited run:

```text
good_rate: 0.59
bad_rate: 0.12
unclear_rate: 0.29
generation_precision: 0.8310
```

Interpretation:

* good rate improved from 0.45 to 0.59
* bad rate dropped from 0.19 to 0.12
* precision improved from 0.7031 to 0.8310
* the improvement came from audit-mined explicit pattern trust, not neural
  weights or backpropagation

#### Makemore vs Microworld Benchmark

The benchmark compares Microworld against a small makemore-style PyTorch MLP on
the same input file, `data/surnames.txt`, with 100 generated names and manual
audit labels `good`, `bad`, and `unclear`.

Makemore-style baseline setup:

* character-level vocabulary with end token
* block size 3
* embedding dimension 16
* hidden dimension 200
* 50,000 training steps
* temperature 0.8
* SGD with learning-rate decay
* PyTorch neural model

Quality comparison:

```text
Microworld:
good: 59
bad: 12
unclear: 29
good_rate: 0.59
bad_rate: 0.12
unclear_rate: 0.29
generation_precision: 0.8310

Makemore:
good: 60
bad: 18
unclear: 22
good_rate: 0.60
bad_rate: 0.18
unclear_rate: 0.22
generation_precision: 0.7692
```

Interpretation:

* Makemore produced one more good name
* Microworld produced six fewer bad names
* Microworld had higher human-rated precision in this benchmark: 0.8310 vs
  0.7692
* the difference came mainly from stronger bad-output suppression

#### Efficiency Benchmark

Runtime and state results from the same benchmark:

```text
Microworld:
build_transition_graph_time_sec: ~0.0537
generation_time_sec: ~0.0090
audit_adaptation_time_sec: ~0.00124
total_explicit_state_size_bytes: 285,224
trainable_parameter_count: 0
uses_backpropagation: false
uses_neural_weights: false

Makemore:
training_time_sec: ~10.5
generation_time_sec: ~0.0321
model_state_size_bytes: 65,561
trainable_parameter_count: 15,659
uses_backpropagation: true
uses_neural_weights: true
```

Ratios:

* Makemore training/build path was about 195x slower than Microworld build
* Makemore generation was about 3.6x slower
* Makemore serialized state was smaller: about 65.6 KB vs 285.2 KB
* Microworld used zero trainable parameters

Interpretation: Microworld trades compact dense neural state for larger explicit
inspectable state. Makemore is more compact on disk/state size. Microworld is
faster and more transparent in this small benchmark.

#### RAM / RSS Benchmark

RSS results are approximate runtime memory-pressure measurements:

```text
Microworld:
build peak RSS: ~28.6 MB
generation peak RSS: ~31.2 MB
audit adaptation peak RSS: ~31.8 MB
build RSS delta: ~2.1 MB
generation RSS delta: ~2.5 MB
audit adaptation RSS delta: ~0.5 MB

Makemore:
before training RSS: ~184.0 MB
training peak RSS: ~625.7 MB
after training RSS: ~481.4 MB
generation peak RSS: ~485.1 MB
training RSS delta: ~297.4 MB
generation RSS delta: ~3.4 MB
```

RSS ratios:

* Makemore peak training RSS vs Microworld build peak RSS: about 21.9x higher
* Makemore peak generation RSS vs Microworld generation peak RSS: about 15.5x
  higher
* Makemore training RSS delta vs Microworld build RSS delta: about 143x higher

Caveats:

* RSS includes the Python interpreter, loaded libraries, PyTorch overhead,
  allocator behavior, and cached memory
* peak RSS is sampled and approximate
* serialized state size and runtime RSS are different metrics
* a future subprocess-isolated benchmark would be cleaner

#### Careful Research Claim

On this small audited name-generation benchmark, Microworld achieved higher
human-rated precision than a small makemore-style MLP baseline while using no
trainable parameters, no backpropagation, faster generation, faster feedback
adaptation, and substantially lower peak runtime RSS. The neural baseline
remained more compact in serialized state size.

This does not prove that explicit graph/trust systems are generally superior to
neural models. It does show that, for audit-driven tasks where human feedback
can be converted into explicit trust and pattern memory, a non-neural system can
be competitive or superior in quality, speed, memory, and explainability.

Run it:

```bash
# 1. baseline generation + audit export (works with any name list)
python3 examples/surname_generate.py --input data/names.txt --count 100 --order 2 \
    --output data/generated_names.csv

# 2. label the manual_label column with good / bad / unclear, then:
python3 examples/surname_audit_summary.py --input data/generated_names.csv

# 3. compress the labelled audit into a trust profile
python3 examples/surname_trust_learn.py --input data/generated_names.csv \
    --order 2 --output data/surname_trust_profile.json

# 4. regenerate with learned trust
python3 examples/surname_generate.py --input data/names.txt --order 2 \
    --trust-profile data/surname_trust_profile.json --output data/generated_learned.csv

# optionally reject low adjusted-quality samples while generating
python3 examples/surname_generate.py --input data/names.txt --order 2 \
    --trust-profile data/surname_trust_profile.json \
    --min-adjusted-quality 0.70 --output data/generated_learned_filtered.csv

# or run the whole baseline-vs-learned experiment in one shot
python3 examples/surname_generation_experiment.py --input data/names.txt --order 2 \
    --trust-profile data/surname_trust_profile.json
```

Reproduce the latest audited name-generation benchmark:

```bash
python3 examples/surname_generate.py \
  --input data/surnames.txt \
  --count 100 \
  --order 3 \
  --seed 49 \
  --avoid-duplicates true \
  --soft-max-length 10 \
  --length-end-bias 1.5 \
  --trust-profile data/name_trust_profile_order3_patterns.json \
  --min-adjusted-quality 0.85 \
  --output data/generated_names_order3_patterns.csv

python3 examples/surname_audit_summary.py \
  --input data/generated_names_order3_patterns.csv

python3 examples/makemore_vs_microworld_benchmark.py \
  --input data/surnames.txt \
  --count 100 \
  --order 3 \
  --seed 50 \
  --audit data/generated_names_order3_patterns.csv \
  --trust-profile data/name_trust_profile_order3_patterns.json \
  --makemore-steps 50000 \
  --makemore-embedding-dim 16 \
  --makemore-hidden-dim 200 \
  --makemore-block-size 3 \
  --makemore-temperature 0.8 \
  --track-memory true \
  --memory-sample-interval-ms 10 \
  --output data/makemore_vs_microworld_benchmark_memory.json
```

### worldpgt QA Layer

Microworld/worldpgt contains controlled QA assistants over explicit accepted
memory and an isolated wiki candidate overlay. They answer, distinguish,
explain, or safely audit supported questions using transparent
analyzer/planner/renderer/validator pipelines, without neural weights,
embeddings, network calls, or model-based generation.

LLMs typically learn language first and compress world knowledge into opaque
weights. Microworld takes the opposite route: explicit world memory first, then
controlled language as an interface to that memory.

**What is implemented:**

* question analyzer - detects QA intent from surface form
* answer planner - selects a response strategy from accepted memory
* answer renderer - composes semantic answer forms (common clues, contexts,
  signs, location-aware phrases, action/agency-aware phrases, contrast
  explanations)
* answer validator - checks correctness and flags quality issues
* helpful audit rendering - produces a safe, informative abstention when the
  question is ambiguous without sufficient context
* accepted knowledge memory provider - 221 items (163 facts, 58 patterns,
  6 ambiguous terms, 12 senses, 104 positive cues)
* deterministic wiki ingestion v2 - 50 local curated pages to 283 reviewable
  candidates
* wiki candidate memory overlay v1 - 283 isolated overlay items, safe for entity
  QA overlay, not safe for general runtime
* entity QA v1 - controlled entity questions over the isolated overlay
* entity QA expansion v1 - broader controlled entity QA over the same overlay
* adversarial entity QA v1 - audits relation inversion, weak-link promotion,
  current/live data, unsupported universal claims, source-qualified volatility,
  category confusion, and private/personal data
* cross-page entity QA v1 - controlled graph-style multi-hop QA over the
  isolated overlay
* Wikipedia Self-Ingestion v1 - offline dry-run overlay delta proposal with
  quarantine and deterministic regression gates
* Promote Overlay Delta v1 - validates self-ingestion delta, promotes only safe
  items into a separate promoted overlay, and runs QA/adversarial/cross-page
  regression gates.

**Supported QA intents:** `define_sense`, `classify_context`, `explain_cue`,
`distinguish_senses`, `unknown_or_ambiguous`

**Supported entity QA intents:** `define_entity`, `relation_lookup`,
`link_explanation`, `source_fact_lookup`, `unknown_or_unsupported`

**Current controlled benchmark summary:**

| Benchmark | File | Total | Correct | Wrong | Answered | Audited | Answer precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Main QA | `worldpgt/experiments/qa_prompts_v1.csv` | 48 | 48 | 0 | 42 | 6 | 1.0 |
| Generalization QA | `worldpgt/experiments/qa_generalization_test_v1.csv` | 24 | 24 | 0 | 19 | 5 | 1.0 |
| Entity QA | `worldpgt/experiments/entity_qa_prompts_v1.csv` | 28 | 28 | 0 | 23 | 5 | 1.0 |
| Entity QA expansion | `worldpgt/experiments/entity_qa_expansion_v1_summary.json` | 111 | 111 | 0 | 79 | 32 | 1.0 |
| Adversarial Entity QA | `worldpgt/experiments/entity_qa_adversarial_v1_summary.json` | 68 | 68 | 0 | 6 | 62 | 1.0 |
| Cross-page Entity QA | `worldpgt/experiments/cross_page_qa_v1_summary.json` | 71 | 71 | 0 | 50 | 21 | 1.0 |

Combined: 350 prompts, 350 correct decisions, 0 wrong answers, 219 answered,
131 safely audited.

The renderer no longer emits flat `associated with` lists. Example outputs:

```text
A baseball bat is a club used to hit a ball in sports such as baseball.
Common clues are pitchers, balls, batters and plates.
It is used to swing, hit or strike the ball, and it is often found at home plate.

Spring is the season that follows winter.
Common signs are thaw, flowers, warmer mornings and rain.
In spring, flowers bloom and snow thaws.

Rock music is a music genre linked to bands, concerts and crowds.
A rock is a solid mineral object found near cliffs, boulders and trails.
```

Example helpful audit (safe abstention on a genuinely ambiguous question):

```text
"Seal" is ambiguous: it can mean a marine animal or a wax/document seal.
I need context to choose the right meaning.
```

Supported generalized phrasings include:

```text
Is a bat with wings an animal or sports equipment?
The seal was swimming near the coast. What kind of seal is it?
The crane had a hook and lifted a load. What does crane mean?
The band played rock on stage. What does rock mean?
Why do wings point to bat as an animal?
```

Conflicting cue prompts audit rather than force an answer.

Entity QA examples:

```text
What does SpaceX develop?
SpaceX develops rockets and spacecraft.

Why is Forbes linked to Elon Musk?
Forbes is linked from the Elon Musk page as a weak contextual mention. It is
not treated as a stable factual relation by this overlay.

What does Forbes estimate about Elon Musk?
According to Forbes as of 2026-06, Elon Musk's estimated net worth is US$1.1
trillion. This is a volatile source-qualified estimate and should be rechecked.
```

**Safety constraints:**

* no neural weights
* no backpropagation
* no fine-tuning
* no GPT renderer
* no embeddings
* no network calls
* no generic trusted fallback
* audits are safe behavior, not failures
* source-qualified volatile facts require recheck
* weak context links are not treated as facts
* current unsupported questions audit
* accepted memory v1 is not modified by wiki overlay
* self-ingestion uses a separate dry-run overlay
* promotion uses a separate promoted overlay artifact
* volatile facts are never auto-applied as stable facts
* `safe_for_general_runtime` remains false for wiki overlay

Previously fixed caveat: an older directional relation verbalization bug could
make `Who is Elon Musk?` say `founded by SpaceX`; this was a surface
verbalization issue, not an accepted-memory, planner, or overlay issue.

See `worldpgt/` for the full QA package and
`worldpgt/RESEARCH_SNAPSHOT.md` for the research history.

## What Was Learned

The main research conclusion is deliberately modest:

* audit feedback can be compressed into a tiny explicit trust state
* that state can transfer to unseen data
* behavior can change without backpropagation
* errors can be debugged directly
* small explicit policy layers can sharply improve behavior
* trust memory, decision policy, and normalization should be separate components

Microworld does not prove general superiority over neural networks. It shows
that explicit graph memory and audit-driven trust learning are useful research
tools for a bounded class of symbolic reasoning problems.

## Running

From this directory:

```bash
pytest -q
```

Example demos:

```bash
python3 examples/full_pipeline_demo.py
python3 examples/trust_transfer_experiment.py
python3 examples/feedback_scaling_benchmark.py
python3 examples/suppression_audit_export.py
```

## Documentation

Start with:

* `docs/index.md`
* `docs/architecture.md`
* `docs/experiments.md`
* `docs/suppression_policy.md`
* `worldpgt/README.md`
* `worldpgt/RESEARCH_SNAPSHOT.md`
* `worldpgt/docs/WIKI_OVERLAY.md`
* `worldpgt/docs/CROSS_PAGE_QA.md`
* `worldpgt/docs/SELF_INGESTION.md`
* `worldpgt/docs/SAFETY_MODEL.md`

## Limitations

* Microworld is not a general-purpose language model.
* Current QA results are controlled benchmark results over supported domains.
* Scope remains narrow and inputs are curated.
* The analyzers are rule/curriculum-based and need explicit expansion.
* The wiki corpus is a 50-page local fixture, not live Wikipedia/Wikidata.
* The wiki overlay is isolated and not accepted memory v1 or general runtime
  memory.
* Weak context links are contextual mentions, not factual relations.
* Source-qualified volatile facts require recheck.
* Current facts are not answered as live truth.
* Source extraction is still narrow.
* There is no autonomous web ingestion.
* Promotion exists only as a separate promoted overlay artifact; it still does
  not modify trusted accepted memory or accepted wiki overlay.
* Renderer surface quality is still being polished.
* Current results are exploratory and based on bounded graph reasoning tasks.
* The ConceptNet work uses filtered samples, not a full benchmark.
* Manual audits are still small.
* The name-generation benchmark uses 100 generated names per audited run.
* Manual name audits are subjective.
* Current generation results use one dataset and one makemore-style baseline
  configuration.
* More seeds are needed.
* Larger 500/1000-name runs are needed.
* Subprocess-isolated RAM benchmarking would be cleaner.
* The current explicit generation state is larger than the neural model file.
* Results should be treated as experimental evidence, not a general proof.
* Mixed-pattern reasoning is conservative and manually allowlisted.
* Trust learning is useful as a signal but should not be the final decision layer.
* Delta-only suppression calibration did not separate useful from harmful cases.
* Normalization/canonicalization is not yet a mature component.
* There is no perception layer, reinforcement learning loop, or neural training.

## Next Steps

**Graph / trust / name generation:**

* normalization candidate export
* typo/canonicalization layer
* target normalization with semantic re-evaluation
* larger audit sample
* relation-specific suppression policies
* compare against LLM-only memory baselines
* repeat the name benchmark across 3-5 seeds
* generate and audit 500-1000 samples
* add subprocess-isolated memory benchmarks
* compare multiple makemore configurations
* add disk-backed Microworld state using SQLite
* track quality-per-RAM, quality-per-training-second, and quality-per-parameter
* extend audit-mined trust to relation filtering, entity normalization,
  reasoning suppression, and data cleaning

**worldpgt QA layer:**

1. Promoted Overlay Provider v1 - load the separate promoted overlay artifact
   for controlled QA runs without modifying trusted accepted memory or the
   current accepted overlay.
2. Add a repeated efficiency benchmark with median/min/max.
3. Compare with a GPT-style baseline on the same controlled questions.
4. Continue generalized analyzer curriculum work while preserving safety on
   conflicting prompts.
5. Interactive QA playground - one-off CLI:
   `python3 -m worldpgt.experiments.ask_answer_planner_v1 --question "..."`
   with optional JSON and trace output
6. Scale accepted memory - more terms, more senses, disk/index-backed provider
   later

## Status

Experimental. The goal is not to build a production knowledge graph or a
replacement for LLMs, but to create a controlled environment where hypotheses
about explicit memory, reasoning, feedback compression, and inspectable learning
can be tested.
