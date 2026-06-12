# worldmvp / Microworld

Microworld is an experimental graph-based memory, reasoning, and learning
system. It explores whether useful behavioral learning can happen through
explicit graph state, audit feedback, and trust calibration without neural
weights, backpropagation, or fine-tuning.

The project is intentionally research-oriented. It does not claim that symbolic
graphs beat neural networks. It explores a complementary path: compact explicit
memory and trust learning for graph reasoning, where behavior can be audited,
compressed, transferred, and corrected without updating neural weights.

Current test status:

```text
870 passing tests
```

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

## Latest Experimental Results

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
-> compact transition trust profile (good *= 1.05, bad *= 0.85, bounded 0.1..2.0)
-> regenerate with learned trust
-> baseline vs learned comparison
```

There are no weights and no backprop anywhere: generation is a counted random
walk, and "learning" is a small JSON of per-transition multipliers that biases
the walk. The quality policy is intentionally not Anglo-centric and does not
require classic surname endings — common Russian, Georgian, Armenian and
European endings (`ov`, `ova`, `dze`, `shvili`, `yan`, `ian`, `sky`, …) are
treated as *positive signals* that relax some checks, but their absence is not
penalised. Given-name-like outputs such as `eleanor`, `eldrick`, and `ebraheem`
score as high quality.

This is **not** meant to beat neural character generators. The point is
interpretability, auditability, compact feedback learning, and explicit control:
every transition, score, and trust nudge is inspectable, and feedback is stored
as a few hundred bytes of multipliers rather than a weight matrix.

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

# or run the whole baseline-vs-learned experiment in one shot
python3 examples/surname_generation_experiment.py --input data/names.txt --order 2 \
    --trust-profile data/surname_trust_profile.json
```

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

## Limitations

* Current results are exploratory and based on bounded graph reasoning tasks.
* The ConceptNet work uses filtered samples, not a full benchmark.
* Manual audits are still small.
* Mixed-pattern reasoning is conservative and manually allowlisted.
* Trust learning is useful as a signal but should not be the final decision layer.
* Delta-only suppression calibration did not separate useful from harmful cases.
* Normalization/canonicalization is not yet a mature component.
* There is no perception layer, reinforcement learning loop, or neural training.

## Next Steps

* normalization candidate export
* typo/canonicalization layer
* target normalization with semantic re-evaluation
* larger audit sample
* relation-specific suppression policies
* compare against LLM-only memory baselines

## Status

Experimental. The goal is not to build a production knowledge graph or a
replacement for LLMs, but to create a controlled environment where hypotheses
about explicit memory, reasoning, feedback compression, and inspectable learning
can be tested.
