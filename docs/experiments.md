# Experiments

Microworld experiments are exploratory research checks for explicit graph
reasoning and audit-driven learning. They should be read as bounded evidence,
not as claims of general superiority over neural systems.

The current research question is:

```text
Can feedback over symbolic graph predictions be compressed into explicit state
that changes future behavior without neural retraining?
```

Current answer: partially yes.

## Human Audit Baseline

Manual audit of ConceptNet-derived predictions:

```text
reviewed: 104
useful overall: 78.8%
made_of useful: 86.2%
part_of useful: 76.7%
is_a useful: 75.6%
mixed reasoning useful: 76.7%
```

`useful` means `correct + plausible`.

Interpretation:

* relation-specific graph reasoning can produce useful predictions
* mixed reasoning is viable but still conservative
* manual audit is necessary because automatic graph completion metrics can mark
  plausible missing edges as false positives

Limit:

* this is a small exploratory audit, not a formal benchmark

## Audit-Driven Trust Learning

Microworld maps manual audit feedback into explicit trust state. The trust state
can lower or raise confidence for relation families, rules, drift categories,
and evidence sources.

Learning path:

```text
manual audit rows
-> compact trust profile
-> prediction confidence adjustment
-> changed behavior on future data
```

This does not use gradient descent, backpropagation, fine-tuning, or hidden
weight updates.

## Trust Transfer Experiment

Trust learned from audit data was evaluated on an unseen TEST split.

```text
baseline accepted: 195
learned accepted: 99
suppressed: 96
```

Interpretation:

```text
feedback -> trust profile -> changed behavior on unseen split
```

The learned trust profile transferred to new rows and changed acceptance
behavior without neural retraining.

## Feedback Compression Benchmark

Benchmark result for 10,000 audit rows:

```text
raw audit history: ~500,360 tokens
trust state: ~313 tokens
compression: ~1598.6x
```

This is a central result because it separates memory from transcript length. The
system does not need to keep the full audit history in context to apply what was
learned. It compresses repeated feedback into explicit reusable buckets.

Interpretation:

```text
large feedback history
-> tiny explicit state
-> future behavior change
```

## Naive Suppression Audit

Initial suppression used a direct trust-threshold rule:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual audit:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

Interpretation:

* learned trust changed behavior
* trust alone was too aggressive as a final suppression decision
* many useful predictions were suppressed

Conclusion:

```text
trust learning is useful as a signal, not as the final decision layer
```

## Delta Calibration

The next attempt used confidence drop magnitude as a calibration signal.

Result:

* useful suppressions and harmful suppressions had similar confidence drops
* delta magnitude alone did not separate bad suppressions from useful predictions

Interpretation:

```text
confidence delta is diagnostic context, not a complete suppression policy
```

## Quality-Aware Suppression

A separate suppression policy layer was added:

```text
graph prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final suppression
```

Quality-aware v1:

```text
exported rows: 12
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was:

```text
talbe --made_of--> wood
```

Reason:

* the prediction is useful
* `talbe` is almost certainly a typo for `table`
* this should be handled by normalization/canonicalization, not suppression

Quality-aware v2:

* source noise no longer triggers suppression
* target noise still triggers suppression
* output contained 11 rows
* all 11 rows had `target = oxegen`
* `talbe --made_of--> wood` disappeared

Interpretation:

* bad target -> suppress
* bad source -> normalize later
* bad relation or pattern -> lower trust
* clean prediction with source typo -> keep after normalization

## Name/Surname Generation Experiment

Microworld now demonstrates audit-driven learning in both reasoning suppression
and small-scale character generation. The generation experiment is a
makemore-like name generator implemented without neural networks. It uses
counted character transitions, explicit trust profiles, quality diagnostics,
and audit-mined bad-pattern trust.

Learning path:

```text
generated names
-> human audit labels
-> explicit transition trust, shape trust, and pattern trust
-> adjusted generation quality
-> regenerated names
```

This path does not use neural weights, backpropagation, or fine-tuning.

Baseline fresh generation:

```text
good_rate: 0.45
bad_rate: 0.19
unclear_rate: 0.36
generation_precision: 0.7031
```

Positive and weak-negative trust learning:

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

Conclusion from early attempts:

* transition trust alone changed the output distribution but did not improve
  precision
* hand-written shape diagnostics helped locally but did not generalize enough
* bad outputs still often received high `quality_score` from generic reasons
  such as `looks like a plausible name`, `reasonable_length`, and
  `balanced_vowels`

Audit-mined bad-pattern trust:

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

## Makemore vs Microworld Benchmark

The benchmark compares Microworld against a small makemore-style PyTorch MLP.
Both systems use the same input file, `data/surnames.txt`, and generate 100
names. Quality is measured only where human labels are available.

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

## Efficiency Benchmark

Runtime and state results:

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

Interpretation:

* Microworld trades compact dense neural state for larger explicit inspectable
  state
* Makemore is more compact on disk/state size
* Microworld is faster and more transparent in this benchmark

## RAM / RSS Benchmark

RSS results:

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

Ratios:

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

## Reproduction Commands

Generate the audited Microworld names:

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
```

Summarize the manual audit:

```bash
python3 examples/surname_audit_summary.py \
  --input data/generated_names_order3_patterns.csv
```

Run the makemore-vs-Microworld benchmark with memory tracking:

```bash
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

## Main Research Conclusion

Microworld supports a narrow but useful result:

* audit feedback can be compressed into a tiny explicit trust state
* that state can transfer to unseen data
* behavior can change without backpropagation
* errors can be debugged directly
* adding explicit policy layers can sharply improve behavior
* trust memory, decision policy, and normalization should be separate components
* on one small audited name-generation benchmark, explicit audit-derived memory
  achieved higher human-rated precision than a small makemore-style MLP baseline

It does not prove that graph systems are generally superior to neural networks.
It shows that explicit memory and trust learning can be useful for bounded
audit-driven tasks where inspectable state and compact feedback memory matter.

## Limitations

* sample size is small: 100 generated names per audited generation run
* manual audit is subjective
* current generation results use one dataset
* current neural comparison uses one makemore-style baseline configuration
* more seeds are needed
* larger 500/1000-name runs are needed
* subprocess-isolated RAM benchmarking would be cleaner
* the current explicit state is larger than the neural model file
* results should be treated as experimental evidence, not a general proof

## Suggested Comparisons

Next research comparisons should include:

* larger audit samples
* repeat the name benchmark across 3-5 seeds
* generate and audit 500-1000 samples
* relation-specific suppression policies
* normalization candidate export
* typo/canonicalization layer
* target normalization with semantic re-evaluation
* LLM-only memory baselines using long context or transcript history
* multiple makemore configurations
* subprocess-isolated memory benchmarks
* disk-backed Microworld state using SQLite
* quality-per-RAM, quality-per-training-second, and quality-per-parameter
* audit-mined trust in other domains: relation filtering, entity normalization,
  reasoning suppression, and data cleaning
