# Microworld Documentation

Microworld is an explainable graph reasoning sandbox.  It represents knowledge
as explicit nodes and typed relations, then studies which symbolic reasoning
rules work, where they break, and how failures can be explained.

The current project is not a neural model and does not use a trainable
`weights + biases` scorer. Confidence is computed from interpretable graph
statistics, audit-informed trust state, and explicit decision policies:

- discovered relation patterns
- hub penalties
- relation trust priors
- node quality
- relation drift penalties
- disabled noisy relation filters
- audit-driven trust learning
- quality-aware suppression policy

Current research result:

```text
feedback -> compact trust profile -> changed behavior on unseen data
```

The strongest current compression benchmark reduces 10,000 audit rows from
about 500,360 raw-history tokens to about 313 trust-state tokens, or roughly
1598.6x compression.

The latest small audited name-generation benchmark compares explicit
audit-derived Microworld memory against a small makemore-style MLP baseline. In
that 100-name run, Microworld reached higher human-rated precision
(`0.8310` vs `0.7692`) while using no trainable parameters or backpropagation.
This is experimental evidence for one bounded task, not a general claim that
graph/trust systems are superior to neural models.

## Quick Start

From the project root:

```bash
cd /Users/arseniyabramidze/mini-worldgrad/worldmvp
python3 examples/application_demo.py
```

Run the full test suite:

```bash
pytest -q
```

Generate an audit CSV:

```bash
python3 examples/pattern_audit_export.py \
  --input data/conceptnet_sample.csv \
  --output data/pattern_audit.csv \
  --mode transitive \
  --limit 100 \
  --use-relation-drift
```

## Recommended Reading Order

1. [Overview](overview.md)
2. [Architecture](architecture.md)
3. [Reasoning Layers](reasoning_layers.md)
4. [Demos](demos.md)
5. [Audit And Evaluation](audit_and_evaluation.md)
6. [Relation Drift](relation_drift.md)
7. [Experiments](experiments.md)
8. [Suppression Policy](suppression_policy.md)
9. [Error Taxonomy](error_taxonomy.md)

## Main Entry Points

- `examples/application_demo.py` - application-level report for humans
- `core/pattern_prediction.py` - transitive and mixed pattern prediction
- `core/relation_proposal.py` - learned relation-label proposal
- `core/relation_drift.py` - semantic-level drift analysis
- `examples/pattern_audit_export.py` - prediction audit export
- `examples/relation_drift_report.py` - drift report
- `examples/trust_transfer_experiment.py` - audit-driven trust transfer check
- `examples/feedback_scaling_benchmark.py` - feedback compression benchmark
- `examples/suppression_audit_export.py` - suppression audit candidate export
