# Microworld Documentation

Microworld is an explainable graph reasoning sandbox.  It represents knowledge
as explicit nodes and typed relations, then studies which symbolic reasoning
rules work, where they break, and how failures can be explained.

The current project is not a neural model and does not use a trainable
`weights + biases` scorer.  Confidence is computed from interpretable graph
statistics and audit-informed heuristics:

- discovered relation patterns
- hub penalties
- relation trust priors
- node quality
- relation drift penalties
- disabled noisy relation filters

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
7. [Error Taxonomy](error_taxonomy.md)

## Main Entry Points

- `examples/application_demo.py` - application-level report for humans
- `core/pattern_prediction.py` - transitive and mixed pattern prediction
- `core/relation_proposal.py` - learned relation-label proposal
- `core/relation_drift.py` - semantic-level drift analysis
- `examples/pattern_audit_export.py` - prediction audit export
- `examples/relation_drift_report.py` - drift report

