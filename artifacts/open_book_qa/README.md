# Open-book QA comparison

## Reproduction

```bash
python3 -m worldpgt.benchmarks.open_book_qa.cli build-dataset --overlay pump-dry-run+experimental-web-graph --output artifacts/open_book_qa --seed 42
python3 -m worldpgt.benchmarks.open_book_qa.cli run-microworld --dataset artifacts/open_book_qa/dataset.jsonl --output artifacts/open_book_qa
python3 -m pip install -U mlx-lm
python3 -m worldpgt.benchmarks.open_book_qa.cli run-qwen --dataset artifacts/open_book_qa/dataset.jsonl --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --output artifacts/open_book_qa
python3 -m worldpgt.benchmarks.open_book_qa.cli evaluate --dataset artifacts/open_book_qa/dataset.jsonl --microworld artifacts/open_book_qa/microworld_results.jsonl --qwen artifacts/open_book_qa/qwen_results.jsonl --output artifacts/open_book_qa
python3 -m worldpgt.benchmarks.plot_open_book_qa_comparison --summary artifacts/open_book_qa/comparison_summary.json --output artifacts/open_book_qa/figures
```

## Configuration

- Hardware/runtime at dataset build: `macOS-26.5-arm64-arm-64bit-Mach-O`, Python `3.13.7`.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.
- Overlay: `pump-dry-run+experimental-web-graph`; fingerprint: `4499eb5f40c5cc1627333ae9c569aec124857131cc4bdaf6148c9d8c605b2e7a`; seed: `42`.
- Dataset: 250 cases ({'paraphrase': 50, 'negative': 50, 'direct': 100, 'multi_evidence': 50}).

## Methodology and limits

Both systems receive the same question and original evidence spans selected for
the case. Microworld serves through its real API path over the prepared proposal
relation graph; Qwen receives raw spans only. This compares serving/inference
over one factual source, not equivalent internal representations. Pump/index
build and model training are excluded from warm-query latency. Provenance is a
native MicroWorld output; Qwen is not asked to emit source IDs. Unsupported
claim detection is surface-based and not a semantic verifier. Do not interpret
results as a general intelligence comparison.
