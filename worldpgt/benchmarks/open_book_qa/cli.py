"""Command line entrypoint for the reproducible open-book comparison."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path

from .dataset import write_dataset
from .microworld_runner import run_file as run_microworld
from .qwen_runner import MODEL, run_file as run_qwen
from .evaluate import write_evaluation


def _readme(output: Path) -> None:
    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    output.joinpath("README.md").write_text(f"""# Open-book QA comparison

## Reproduction

```bash
python3 -m worldpgt.benchmarks.open_book_qa.cli build-dataset --overlay pump-dry-run+experimental-web-graph --output artifacts/open_book_qa --seed 42
python3 -m worldpgt.benchmarks.open_book_qa.cli run-microworld --dataset artifacts/open_book_qa/dataset.jsonl --output artifacts/open_book_qa
python3 -m pip install -U mlx-lm
python3 -m worldpgt.benchmarks.open_book_qa.cli run-qwen --dataset artifacts/open_book_qa/dataset.jsonl --model {MODEL} --output artifacts/open_book_qa
python3 -m worldpgt.benchmarks.open_book_qa.cli evaluate --dataset artifacts/open_book_qa/dataset.jsonl --microworld artifacts/open_book_qa/microworld_results.jsonl --qwen artifacts/open_book_qa/qwen_results.jsonl --output artifacts/open_book_qa
python3 -m worldpgt.benchmarks.plot_open_book_qa_comparison --summary artifacts/open_book_qa/comparison_summary.json --output artifacts/open_book_qa/figures
```

## Configuration

- Hardware/runtime at dataset build: `{platform.platform()}`, Python `{platform.python_version()}`.
- Model: `{MODEL}`.
- Overlay: `{summary['overlay']}`; fingerprint: `{summary['overlay_fingerprint']}`; seed: `{summary['random_seed']}`.
- Dataset: {summary['total_cases']} cases ({summary['cases_per_category']}).

## Methodology and limits

Both systems receive the same question and original evidence spans selected for
the case. Microworld serves through its real API path over the prepared proposal
relation graph; Qwen receives raw spans only. This compares serving/inference
over one factual source, not equivalent internal representations. Pump/index
build and model training are excluded from warm-query latency. Provenance is a
native MicroWorld output; Qwen is not asked to emit source IDs. Unsupported
claim detection is surface-based and not a semantic verifier. Do not interpret
results as a general intelligence comparison.
""", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build-dataset", "all"):
        item = sub.add_parser(name); item.add_argument("--overlay", default="pump-dry-run+experimental-web-graph"); item.add_argument("--output", required=True); item.add_argument("--seed", type=int, default=42)
        if name == "all": item.add_argument("--model", default=MODEL)
    for name in ("run-microworld", "run-qwen"):
        item = sub.add_parser(name); item.add_argument("--dataset", required=True); item.add_argument("--output", required=True); item.add_argument("--seed", type=int, default=42); item.add_argument("--repeats", type=int, default=5); item.add_argument("--warmups", type=int, default=50)
        if name == "run-qwen": item.add_argument("--model", default=MODEL)
    item = sub.add_parser("evaluate"); item.add_argument("--dataset", required=True); item.add_argument("--microworld", required=True); item.add_argument("--qwen", required=True); item.add_argument("--output", required=True)
    args = parser.parse_args(argv); output = Path(args.output)
    if args.command == "build-dataset": write_dataset(output, overlay=args.overlay, seed=args.seed); _readme(output)
    elif args.command == "run-microworld": run_microworld(args.dataset, output, seed=args.seed, repeats=args.repeats, warmups=args.warmups)
    elif args.command == "run-qwen": run_qwen(args.dataset, output, model_name=args.model, seed=args.seed, repeats=args.repeats, warmups=args.warmups)
    elif args.command == "evaluate": write_evaluation(args.dataset, args.microworld, args.qwen, output)
    else:
        write_dataset(output, overlay=args.overlay, seed=args.seed); _readme(output)
        run_microworld(output / "dataset.jsonl", output, seed=args.seed)
        run_qwen(output / "dataset.jsonl", output, model_name=args.model, seed=args.seed)
        write_evaluation(output / "dataset.jsonl", output / "microworld_results.jsonl", output / "qwen_results.jsonl", output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
