"""MLX runner. Importing this module never downloads a model."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Iterable

from .dataset import read_jsonl

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
SYSTEM = """You answer questions using only the supplied evidence.
Do not use outside knowledge.
Do not infer claims that are not explicitly supported.
If the evidence does not contain the answer, output exactly:
UNKNOWN
For multi-evidence cases: preserve disagreements between evidence spans. Do not merge conflicting claims into one certain statement. Keep the answer concise."""


def prompt_for(case: dict) -> str:
    evidence = "\n\n".join(f"Evidence {index}:\n{text}" for index, text in enumerate(case["contexts"], 1))
    return f"System:\n{SYSTEM}\n\nUser:\n{evidence}\n\nQuestion:\n{case['question']}\n\nAnswer:\n"


def _model_prompt(tokenizer, case: dict) -> str:
    """Use Qwen's installed chat template when available.

    The information and wording are identical to :func:`prompt_for`; the
    template merely supplies Qwen's required turn delimiters, preventing it
    from continuing the literal ``User:`` transcript after a valid answer.
    """
    evidence = "\n\n".join(f"Evidence {index}:\n{text}" for index, text in enumerate(case["contexts"], 1))
    user = f"{evidence}\n\nQuestion:\n{case['question']}\n\nAnswer:"
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return prompt_for(case)
    return apply([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def _load(model_name: str):
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("MLX-LM is required. Install it with: python3 -m pip install -U mlx-lm") from exc
    model, tokenizer = load(model_name)
    return model, tokenizer, generate, make_sampler(temp=0.0)


def run(dataset: Iterable[dict], *, model_name: str = MODEL, warmups: int = 50, repeats: int = 5, seed: int = 42) -> tuple[list[dict], dict]:
    before = time.perf_counter_ns(); model, tokenizer, generate, sampler = _load(model_name)
    load_ms = (time.perf_counter_ns() - before) / 1_000_000
    rows = list(dataset)
    def invoke(case: dict) -> dict:
        prompt = _model_prompt(tokenizer, case); limit = 256 if case["category"] == "multi_evidence" else 128
        started = time.perf_counter_ns()
        try:
            answer = generate(model, tokenizer, prompt=prompt, max_tokens=limit, sampler=sampler, verbose=False)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            prompt_tokens = len(tokenizer.encode(prompt))
            generated_tokens = len(tokenizer.encode(answer))
            return {"id": case["id"], "answer": answer.strip(), "exact_unknown": answer.strip() == "UNKNOWN",
                    "prompt_token_count": prompt_tokens, "generated_token_count": generated_tokens,
                    "ttft_ms": None, "total_latency_ms": elapsed,
                    "tokens_per_second": generated_tokens / (elapsed / 1000) if elapsed else None, "exception": None}
        except Exception as exc:
            return {"id": case["id"], "answer": "", "exact_unknown": False, "prompt_token_count": None,
                    "generated_token_count": None, "ttft_ms": None,
                    "total_latency_ms": (time.perf_counter_ns() - started) / 1_000_000, "tokens_per_second": None, "exception": repr(exc)}
    for index in range(min(warmups, max(1, len(rows)))): invoke(rows[index % len(rows)])
    order = list(range(len(rows))) * repeats; random.Random(seed).shuffle(order)
    results = []
    for sequence, index in enumerate(order):
        row = invoke(rows[index]); row["repeat"] = sequence // len(rows); results.append(row)
    return results, {"system": "Qwen2.5-0.5B-Instruct 4-bit", "model": model_name, "startup_ms": load_ms,
                     "warmup_queries": warmups, "repeats": repeats, "generated_timestamp": datetime.now(timezone.utc).isoformat(),
                     "ttft_available": False, "note": "MLX generate API did not expose TTFT; recorded as null rather than estimated."}


def run_file(dataset_path: str | Path, output: str | Path, **kwargs: object) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True); results, metadata = run(read_jsonl(dataset_path), **kwargs)
    (output / "qwen_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    (output / "qwen_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "latency_samples_qwen.csv").write_text("id,total_latency_ms,ttft_ms\n" + "".join(f"{row['id']},{row['total_latency_ms']:.6f},{row['ttft_ms'] or ''}\n" for row in results), encoding="utf-8")
    (output / "failures_qwen.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results if row["exception"]), encoding="utf-8")
    return metadata
