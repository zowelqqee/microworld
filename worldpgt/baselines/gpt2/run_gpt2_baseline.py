"""Inference-only GPT-2 baseline runner using local nanoGPT when possible.

Usage:
    python3 -m worldpgt.baselines.gpt2.run_gpt2_baseline \
        --input worldpgt/experiments/continuation_prompts_v1.csv \
        --output worldpgt/experiments/gpt2_continuation_outputs_smoke.csv \
        --nanogpt-dir nanogpt \
        --device mps \
        --max-new-tokens 32 \
        --temperature 0.8 \
        --top-k 40 \
        --num-samples 1 \
        --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from worldpgt.baselines.gpt2.parse_outputs import (
    extract_completion,
    load_prompt_rows,
    write_gpt2_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPT-2 baseline continuations over prompt CSV.")
    parser.add_argument("--input", required=True, help="Input prompt CSV")
    parser.add_argument("--output", required=True, help="Output GPT-2 baseline CSV")
    parser.add_argument("--nanogpt-dir", default=None, help="Path to local nanoGPT checkout")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu, mps, cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    return parser


def resolve_nanogpt_dir(nanogpt_dir: str | None = None, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    candidates = [nanogpt_dir] if nanogpt_dir else ["nanogpt", "nanoGPT"]
    tried = []
    for candidate in candidates:
        path = Path(candidate)
        resolved = path if path.is_absolute() else root / path
        tried.append(str(resolved))
        if (resolved / "model.py").exists():
            return resolved
    raise FileNotFoundError(f"Could not find local nanoGPT model.py. Tried: {tried}")


@contextmanager
def _temporary_sys_path(path: Path):
    path_text = str(path)
    inserted = path_text not in sys.path
    if inserted:
        sys.path.insert(0, path_text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass


def _load_nanogpt_generator(
    nanogpt_dir: Path,
    device: str,
    seed: int,
    compile_model: bool,
) -> tuple[str, Callable[[str, int, float, int], str]]:
    try:
        import torch
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(f"nanoGPT backend dependency import failed: {exc}") from exc

    with _temporary_sys_path(nanogpt_dir):
        try:
            from model import GPT
        except Exception as exc:
            raise RuntimeError(f"nanoGPT local import failed: {exc}") from exc

    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
        model.eval()
        model.to(device)
        if compile_model:
            model = torch.compile(model)
        enc = tiktoken.get_encoding("gpt2")
    except Exception as exc:
        raise RuntimeError(f"nanoGPT GPT-2 load failed: {exc}") from exc

    def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        encoded = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        x = torch.tensor(encoded, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        return enc.decode(y[0].tolist())

    return "nanogpt:gpt2", generate


def _load_transformers_generator(
    device: str,
    seed: int,
    compile_model: bool,
) -> tuple[str, Callable[[str, int, float, int], str]]:
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast, set_seed
    except ImportError as exc:
        raise RuntimeError(f"transformers fallback dependency import failed: {exc}") from exc

    set_seed(seed)
    torch.manual_seed(seed)
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        model.to(device)
        if compile_model:
            model = torch.compile(model)
    except Exception as exc:
        raise RuntimeError(f"transformers GPT-2 load failed: {exc}") from exc

    def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0], skip_special_tokens=False)

    return "transformers:gpt2", generate


def load_generator(
    nanogpt_dir: Path,
    device: str,
    seed: int,
    compile_model: bool,
) -> tuple[str, Callable[[str, int, float, int], str], str | None]:
    try:
        model_name, generator = _load_nanogpt_generator(nanogpt_dir, device, seed, compile_model)
        return model_name, generator, None
    except RuntimeError as nanogpt_error:
        try:
            model_name, generator = _load_transformers_generator(device, seed, compile_model)
            return model_name, generator, str(nanogpt_error)
        except RuntimeError as fallback_error:
            raise RuntimeError(
                "Unable to load GPT-2 via local nanoGPT or transformers fallback. "
                f"nanoGPT error: {nanogpt_error}; fallback error: {fallback_error}"
            ) from fallback_error


def run(args: argparse.Namespace) -> list[dict]:
    nanogpt_dir = resolve_nanogpt_dir(args.nanogpt_dir)
    prompt_rows = load_prompt_rows(args.input)
    if args.limit is not None:
        prompt_rows = prompt_rows[: args.limit]

    model_name, generator, fallback_reason = load_generator(
        nanogpt_dir=nanogpt_dir,
        device=args.device,
        seed=args.seed,
        compile_model=args.compile_model,
    )
    if fallback_reason:
        print(f"Local nanoGPT backend failed; used fallback. Reason: {fallback_reason}")
    else:
        print(f"Using local nanoGPT backend at {nanogpt_dir}")

    output_rows = []
    for row in prompt_rows:
        prompt = row.get("prompt", "")
        for sample_index in range(args.num_samples):
            start = time.perf_counter()
            full_text = generator(prompt, args.max_new_tokens, args.temperature, args.top_k)
            elapsed = time.perf_counter() - start
            completion = extract_completion(prompt, full_text)
            output_rows.append(
                {
                    "id": row.get("id", ""),
                    "prompt": prompt,
                    "ambiguous_term": row.get("ambiguous_term", ""),
                    "expected_sense": row.get("expected_sense", ""),
                    "difficulty_type": row.get("difficulty_type", ""),
                    "notes": row.get("notes", ""),
                    "model": model_name,
                    "sample_index": sample_index,
                    "completion": completion,
                    "full_text": full_text,
                    "generation_time_sec": f"{elapsed:.4f}",
                    "device": args.device,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "seed": args.seed,
                }
            )

    write_gpt2_rows(args.output, output_rows)
    return output_rows


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} GPT-2 baseline rows to {args.output}")


if __name__ == "__main__":
    main()
