"""Explicit state and model-size comparison for Microworld and GPT-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory


GPT2_PARAMETER_COUNT = 123_650_000


def serialize_microworld_state() -> dict:
    memory = ExplicitSenseMemory()
    policy = ContinuationPolicy()
    senses = []
    for term in sorted(memory.known_terms()):
        for entry in memory.get_senses(term):
            senses.append(
                {
                    "term": entry.term,
                    "sense_id": entry.sense_id,
                    "cues": entry.cues,
                    "continuations": entry.continuations,
                    "continuation_templates": entry.continuation_templates,
                    "trust": entry.trust,
                }
            )
    state = {
        "sense_memory": senses,
        "anti_cues": {
            f"{term}:{sense_id}": phrases
            for (term, sense_id), phrases in sorted(memory._anti_cues.items())
        },
        "policy_config": {
            "min_score": policy.min_score,
            "min_margin": policy.min_margin,
            "banned_patterns": policy.banned_patterns,
        },
    }
    return state


def _json_size_bytes(value: dict) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def locate_gpt2_cached_weight_size() -> tuple[int | None, str]:
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--gpt2",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    ]
    suffixes = {".bin", ".safetensors", ".pt", ".pth"}
    matched: list[Path] = []
    for base in candidates:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in suffixes and "gpt2" in str(path).lower():
                matched.append(path)
    if not matched:
        return None, "No local GPT-2 weight file was found in common HuggingFace/Torch cache paths; no download attempted."
    return sum(path.stat().st_size for path in matched), f"Summed {len(matched)} local GPT-2-like cached weight files."


def build_summary(nanogpt_dir: str | Path) -> dict:
    state = serialize_microworld_state()
    state_size = _json_size_bytes(state)
    model_state_size, model_state_note = locate_gpt2_cached_weight_size()
    nanogpt_path = Path(nanogpt_dir)
    nanogpt_source_size = _directory_size(nanogpt_path) if nanogpt_path.exists() else None
    return {
        "microworld": {
            "explicit_state_size_bytes": state_size,
            "explicit_state_items": {
                "sense_entries": len(state["sense_memory"]),
                "anti_cue_groups": len(state["anti_cues"]),
                "policy_config_fields": len(state["policy_config"]),
            },
            "guard_policy_notes": [
                "bat:sports_equipment has a player-alone guard in ExplicitSenseMemory._guard_failures",
                "bank:financial_institution has a cash/card-alone guard in ExplicitSenseMemory._guard_failures",
            ],
            "trainable_parameter_count": 0,
            "uses_neural_weights": False,
            "uses_backpropagation": False,
            "state_note": "Size is UTF-8 JSON serialization of built-in explicit sense memory, templates, anti-cues, and policy config; code-level guard rules are reported separately.",
        },
        "gpt2": {
            "model_name": "gpt2",
            "trainable_parameter_count": GPT2_PARAMETER_COUNT,
            "uses_neural_weights": True,
            "uses_backpropagation_for_inference": False,
            "trained_with_backpropagation": True,
            "model_state_size_bytes": model_state_size,
            "model_state_size_note": model_state_note,
            "nanogpt_dir": str(nanogpt_dir),
            "nanogpt_source_size_bytes": nanogpt_source_size,
            "architecture_note": "GPT-2 124M parameter transformer loaded through local nanoGPT for inference.",
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Microworld explicit state with GPT-2 model size.")
    parser.add_argument("--nanogpt-dir", default="nanogpt")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_summary(args.nanogpt_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
