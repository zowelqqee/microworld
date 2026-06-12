"""IO helpers for GPT-2 continuation baseline outputs."""

from __future__ import annotations

import csv


PROMPT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]

GPT2_OUTPUT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
    "model",
    "sample_index",
    "completion",
    "full_text",
    "generation_time_sec",
    "device",
    "max_new_tokens",
    "temperature",
    "top_k",
    "seed",
]


def extract_completion(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].lstrip()
    return generated_text.lstrip()


def load_prompt_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in PROMPT_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required prompt columns: {missing}")
        return list(reader)


def write_gpt2_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPT2_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
