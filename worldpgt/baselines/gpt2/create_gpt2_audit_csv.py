"""Create an audit-ready CSV from GPT-2 continuation baseline outputs.

Usage:
    python3 -m worldpgt.baselines.gpt2.create_gpt2_audit_csv \
        --input worldpgt/experiments/gpt2_continuation_outputs.csv \
        --output worldpgt/experiments/gpt2_continuation_audit.csv
"""

from __future__ import annotations

import argparse
import csv


AUDIT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
    "model",
    "completion",
    "full_text",
    "judged_text",
    "judged_sense",
    "label",
    "audit_notes",
]


def make_judged_text(completion: str) -> str:
    text = (completion or "").strip()
    if "\n" in text:
        return text.split("\n", 1)[0].strip()

    sentence_ends = [
        index
        for marker in (".", "!", "?")
        if (index := text.find(marker)) != -1
    ]
    if sentence_ends:
        end = min(sentence_ends) + 1
        return text[:end].strip()
    return text


def create_audit_rows(input_rows: list[dict]) -> list[dict]:
    audit_rows = []
    for row in input_rows:
        audit_rows.append(
            {
                "id": row.get("id", ""),
                "prompt": row.get("prompt", ""),
                "ambiguous_term": row.get("ambiguous_term", ""),
                "expected_sense": row.get("expected_sense", ""),
                "difficulty_type": row.get("difficulty_type", ""),
                "notes": row.get("notes", ""),
                "model": row.get("model", ""),
                "completion": row.get("completion", ""),
                "full_text": row.get("full_text", ""),
                "judged_text": make_judged_text(row.get("completion", "")),
                "judged_sense": "",
                "label": "",
                "audit_notes": "",
            }
        )
    return audit_rows


def read_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_audit_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def create_audit_csv(input_path: str, output_path: str) -> list[dict]:
    rows = create_audit_rows(read_rows(input_path))
    write_audit_rows(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GPT-2 continuation audit CSV.")
    parser.add_argument("--input", required=True, help="Input GPT-2 continuation output CSV")
    parser.add_argument("--output", required=True, help="Output audit CSV")
    args = parser.parse_args()

    rows = create_audit_csv(args.input, args.output)
    print(f"Wrote {len(rows)} GPT-2 audit rows to {args.output}")


if __name__ == "__main__":
    main()
