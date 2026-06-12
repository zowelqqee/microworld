from __future__ import annotations

import csv
import sys

from worldpgt.baselines.gpt2.create_gpt2_audit_csv import (
    AUDIT_FIELDS,
    create_audit_csv,
    create_audit_rows,
    make_judged_text,
)


def test_judged_text_cuts_at_newline():
    text = make_judged_text("first line\nsecond line.")
    assert text == "first line"


def test_judged_text_cuts_at_first_sentence():
    text = make_judged_text("First sentence. Second sentence.")
    assert text == "First sentence."


def test_judged_text_keeps_text_without_punctuation():
    text = make_judged_text("fragment without ending punctuation")
    assert text == "fragment without ending punctuation"


def test_audit_csv_preserves_all_input_rows(tmp_path):
    input_path = tmp_path / "gpt2.csv"
    output_path = tmp_path / "audit.csv"
    _write_gpt2_input(
        input_path,
        [
            _gpt2_row("1", "completion one."),
            _gpt2_row("2", "completion two."),
        ],
    )

    rows = create_audit_csv(str(input_path), str(output_path))

    assert len(rows) == 2
    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert [row["id"] for row in written] == ["1", "2"]


def test_audit_csv_has_required_columns(tmp_path):
    input_path = tmp_path / "gpt2.csv"
    output_path = tmp_path / "audit.csv"
    _write_gpt2_input(input_path, [_gpt2_row("1", "completion.")])

    create_audit_csv(str(input_path), str(output_path))

    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == AUDIT_FIELDS
    assert rows[0]["judged_sense"] == ""
    assert rows[0]["label"] == ""
    assert rows[0]["audit_notes"] == ""


def test_create_audit_rows_does_not_infer_labels():
    rows = create_audit_rows([_gpt2_row("1", "The teller opened an account.")])
    assert rows[0]["judged_sense"] == ""
    assert rows[0]["label"] == ""
    assert rows[0]["audit_notes"] == ""


def test_no_torch_tiktoken_transformers_imports_required():
    for name in ["torch", "tiktoken", "transformers"]:
        sys.modules.pop(name, None)

    make_judged_text("A sentence.")

    for name in ["torch", "tiktoken", "transformers"]:
        assert name not in sys.modules


def _write_gpt2_input(path, rows: list[dict]) -> None:
    fields = [
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gpt2_row(row_id: str, completion: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bank",
        "expected_sense": "financial_institution",
        "difficulty_type": "cue_rich",
        "notes": "test",
        "model": "nanogpt:gpt2",
        "sample_index": "0",
        "completion": completion,
        "full_text": f"prompt {row_id} {completion}",
        "generation_time_sec": "0.1",
        "device": "cpu",
        "max_new_tokens": "32",
        "temperature": "0.8",
        "top_k": "40",
        "seed": "1337",
    }
