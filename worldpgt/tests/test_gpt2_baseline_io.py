from __future__ import annotations

import csv
import sys

from worldpgt.baselines.gpt2.parse_outputs import (
    GPT2_OUTPUT_FIELDS,
    extract_completion,
    load_prompt_rows,
    write_gpt2_rows,
)
from worldpgt.baselines.gpt2.run_gpt2_baseline import (
    build_arg_parser,
    resolve_nanogpt_dir,
)


PROMPT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]


def test_extract_completion_removes_prompt_prefix():
    completion = extract_completion("The bank", "The bank opened at noon")
    assert completion == "opened at noon"


def test_extract_completion_handles_generated_text_without_prompt_prefix():
    completion = extract_completion("The bank", " opened at noon")
    assert completion == "opened at noon"


def test_load_prompt_rows_reads_required_columns(tmp_path):
    path = tmp_path / "prompts.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "prompt": "The bank",
                "ambiguous_term": "bank",
                "expected_sense": "financial_institution",
                "difficulty_type": "cue_rich",
                "notes": "test",
            }
        )

    rows = load_prompt_rows(str(path))

    assert len(rows) == 1
    assert rows[0]["prompt"] == "The bank"


def test_write_gpt2_rows_writes_expected_columns(tmp_path):
    path = tmp_path / "outputs.csv"
    write_gpt2_rows(
        str(path),
        [
            {
                "id": "1",
                "prompt": "The bank",
                "ambiguous_term": "bank",
                "expected_sense": "financial_institution",
                "difficulty_type": "cue_rich",
                "notes": "test",
                "model": "nanogpt:gpt2",
                "sample_index": 0,
                "completion": "opened",
                "full_text": "The bank opened",
                "generation_time_sec": "0.1000",
                "device": "cpu",
                "max_new_tokens": 32,
                "temperature": 0.8,
                "top_k": 40,
                "seed": 1337,
            }
        ],
    )

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == GPT2_OUTPUT_FIELDS
    assert len(rows) == 1
    assert rows[0]["completion"] == "opened"


def test_default_nanogpt_path_resolver_prefers_nanogpt_then_supports_nanoGPT(tmp_path):
    nano_lower = tmp_path / "nanogpt"
    nano_lower.mkdir()
    (nano_lower / "model.py").write_text("# test model\n", encoding="utf-8")
    nano_upper = tmp_path / "nanoGPT"
    nano_upper.mkdir(exist_ok=True)
    (nano_upper / "model.py").write_text("# test model\n", encoding="utf-8")

    resolved = resolve_nanogpt_dir(repo_root=tmp_path)

    assert resolved == nano_lower


def test_explicit_nanogpt_path_resolver(tmp_path):
    nano = tmp_path / "nanogpt"
    nano.mkdir()
    (nano / "model.py").write_text("# test model\n", encoding="utf-8")

    resolved = resolve_nanogpt_dir("nanogpt", repo_root=tmp_path)

    assert resolved == nano


def test_cli_parser_constructs_without_importing_torch():
    sys.modules.pop("torch", None)

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "in.csv",
            "--output",
            "out.csv",
        ]
    )

    assert args.input == "in.csv"
    assert args.output == "out.csv"
    assert args.device == "cpu"
    assert "torch" not in sys.modules
