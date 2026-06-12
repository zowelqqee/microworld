import csv
import json
import sys

from worldpgt.benchmarks.full_comparison_report import build_report, render_markdown
from worldpgt.benchmarks.runtime_benchmark import summarize_generation_times
from worldpgt.benchmarks.state_size_benchmark import build_summary


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_runtime_summary_from_synthetic_csv_rows():
    summary = summarize_generation_times(
        [
            {"generation_time_sec": "0.1"},
            {"generation_time_sec": "0.3"},
            {"generation_time_sec": "0.2"},
        ]
    )

    assert summary["total_rows"] == 3
    assert summary["total_generation_time_sec"] == 0.6
    assert summary["avg_generation_time_sec"] == 0.2
    assert summary["median_generation_time_sec"] == 0.2
    assert summary["min_generation_time_sec"] == 0.1
    assert summary["max_generation_time_sec"] == 0.3


def test_state_size_summary_contains_required_fields(tmp_path):
    nanogpt_dir = tmp_path / "nanogpt"
    nanogpt_dir.mkdir()

    summary = build_summary(nanogpt_dir)

    assert summary["microworld"]["explicit_state_size_bytes"] > 0
    assert summary["microworld"]["trainable_parameter_count"] == 0
    assert summary["microworld"]["uses_neural_weights"] is False
    assert summary["gpt2"]["model_name"] == "gpt2"
    assert summary["gpt2"]["trainable_parameter_count"] == 123650000
    assert summary["gpt2"]["uses_neural_weights"] is True
    assert "architecture_note" in summary["gpt2"]


def test_full_report_json_contains_required_top_level_sections(tmp_path):
    mw_output = tmp_path / "mw.csv"
    gpt2_audit = tmp_path / "gpt2.csv"
    mw_risk = tmp_path / "mw_risk.json"
    gpt2_summary = tmp_path / "gpt2_summary.json"
    runtime = tmp_path / "runtime.json"
    state = tmp_path / "state.json"
    rss = tmp_path / "rss.json"

    _write_csv(
        mw_output,
        [
            "id",
            "prompt",
            "expected_sense",
            "selected_sense",
            "decision",
            "continuation",
            "difficulty_type",
            "reasons",
        ],
        [
            {
                "id": "1",
                "prompt": "The bank teller",
                "expected_sense": "financial_institution",
                "selected_sense": "financial_institution",
                "decision": "continue",
                "continuation": "helped the customer",
                "difficulty_type": "cue_rich",
                "reasons": "score",
            },
            {
                "id": "2",
                "prompt": "He went to the bank",
                "expected_sense": "",
                "selected_sense": "",
                "decision": "audit",
                "continuation": "",
                "difficulty_type": "no_clear_answer",
                "reasons": "all_sense_scores_zero",
            },
        ],
    )
    _write_csv(
        gpt2_audit,
        ["id", "prompt", "expected_sense", "judged_sense", "label", "judged_text", "difficulty_type", "audit_notes"],
        [
            {
                "id": "1",
                "prompt": "The bank teller",
                "expected_sense": "financial_institution",
                "judged_sense": "financial_institution",
                "label": "good",
                "judged_text": "helped.",
                "difficulty_type": "cue_rich",
                "audit_notes": "",
            },
            {
                "id": "2",
                "prompt": "He went to the bank",
                "expected_sense": "",
                "judged_sense": "river_edge",
                "label": "bad",
                "judged_text": "by the river.",
                "difficulty_type": "no_clear_answer",
                "audit_notes": "",
            },
        ],
    )
    mw_risk.write_text(
        json.dumps(
            {
                "total": 2,
                "expected_answerable_count": 1,
                "expected_no_answer_count": 1,
                "continue_count": 1,
                "audit_count": 1,
                "coverage_rate": 0.5,
                "answerable_recall": 1.0,
                "correct_continue_count": 1,
                "wrong_continue_count": 0,
                "wrong_continue_rate": 0.0,
                "precision_on_continued": 1.0,
            }
        ),
        encoding="utf-8",
    )
    gpt2_summary.write_text(
        json.dumps(
            {
                "total": 2,
                "good": 1,
                "bad": 1,
                "unclear": 0,
                "precision": 0.5,
                "correct_sense_count": 1,
                "wrong_sense_count": 1,
                "bad_rate": 0.5,
                "unclear_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            {
                "microworld": {"timed_run": {"total_time_sec": 0.01, "avg_time_sec_per_prompt": 0.005}},
                "gpt2": {"existing_output": {"total_generation_time_sec": 2.0, "avg_generation_time_sec": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps(
            {
                "microworld": {"explicit_state_size_bytes": 100, "trainable_parameter_count": 0},
                "gpt2": {"trainable_parameter_count": 123650000, "model_state_size_bytes": None},
            }
        ),
        encoding="utf-8",
    )
    rss.write_text(
        json.dumps({"microworld_peak_rss_mb": 10.0, "gpt2_peak_rss_mb": 1000.0}),
        encoding="utf-8",
    )

    report = build_report(mw_output, mw_risk, gpt2_audit, gpt2_summary, runtime, state, rss)

    for section in [
        "benchmark_scope",
        "dataset",
        "microworld_architecture",
        "gpt2_baseline_architecture",
        "quality_comparison",
        "risk_coverage_comparison",
        "runtime_comparison",
        "memory_rss_comparison",
        "state_model_size_comparison",
        "examples",
        "limitations",
        "non_claims",
    ]:
        assert section in report


def test_markdown_report_includes_caveats_and_non_claims(tmp_path):
    report = {
        "benchmark_scope": {"task": "Controlled continuation", "scope_note": "Small comparison."},
        "dataset": {"rows": 120, "expected_answerable_count": 110, "expected_no_answer_count": 10},
        "microworld_architecture": {"summary": "Explicit policy."},
        "gpt2_baseline_architecture": {"summary": "GPT-2 base."},
        "quality_comparison": {
            "microworld": {"correct_continue_count": 38, "wrong_continue_count": 0, "precision_on_continued": 1.0},
            "gpt2_audited": {"good": 76, "bad": 11, "unclear": 33, "precision": 0.8736, "wrong_sense_count": 7},
        },
        "risk_coverage_comparison": {
            "microworld": {"continue_count": 38, "audit_count": 82, "coverage_rate": 0.3167, "answerable_recall": 0.3455}
        },
        "runtime_comparison": {
            "microworld": {"timed_run": {"total_time_sec": 0.01, "avg_time_sec_per_prompt": 0.001}},
            "gpt2": {"existing_output": {"total_generation_time_sec": 1.0, "avg_generation_time_sec": 0.1}},
        },
        "memory_rss_comparison": {"microworld_peak_rss_mb": 10, "gpt2_peak_rss_mb": 1000},
        "state_model_size_comparison": {
            "microworld": {"explicit_state_size_bytes": 100},
            "gpt2": {"trainable_parameter_count": 123650000, "model_state_size_bytes": None},
        },
        "examples": {
            "microworld_continue_examples": [],
            "microworld_audit_examples": [],
            "gpt2_good_examples": [],
            "gpt2_bad_unclear_examples": [],
        },
        "limitations": [
            "This is a small 120-row controlled continuation benchmark.",
            "GPT-2 has no native audit path; labels were audited after generation.",
        ],
        "non_claims": ["No claim that Microworld beats neural networks generally."],
        "headline": {"summary": "Honest comparison.", "risk_coverage_tradeoff": "Risk/coverage tradeoff."},
    }

    markdown = render_markdown(report)

    assert "small 120-row controlled continuation benchmark" in markdown
    assert "GPT-2 has no native audit path" in markdown
    assert "No claim that Microworld beats neural networks generally" in markdown


def test_benchmark_imports_do_not_import_torch_tiktoken_or_transformers():
    for module_name in ["torch", "tiktoken", "transformers"]:
        sys.modules.pop(module_name, None)

    __import__("worldpgt.benchmarks.runtime_benchmark")
    __import__("worldpgt.benchmarks.state_size_benchmark")
    __import__("worldpgt.benchmarks.full_comparison_report")
    __import__("worldpgt.benchmarks.rss_benchmark")

    assert "torch" not in sys.modules
    assert "tiktoken" not in sys.modules
    assert "transformers" not in sys.modules


def test_rss_script_can_be_imported_without_running_gpt2():
    import worldpgt.benchmarks.rss_benchmark as rss_benchmark

    assert rss_benchmark.build_arg_parser() is not None

