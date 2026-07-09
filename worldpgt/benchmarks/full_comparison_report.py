"""Build a JSON and Markdown Microworld-vs-GPT-2 benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NON_CLAIMS = [
    "No claim that Microworld beats neural networks generally.",
    "No claim that this small benchmark predicts open-domain generation quality.",
    "No claim that GPT-2 represents modern instruction-tuned assistants.",
]

CAVEATS = [
    "This is a small 120-row controlled continuation benchmark.",
    "GPT-2 is an old base model, not ChatGPT or an instruction-tuned model.",
    "GPT-2 has no native audit path; labels were audited after generation.",
    "Microworld has low coverage and template-based realization.",
    "RSS is approximate and environment-dependent.",
]


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pick(rows: list[dict], predicate, limit: int) -> list[dict]:
    picked = []
    for row in rows:
        if predicate(row):
            picked.append(row)
            if len(picked) == limit:
                break
    return picked


def _compact_microworld(row: dict) -> dict:
    return {
        "id": row.get("id", ""),
        "prompt": row.get("prompt", ""),
        "expected_sense": row.get("expected_sense", ""),
        "selected_sense": row.get("selected_sense", ""),
        "decision": row.get("decision", ""),
        "continuation": row.get("continuation", ""),
        "difficulty_type": row.get("difficulty_type", ""),
        "reasons": row.get("reasons", ""),
    }


def _compact_gpt2(row: dict) -> dict:
    return {
        "id": row.get("id", ""),
        "prompt": row.get("prompt", ""),
        "expected_sense": row.get("expected_sense", ""),
        "judged_sense": row.get("judged_sense", ""),
        "label": row.get("label", ""),
        "judged_text": row.get("judged_text", ""),
        "difficulty_type": row.get("difficulty_type", ""),
        "audit_notes": row.get("audit_notes", ""),
    }


def build_report(
    microworld_output_path: str | Path,
    microworld_risk_path: str | Path,
    gpt2_audit_path: str | Path,
    gpt2_summary_path: str | Path,
    runtime_path: str | Path,
    state_size_path: str | Path,
    rss_path: str | Path,
) -> dict:
    microworld_rows = _read_csv(microworld_output_path)
    gpt2_rows = _read_csv(gpt2_audit_path)
    microworld_risk = _read_json(microworld_risk_path)
    gpt2_summary = _read_json(gpt2_summary_path)
    runtime = _read_json(runtime_path)
    state_size = _read_json(state_size_path)
    rss = _read_json(rss_path)

    examples = {
        "microworld_continue_examples": [
            _compact_microworld(row)
            for row in _pick(microworld_rows, lambda r: r.get("decision") == "continue", 5)
        ],
        "microworld_audit_examples": [
            _compact_microworld(row)
            for row in _pick(microworld_rows, lambda r: r.get("decision") == "audit", 5)
        ],
        "gpt2_good_examples": [
            _compact_gpt2(row) for row in _pick(gpt2_rows, lambda r: r.get("label") == "good", 5)
        ],
        "gpt2_bad_unclear_examples": [
            _compact_gpt2(row)
            for row in _pick(gpt2_rows, lambda r: r.get("label") in {"bad", "unclear"}, 5)
        ],
    }

    return {
        "benchmark_scope": {
            "task": "Controlled continuation / ambiguity resolution.",
            "scope_note": "This compares explicit audit-aware continuation with GPT-2 next-token generation on the v1 prompt set.",
        },
        "dataset": {
            "name": "Controlled Continuation v1",
            "rows": microworld_risk.get("total"),
            "expected_answerable_count": microworld_risk.get("expected_answerable_count"),
            "expected_no_answer_count": microworld_risk.get("expected_no_answer_count"),
        },
        "microworld_architecture": {
            "summary": "Explicit sense memory, deterministic cue scoring, anti-cues/guards, conservative policy, template realization, and surface-risk audit gate.",
            "native_audit_path": True,
            "uses_neural_weights": False,
        },
        "gpt2_baseline_architecture": {
            "summary": "GPT-2 base model inference through local nanoGPT; no training or fine-tuning performed here.",
            "native_audit_path": False,
            "uses_neural_weights": True,
        },
        "architecture_comparison": {
            "microworld": "Small explicit state machine with inspectable evidence, thresholds, anti-cues, guards, and audit/suppress decisions.",
            "gpt2": "Large pretrained transformer that samples next tokens from neural weights and does not expose a native selected-sense or audit decision.",
            "comparison_note": "Microworld is designed for abstention and auditability; GPT-2 is designed for open-ended language modeling.",
        },
        "quality_comparison": {
            "microworld": {
                "correct_continue_count": microworld_risk.get("correct_continue_count"),
                "wrong_continue_count": microworld_risk.get("wrong_continue_count"),
                "precision_on_continued": microworld_risk.get("precision_on_continued"),
            },
            "gpt2_audited": {
                "good": gpt2_summary.get("good"),
                "bad": gpt2_summary.get("bad"),
                "unclear": gpt2_summary.get("unclear"),
                "precision": gpt2_summary.get("precision"),
                "correct_sense_count": gpt2_summary.get("correct_sense_count"),
                "wrong_sense_count": gpt2_summary.get("wrong_sense_count"),
            },
        },
        "risk_coverage_comparison": {
            "microworld": {
                "continue_count": microworld_risk.get("continue_count"),
                "audit_count": microworld_risk.get("audit_count"),
                "coverage_rate": microworld_risk.get("coverage_rate"),
                "answerable_recall": microworld_risk.get("answerable_recall"),
                "wrong_continue_rate": microworld_risk.get("wrong_continue_rate"),
            },
            "gpt2": {
                "native_audit_path": False,
                "generated_rows": gpt2_summary.get("total"),
                "bad_rate_after_audit": gpt2_summary.get("bad_rate"),
                "unclear_rate_after_audit": gpt2_summary.get("unclear_rate"),
            },
        },
        "runtime_comparison": runtime,
        "memory_rss_comparison": rss,
        "state_model_size_comparison": state_size,
        "examples": examples,
        "limitations": CAVEATS,
        "non_claims": NON_CLAIMS,
        "headline": {
            "summary": "Microworld emits fewer continuations but had zero measured wrong continuations among emitted rows in this benchmark; GPT-2 produced more usable continuations overall but also bad/unclear continuations and has no native audit path.",
            "risk_coverage_tradeoff": "The result exposes a risk/coverage tradeoff between explicit policy continuation and open-ended next-token generation.",
        },
    }


def _format_metric(value: object) -> str:
    return "null" if value is None else str(value)


def render_markdown(report: dict) -> str:
    mw_quality = report["quality_comparison"]["microworld"]
    gpt_quality = report["quality_comparison"]["gpt2_audited"]
    mw_risk = report["risk_coverage_comparison"]["microworld"]
    runtime = report["runtime_comparison"]
    rss = report["memory_rss_comparison"]
    state = report["state_model_size_comparison"]
    architecture_note = report.get("architecture_comparison", {}).get(
        "comparison_note",
        "Microworld is audit-oriented explicit policy; GPT-2 is open-ended language modeling.",
    )
    lines = [
        "# Microworld vs GPT-2 Controlled Continuation Report",
        "",
        "## Benchmark Scope",
        report["benchmark_scope"]["task"],
        "",
        report["benchmark_scope"]["scope_note"],
        "",
        "## Dataset",
        f"- Rows: {_format_metric(report['dataset']['rows'])}",
        f"- Expected answerable: {_format_metric(report['dataset']['expected_answerable_count'])}",
        f"- Expected no-answer: {_format_metric(report['dataset']['expected_no_answer_count'])}",
        "",
        "## Architecture",
        f"- Microworld: {report['microworld_architecture']['summary']}",
        f"- GPT-2: {report['gpt2_baseline_architecture']['summary']}",
        f"- Comparison: {architecture_note}",
        "",
        "## Quality Comparison",
        f"- Microworld correct continuations: {_format_metric(mw_quality['correct_continue_count'])}",
        f"- Microworld wrong continuations: {_format_metric(mw_quality['wrong_continue_count'])}",
        f"- Microworld precision on continued: {_format_metric(mw_quality['precision_on_continued'])}",
        f"- GPT-2 good/bad/unclear: {_format_metric(gpt_quality['good'])} / {_format_metric(gpt_quality['bad'])} / {_format_metric(gpt_quality['unclear'])}",
        f"- GPT-2 audited precision: {_format_metric(gpt_quality['precision'])}",
        f"- GPT-2 wrong sense count: {_format_metric(gpt_quality['wrong_sense_count'])}",
        "",
        "## Risk/Coverage Comparison",
        f"- Microworld continue/audit: {_format_metric(mw_risk['continue_count'])} / {_format_metric(mw_risk['audit_count'])}",
        f"- Microworld coverage rate: {_format_metric(mw_risk['coverage_rate'])}",
        f"- Microworld answerable recall: {_format_metric(mw_risk['answerable_recall'])}",
        "- GPT-2 has no native audit path; GPT-2 quality labels were assigned after generation.",
        "",
        "## Runtime Comparison",
        f"- Microworld timed run total sec: {_format_metric(runtime.get('microworld', {}).get('timed_run', {}).get('total_time_sec'))}",
        f"- Microworld avg sec/prompt: {_format_metric(runtime.get('microworld', {}).get('timed_run', {}).get('avg_time_sec_per_prompt'))}",
        f"- GPT-2 total generation sec from CSV: {_format_metric(runtime.get('gpt2', {}).get('existing_output', {}).get('total_generation_time_sec'))}",
        f"- GPT-2 avg generation sec/prompt from CSV: {_format_metric(runtime.get('gpt2', {}).get('existing_output', {}).get('avg_generation_time_sec'))}",
        "",
        "## Memory/RSS Comparison",
        f"- Microworld peak RSS MB: {_format_metric(rss.get('microworld_peak_rss_mb'))}",
        f"- GPT-2 peak RSS MB: {_format_metric(rss.get('gpt2_peak_rss_mb'))}",
        "",
        "## State/Model Size Comparison",
        f"- Microworld explicit state bytes: {_format_metric(state.get('microworld', {}).get('explicit_state_size_bytes'))}",
        f"- GPT-2 parameter count: {_format_metric(state.get('gpt2', {}).get('trainable_parameter_count'))}",
        f"- GPT-2 model state size bytes: {_format_metric(state.get('gpt2', {}).get('model_state_size_bytes'))}",
        "",
        "## Examples",
    ]
    for section_name, rows in report["examples"].items():
        lines.append(f"### {section_name.replace('_', ' ').title()}")
        for row in rows:
            text = row.get("continuation") or row.get("judged_text") or ""
            lines.append(f"- `{row.get('id')}` {row.get('prompt')} -> {text}")
        lines.append("")

    lines.extend(["## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## Non-Claims"])
    lines.extend(f"- {item}" for item in report["non_claims"])
    lines.extend(["", "## Framing", report["headline"]["summary"], "", report["headline"]["risk_coverage_tradeoff"], ""])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full Microworld-vs-GPT-2 report.")
    parser.add_argument("--microworld-output", required=True)
    parser.add_argument("--microworld-risk", required=True)
    parser.add_argument("--gpt2-audit", required=True)
    parser.add_argument("--gpt2-summary", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--state-size", required=True)
    parser.add_argument("--rss", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        args.microworld_output,
        args.microworld_risk,
        args.gpt2_audit,
        args.gpt2_summary,
        args.runtime,
        args.state_size,
        args.rss,
    )
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_output": str(json_path), "md_output": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
