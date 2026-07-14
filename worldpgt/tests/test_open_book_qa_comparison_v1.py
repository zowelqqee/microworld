"""Focused offline tests for the reproducible open-book benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa import dataset as ds
from worldpgt.benchmarks.open_book_qa.evaluate import evaluate, percentile
from worldpgt.benchmarks.open_book_qa.qwen_runner import prompt_for, run as run_qwen


def _relation(subject="Graph system", predicate="uses", obj="edge labels"):
    return {"overlay_type": "overlay_relation", "subject": subject, "predicate": predicate, "object": obj,
            "evidence_text": f"{subject} {predicate} {obj}.", "source_url": "https://example.test/source",
            "experimental_tier": "evidence_grounded_abstract_relation_v1"}


def test_dataset_is_deterministic_and_excludes_deictic(monkeypatch):
    rows = [_relation(), _relation("Our technique", obj="edge labels"), *[_relation(f"System {i}", obj=f"object {i}") for i in range(200)]]
    # Give half the synthetic entries a second fact so multi-evidence can form.
    rows.extend(_relation(f"System {i}", "enables", f"capability {i}") for i in range(60))
    monkeypatch.setattr(ds, "load_experimental_relations", lambda overlay: rows)
    one = ds.build_dataset(seed=7); two = ds.build_dataset(seed=7)
    assert one[0] == two[0]
    assert one[2]["total_cases"] == 250
    assert all("our technique" not in case["expected_subject"].casefold() for case in one[0])
    assert any(item["reason"] == "deictic_node" for item in one[1])


def test_direct_questions_and_negative_contexts_are_evidence_scoped(monkeypatch):
    rows = [_relation(f"System {i}", "uses", f"object {i}") for i in range(200)]
    rows.extend(_relation(f"System {i}", "enables", f"capability {i}") for i in range(60))
    monkeypatch.setattr(ds, "load_experimental_relations", lambda overlay: rows)
    cases, _, _ = ds.build_dataset()
    for case in cases:
        assert "expected_objects" not in prompt_for({"question": case["question"], "contexts": case["contexts"]})
        if case["category"] == "direct": assert case["expected_predicate"]
        if case["category"] == "negative":
            assert case["expected_objects"] == []
            assert case["expected_predicate"][0].replace("_", " ") not in " ".join(case["contexts"]).casefold()


def test_qwen_prompt_and_exact_unknown_with_mocked_model():
    case = {"id": "one", "question": "What does Graph use?", "contexts": ["Graph uses labels."], "category": "direct"}
    prompt = prompt_for(case)
    assert "Graph uses labels." in prompt and "expected" not in prompt.casefold()
    class Tokenizer:
        def encode(self, text): return text.split()
    def fake_load(_): return object(), Tokenizer(), lambda *args, **kwargs: "UNKNOWN", object()
    import worldpgt.benchmarks.open_book_qa.qwen_runner as runner
    old = runner._load; runner._load = fake_load
    try: results, _ = run_qwen([case], warmups=1, repeats=1)
    finally: runner._load = old
    assert results[0]["exact_unknown"] is True and set(results[0]) >= {"answer", "total_latency_ms", "exception"}


def test_metrics_percentiles_and_provenance_schema():
    case = {"id": "one", "category": "direct", "contexts": ["Graph uses labels."], "expected_objects": ["labels"], "expected_decision": "answer", "relation_ids": ["edge:graph|uses|labels"]}
    mw = [{"id": "one", "answer": "Graph uses labels.", "decision": "answer", "selected_relation_ids": ["edge:graph|uses|labels"], "total_latency_ms": 2.0}]
    qw = [{"id": "one", "answer": "labels", "exact_unknown": False, "total_latency_ms": 5.0, "ttft_ms": None, "tokens_per_second": 10.0}]
    summary, rows = evaluate([case], mw, qw)
    assert len(rows) == 2 and rows[0]["exact_evidence_provenance_accuracy"] == 1.0
    assert percentile([1, 2, 3], .95) == 2.9 and "limitation" in summary["methodology"]


def test_plotting_smoke_without_gui(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from worldpgt.benchmarks.plot_open_book_qa_comparison import main
    source = tmp_path / "summary.json"; source.write_text(json.dumps({"rows": [{"system": "MicroWorld explicit graph runtime", "category": "direct", "cases": 1, "latency_p50_ms": 1, "latency_p95_ms": 2, "latency_p99_ms": 3, "answer_accuracy": 1, "negative_accuracy": 0, "unsupported_claim_rate": 0, "predicate_adherence": 1, "artifact_size_mib": 0, "startup_ms": 0, "extra_memory_mib": 0, "ttft_p50_ms": None}]}))
    assert main(["--summary", str(source), "--output", str(tmp_path / "figures")]) == 0
    assert (tmp_path / "figures" / "complete_latency.png").is_file()


def test_cli_smoke_with_mocked_qwen_runner(tmp_path, monkeypatch):
    from worldpgt.benchmarks.open_book_qa import cli
    dataset = tmp_path / "dataset.jsonl"; dataset.write_text("{}\n")
    called = {}
    monkeypatch.setattr(cli, "run_qwen", lambda *args, **kwargs: called.update(args=args, kwargs=kwargs) or {})
    assert cli.main(["run-qwen", "--dataset", str(dataset), "--output", str(tmp_path)]) == 0
    assert called["args"][0] == str(dataset)
