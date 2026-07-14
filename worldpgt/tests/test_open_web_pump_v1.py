from __future__ import annotations

from pathlib import Path

from worldpgt.knowledge_pump.open_web_pump import (
    BROAD_OPEN_WEB_TOPICS,
    OpenWebTopic,
    build_query_plan,
    build_paged_query_plan,
    collect_records,
    parse_arxiv,
    parse_openalex,
    _open_web_source_gate,
    build_exploratory_relation_overlay,
    build_evidence_grounded_experimental_graph,
    extract_open_web_abstract_definitions,
    write_snapshot_artifacts,
    build_proposal_overlay,
    consolidate_regated_campaign,
    run_open_web_pump,
)
from worldpgt.knowledge_pump.open_web_campaign import run_open_web_campaign


def test_broad_plan_covers_diverse_domains_and_curated_sources():
    plan = build_query_plan()
    buckets = {topic.bucket for topic, _source in plan}
    sources = {source for _topic, source in plan}

    assert {"history", "health", "mathematics", "arts_culture", "society"} <= buckets
    assert sources == {"openalex", "crossref", "arxiv"}
    assert len(plan) > len(BROAD_OPEN_WEB_TOPICS)


def test_openalex_parser_reconstructs_inverted_abstract_with_provenance():
    topic = OpenWebTopic("test topic", "testing", ("openalex",))
    payload = {
        "results": [{
            "id": "https://openalex.org/W1",
            "display_name": "Test Work",
            "publication_year": 2024,
            "doi": "https://doi.org/10.1/test",
            "abstract_inverted_index": {"Test": [0], "abstract": [1]},
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {"landing_page_url": "https://doi.org/10.1/test"},
        }],
    }

    records = parse_openalex(payload, topic, "2026-07-12T00:00:00Z")

    assert len(records) == 1
    assert records[0].text == "Test abstract"
    assert records[0].source_url == "https://doi.org/10.1/test"
    assert records[0].topic_bucket == "testing"
    assert records[0].authors == ("Ada Lovelace",)


def test_arxiv_parser_reads_abstract_and_author():
    topic = OpenWebTopic("test topic", "testing", ("arxiv",))
    xml = """<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>http://arxiv.org/abs/1234.5678</id><title>Test Paper</title>
      <summary>A concise abstract.</summary><published>2025-01-01T00:00:00Z</published>
      <author><name>Grace Hopper</name></author></entry></feed>"""

    records = parse_arxiv(xml, topic, "2026-07-12T00:00:00Z")

    assert len(records) == 1
    assert records[0].source_kind == "arxiv"
    assert records[0].authors == ("Grace Hopper",)
    assert records[0].text == "A concise abstract."


def test_no_network_mode_writes_a_plan_only(tmp_path: Path):
    result = run_open_web_pump(output_dir=tmp_path, max_queries=3)

    assert result["collection"]["status"] == "planned_no_network"
    assert result["accepted_memory_modified"] is False
    assert (tmp_path / "open_web_pump_plan.json").is_file()
    assert not (tmp_path / "open_web_proposal_overlay.json").exists()


def test_query_offset_selects_a_non_overlapping_resume_segment():
    plan = build_query_plan()
    first = collect_records(max_queries=3, allow_network=False)[1]
    resumed = collect_records(start_query=3, max_queries=3, allow_network=False)[1]

    assert first["planned_total"] == resumed["planned_total"]
    assert first["start_query"] == 0
    assert resumed["start_query"] == 3
    assert plan[:3] != plan[3:6]


def test_paged_plan_interleaves_subjects_and_uses_source_specific_page_offsets():
    topics = (
        OpenWebTopic("alpha", "testing", ("openalex",)),
        OpenWebTopic("beta", "testing", ("crossref",)),
        OpenWebTopic("gamma", "testing", ("arxiv",)),
    )
    urls: list[str] = []

    def get_json(url: str):
        urls.append(url)
        return {"results": []} if "openalex" in url else {"message": {"items": []}}

    def get_text(url: str):
        urls.append(url)
        return "<feed xmlns='http://www.w3.org/2005/Atom'/>"

    plan = build_paged_query_plan(topics, pages_per_query=2)
    _records, report = collect_records(
        topics=topics, pages_per_query=2, records_per_query=10, allow_network=True,
        request_delay_sec=0, get_json=get_json, get_text=get_text,
    )

    assert [(topic.query, source, page) for topic, source, page in plan] == [
        ("alpha", "openalex", 0), ("beta", "crossref", 0), ("gamma", "arxiv", 0),
        ("alpha", "openalex", 1), ("beta", "crossref", 1), ("gamma", "arxiv", 1),
    ]
    assert report["planned_total"] == 6
    assert "per-page=10&page=1" in urls[0]
    assert "rows=10&offset=0" in urls[1]
    assert "start=0&max_results=10" in urls[2]
    assert "per-page=10&page=2" in urls[3]
    assert "rows=10&offset=10" in urls[4]
    assert "start=10&max_results=10" in urls[5]


def test_page_start_extends_the_frontier_without_repeating_prior_pages():
    topics = (OpenWebTopic("alpha", "testing", ("openalex", "crossref", "arxiv")),)
    urls: list[str] = []

    def get_json(url: str):
        urls.append(url)
        return {"results": []} if "openalex" in url else {"message": {"items": []}}

    def get_text(url: str):
        urls.append(url)
        return "<feed xmlns='http://www.w3.org/2005/Atom'/>"

    plan = build_paged_query_plan(topics, pages_per_query=2, page_start=12)
    _records, report = collect_records(
        topics=topics, pages_per_query=2, page_start=12, records_per_query=10,
        allow_network=True, request_delay_sec=0, get_json=get_json, get_text=get_text,
    )

    assert [(source, page) for _topic, source, page in plan] == [
        ("openalex", 12), ("crossref", 12), ("arxiv", 12),
        ("openalex", 13), ("crossref", 13), ("arxiv", 13),
    ]
    assert report["page_start"] == 12
    assert "per-page=10&page=13" in urls[0]
    assert "rows=10&offset=120" in urls[1]
    assert "start=120&max_results=10" in urls[2]


def test_rate_limit_opens_a_source_local_circuit_breaker():
    topics = (
        OpenWebTopic("one", "testing", ("openalex",)),
        OpenWebTopic("two", "testing", ("openalex",)),
    )
    calls = []

    def get_json(url: str):
        calls.append(url)
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    _records, report = collect_records(
        topics=topics, allow_network=True, get_json=get_json, request_delay_sec=0,
    )

    assert len(calls) == 1
    assert report["rate_limited_sources"] == ["openalex"]


def test_skip_source_removes_it_from_the_network_plan():
    _records, report = collect_records(max_queries=6, allow_network=False, skip_sources=("openalex",))

    assert report["skipped_sources"] == ["openalex"]
    assert "openalex" not in report["planned_by_source"]


def test_open_web_gate_accepts_only_a_direct_title_led_definition():
    accepted = {
        "overlay_type": "overlay_definition", "subject": "Mizar Mathematical Library",
        "source_page": "Mizar Mathematical Library", "definition": "large corpus of formalised mathematical knowledge",
        "evidence_text": "The Mizar Mathematical Library (MML) is a large corpus of formalised mathematical knowledge.",
    }
    relation = {
        "overlay_type": "overlay_relation", "subject": "North", "predicate": "located_in", "object": "St",
        "source_page": "North", "evidence_text": "North is Professor at Washington University in St. Louis.",
    }

    result = _open_web_source_gate([accepted, relation])

    assert result["accepted"] == [accepted]
    assert result["quarantine"][0]["reason"] == "open_web_relation_requires_source_specific_extractor"


def test_exploratory_lane_keeps_typed_connections_separate_from_raw_candidates():
    rows = [
        {
            "overlay_type": "overlay_relation", "subject": "Graph neural networks", "predicate": "uses",
            "object": "message passing", "evidence_text": "Graph neural networks use message passing.",
            "source_url": "https://example.test/gnn",
        },
        {
            "overlay_type": "overlay_relation", "subject": "Message passing", "predicate": "enables",
            "object": "information flow", "evidence_text": "Message passing enables information flow.",
            "source_url": "https://example.test/message-passing",
        },
        {
            "overlay_type": "overlay_relation", "subject": "This paper", "predicate": "uses",
            "object": "a benchmark", "evidence_text": "This paper uses a benchmark.",
            "source_url": "https://example.test/paper",
        },
        {
            "overlay_type": "overlay_relation", "subject": "A handbook", "predicate": "provides",
            "object": "an overview", "evidence_text": "A handbook provides an overview.",
            "source_url": "https://example.test/handbook",
        },
    ]

    result = build_exploratory_relation_overlay(rows)

    assert len(result["raw"]) == 4
    assert [(row["subject"], row["predicate"], row["object"]) for row in result["selected"]] == [
        ("Graph neural networks", "uses", "message passing"),
        ("Message passing", "enables", "information flow"),
    ]
    assert all(row["risk"] == "high" and row["safe_for_general_runtime"] is False for row in result["selected"])


def test_evidence_grounding_replaces_authorial_paper_title_with_named_system():
    title = "A chemical language model for reticular materials design"
    edge = {
        "overlay_type": "overlay_relation",
        "subject": title,
        "predicate": "enables",
        "object": "inverse design in reticular chemistry",
        "source_page": title,
        "source_record_title": title,
        "source_url": "https://arxiv.org/abs/example",
        "evidence_text": (
            "Here, we introduce Nexerra-R1, a building-block chemical language model "
            "that enables inverse design in reticular chemistry."
        ),
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert [(row["subject"], row["predicate"], row["object"]) for row in graph["relations"]] == [
        ("Nexerra-R1", "enables", "inverse design in reticular chemistry"),
    ]
    assert graph["relations"][0]["evidence_grounding"]["method"] == "distinctive_evidence_surface"
    assert graph["relations"][0]["source_record_title"] == title
    assert graph["query_relations"] == graph["relations"]
    assert [(row["subject"], row["definition"]) for row in graph["definitions"]] == [
        ("Nexerra-R1", "building-block chemical language model"),
    ]


def test_evidence_grounding_rejects_title_fallback_when_evidence_has_no_named_subject():
    title = "A generic method for modelling a material"
    edge = {
        "overlay_type": "overlay_relation", "subject": title, "predicate": "enables",
        "object": "contribution of each slip system", "source_page": title,
        "source_record_title": title,
        "evidence_text": "These predictive results enable contribution of each slip system.",
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert graph["relations"] == []
    assert graph["rejected"][0]["reason"] == "no_explicit_evidence_local_subject"


def test_evidence_grounding_never_treats_a_year_as_a_named_subject():
    title = "Historical computing study"
    edge = {
        "overlay_type": "overlay_relation", "subject": title, "predicate": "runs_on",
        "object": "FreeBSD", "source_page": title, "source_record_title": title,
        "evidence_text": "In 2014, the website runs on FreeBSD.",
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert graph["relations"] == []


def test_evidence_quality_holds_a_dangling_object_in_review_not_ui_graph():
    title = "An AI study"
    edge = {
        "overlay_type": "overlay_relation", "subject": title, "predicate": "uses",
        "object": "them", "source_page": title, "source_record_title": title,
        "evidence_text": "AI uses them to accelerate the analysis.",
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert len(graph["relations"]) == 1
    assert graph["relations"][0]["evidence_quality"]["queryable"] is False
    assert graph["query_relations"] == []
    assert graph["review_relations"][0]["evidence_quality"]["issues"] == [
        "object_starts_as_discourse_fragment",
    ]


def test_evidence_alias_grounding_merges_acronym_subjects_from_explicit_parenthetical_proof():
    title = "Language model study"
    rows = [
        {
            "overlay_type": "overlay_relation", "subject": title, "predicate": "uses",
            "object": "attention mechanisms", "source_page": title, "source_record_title": title,
            "evidence_text": "Large Language Model (LLM) uses attention mechanisms.",
        },
        {
            "overlay_type": "overlay_relation", "subject": title, "predicate": "enables",
            "object": "few-shot learning", "source_page": title, "source_record_title": title,
            "evidence_text": "LLM enables few-shot learning.",
        },
    ]

    graph = build_evidence_grounded_experimental_graph(rows)

    assert {row["subject"] for row in graph["query_relations"]} == {"Large Language Model"}
    entity = next(row for row in graph["entities"] if row["label"] == "Large Language Model")
    assert entity["aliases"] == ["LLM"]


def test_evidence_grounding_rejects_discourse_pronoun_as_subject():
    edge = {
        "overlay_type": "overlay_relation", "subject": "A study", "predicate": "uses",
        "object": "multiple knowledge representations", "source_page": "A study",
        "source_record_title": "A study",
        "evidence_text": "It uses multiple knowledge representations.",
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert graph["relations"] == []
    assert graph["rejected"][0]["reason"] == "no_explicit_evidence_local_subject"


def test_evidence_grounding_rejects_discourse_led_phrase_as_subject():
    edge = {
        "overlay_type": "overlay_relation", "subject": "A study", "predicate": "uses",
        "object": "a normative juridical research type", "source_page": "A study",
        "source_record_title": "A study",
        "evidence_text": "This writing uses a normative juridical research type.",
    }

    graph = build_evidence_grounded_experimental_graph([edge])

    assert graph["relations"] == []


def test_abstract_extractor_accepts_direct_named_definition_only(tmp_path: Path):
    from worldpgt.knowledge_pump.open_web_pump import OpenWebRecord

    good = OpenWebRecord(
        source_id="arxiv:good", source_kind="arxiv", topic_bucket="mathematics",
        title="Licensing the Mizar Mathematical Library", source_url="https://arxiv.org/abs/1107.3212",
        retrieved_at="2026-07-13T00:00:00Z", text="The Mizar Mathematical Library (MML) is a large corpus of formalised mathematical knowledge.",
        license_note="test",
    )
    bad = OpenWebRecord(
        source_id="arxiv:bad", source_kind="arxiv", topic_bucket="computing",
        title="A computer science paper", source_url="https://arxiv.org/abs/bad",
        retrieved_at="2026-07-13T00:00:00Z", text="Peer review is the driving force of journal development.", license_note="test",
    )
    pronoun = OpenWebRecord(
        source_id="arxiv:pronoun", source_kind="arxiv", topic_bucket="history",
        title="A biography", source_url="https://arxiv.org/abs/pronoun",
        retrieved_at="2026-07-13T00:00:00Z", text="He is a former professor of history.", license_note="test",
    )
    one_word_concept = OpenWebRecord(
        source_id="arxiv:concept", source_kind="arxiv", topic_bucket="geography",
        title="Urbanization research", source_url="https://arxiv.org/abs/concept",
        retrieved_at="2026-07-13T00:00:00Z", text="Urbanization is a process of population concentration in cities.", license_note="test",
    )
    methodological = OpenWebRecord(
        source_id="arxiv:methodological", source_kind="arxiv", topic_bucket="geography",
        title="Urbanization theory", source_url="https://arxiv.org/abs/methodological",
        retrieved_at="2026-07-13T00:00:00Z", text="Urbanization is a process that can be studied both historically and philosophically.", license_note="test",
    )
    docs, _manifest = write_snapshot_artifacts([good, bad, pronoun, one_word_concept, methodological], tmp_path)

    candidates = extract_open_web_abstract_definitions(docs)
    proposal = build_proposal_overlay(docs)

    assert [(row["subject"], row["definition"]) for row in candidates] == [
        ("Mizar Mathematical Library", "large corpus of formalised mathematical knowledge"),
        ("Urbanization", "process of population concentration in cities"),
    ]
    assert [row["subject"] for row in proposal["proposal_overlay"]] == ["Mizar Mathematical Library", "Urbanization"]


def test_abstract_extractor_accepts_lowercase_term_words_but_not_authorial_prose(tmp_path: Path):
    from worldpgt.knowledge_pump.open_web_pump import OpenWebRecord

    computer_science = OpenWebRecord(
        source_id="arxiv:cs", source_kind="arxiv", topic_bucket="computing",
        title="Computing survey", source_url="https://arxiv.org/abs/cs",
        retrieved_at="2026-07-13T00:00:00Z", text="Computer science is the study of the phenomena surrounding computers.",
        license_note="test",
    )
    jaguar = OpenWebRecord(
        source_id="arxiv:jaguar", source_kind="arxiv", topic_bucket="computing",
        title="Electronic structure software", source_url="https://arxiv.org/abs/jaguar",
        retrieved_at="2026-07-13T00:00:00Z", text="Jaguar is an ab initio quantum chemical program.", license_note="test",
    )
    prose = OpenWebRecord(
        source_id="arxiv:prose", source_kind="arxiv", topic_bucket="computing",
        title="Epilogue", source_url="https://arxiv.org/abs/prose",
        retrieved_at="2026-07-13T00:00:00Z",
        text="The Epilogue concludes that although computer science is a science of the artificial.", license_note="test",
    )
    recent_approach = OpenWebRecord(
        source_id="arxiv:recent", source_kind="arxiv", topic_bucket="computing",
        title="Computability logic", source_url="https://arxiv.org/abs/recent",
        retrieved_at="2026-07-13T00:00:00Z",
        text="A recently initiated approach called computability logic is a formal theory of interactive computation.",
        license_note="test",
    )
    subjective_quote = OpenWebRecord(
        source_id="arxiv:quote", source_kind="arxiv", topic_bucket="health",
        title="Positive psychology", source_url="https://arxiv.org/abs/quote",
        retrieved_at="2026-07-13T00:00:00Z",
        text='Positive psychology is the study of what is "right" about people and their strengths.', license_note="test",
    )
    docs, _manifest = write_snapshot_artifacts(
        [computer_science, jaguar, prose, recent_approach, subjective_quote], tmp_path,
    )

    candidates = extract_open_web_abstract_definitions(docs)

    assert [(row["subject"], row["definition"]) for row in candidates] == [
        ("Computer science", "study of the phenomena surrounding computers"),
        ("Jaguar", "ab initio quantum chemical program"),
    ]


def test_campaign_consolidation_deduplicates_regated_segment_overlays(tmp_path: Path):
    item = {"overlay_type": "overlay_definition", "subject": "Mizar Mathematical Library", "definition": "large corpus", "source_kind": "arxiv"}
    for name, rows in (("segment_00", [item]), ("segment_18", [item, {**item, "subject": "Creative Problem Solving", "definition": "sub-area within Artificial Intelligence"}])):
        path = tmp_path / name
        path.mkdir()
        (path / "open_web_proposal_overlay.json").write_text(__import__("json").dumps(rows), encoding="utf-8")

    result = consolidate_regated_campaign(tmp_path)

    assert result["proposal_item_count"] == 2
    assert result["exploratory_relation_item_count"] == 0
    assert result["accepted_memory_modified"] is False
    assert (tmp_path / "open_web_campaign_proposal_overlay.json").is_file()


def test_campaign_consolidation_keeps_relation_support_provenance(tmp_path: Path):
    relation = {
        "overlay_type": "overlay_relation", "subject": "Graph neural networks", "predicate": "uses",
        "object": "message passing", "evidence_text": "Graph neural networks use message passing.",
        "source_url": "https://example.test/a", "risk": "high",
    }
    for name, source_url in (("segment_00", "https://example.test/a"), ("segment_18", "https://example.test/b")):
        path = tmp_path / name
        path.mkdir()
        (path / "open_web_proposal_overlay.json").write_text("[]", encoding="utf-8")
        (path / "open_web_exploratory_relation_overlay.json").write_text(
            __import__("json").dumps([{**relation, "source_url": source_url}]), encoding="utf-8",
        )

    result = consolidate_regated_campaign(tmp_path)
    rows = __import__("json").loads((tmp_path / "open_web_campaign_exploratory_relation_overlay.json").read_text())

    assert result["exploratory_relation_item_count"] == 1
    assert rows[0]["support_count"] == 2
    assert rows[0]["supporting_source_count"] == 2
    graph = __import__("json").loads((tmp_path / "open_web_campaign_exploratory_graph_overlay.json").read_text())
    assert any(item.get("overlay_type") == "overlay_entity" and item.get("label") == "Graph neural networks" for item in graph)
    assert any(item.get("overlay_type") == "overlay_relation" and item.get("risk") == "medium" for item in graph)


def test_campaign_consolidation_writes_evidence_grounded_graph_separately(tmp_path: Path):
    title = "A chemical language model for reticular materials design"
    relation = {
        "overlay_type": "overlay_relation", "subject": title, "predicate": "enables",
        "object": "inverse design in reticular chemistry", "source_page": title,
        "source_record_title": title, "source_url": "https://example.test/nexerra",
        "evidence_text": (
            "Here, we introduce Nexerra-R1, a building-block chemical language model "
            "that enables inverse design in reticular chemistry."
        ),
        "risk": "high",
    }
    path = tmp_path / "segment_00"
    path.mkdir()
    (path / "open_web_proposal_overlay.json").write_text("[]", encoding="utf-8")
    (path / "open_web_exploratory_relation_overlay.json").write_text(
        __import__("json").dumps([relation]), encoding="utf-8",
    )

    result = consolidate_regated_campaign(tmp_path)
    grounded = __import__("json").loads(
        (tmp_path / "open_web_campaign_evidence_grounded_graph_overlay.json").read_text()
    )

    assert result["evidence_grounded_relation_item_count"] == 1
    assert result["evidence_grounded_definition_count"] == 1
    assert any(item.get("subject") == "Nexerra-R1" for item in grounded)
    assert not any(item.get("subject") == title for item in grounded)


def test_campaign_consolidation_adds_only_unique_long_title_method_aliases(tmp_path: Path):
    title = "A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs"
    relation = {
        "overlay_type": "overlay_relation", "subject": title, "predicate": "uses",
        "object": "Large Language Models (LLMs)", "evidence_text": "test evidence",
        "source_url": "https://example.test/paper", "risk": "high",
    }
    path = tmp_path / "segment_00"
    path.mkdir()
    (path / "open_web_proposal_overlay.json").write_text("[]", encoding="utf-8")
    (path / "open_web_exploratory_relation_overlay.json").write_text(
        __import__("json").dumps([relation]), encoding="utf-8",
    )

    consolidate_regated_campaign(tmp_path)
    graph = __import__("json").loads((tmp_path / "open_web_campaign_exploratory_graph_overlay.json").read_text())
    entity = next(item for item in graph if item.get("overlay_type") == "overlay_entity")

    assert entity["aliases"] == [
        "A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities"
    ]


def test_network_run_writes_separate_proposal_with_source_provenance(tmp_path: Path):
    topic = OpenWebTopic("Ada Labs", "testing", ("crossref",))

    def get_json(_url: str):
        return {"message": {"items": [{
            "DOI": "10.1234/ada", "title": ["Ada Labs"],
            "abstract": "Ada Labs was founded by Ada Lovelace.",
            "URL": "https://doi.org/10.1234/ada",
            "published-online": {"date-parts": [[2020, 1, 1]]},
        }]}}

    result = run_open_web_pump(
        output_dir=tmp_path, topics=(topic,), max_queries=1, allow_network=True,
        get_json=get_json,
    )

    assert result["accepted_memory_modified"] is False
    assert result["promoted_overlay_modified"] is False
    assert result["proposal"]["docs_processed"] == 1
    assert (tmp_path / "open_web_proposal_overlay.json").is_file()
    assert (tmp_path / "source_manifest.json").is_file()
    candidates = (tmp_path / "open_web_candidates.json").read_text(encoding="utf-8")
    assert "https://doi.org/10.1234/ada" in candidates
    assert '"source_kind": "crossref"' in candidates


def test_campaign_runner_checkpoints_batches_and_skips_rate_limited_source(tmp_path: Path):
    topics = (
        OpenWebTopic("one", "testing", ("openalex",)),
        OpenWebTopic("two", "testing", ("crossref",)),
        OpenWebTopic("three", "testing", ("arxiv",)),
    )
    calls = []

    def run_batch(**kwargs):
        calls.append(kwargs)
        return {
            "collection": {
                "records_total": 2,
                "rate_limited_sources": ["openalex"] if len(calls) == 1 else [],
                "errors": [],
            },
            "proposal": {"proposal_item_count": 1},
        }

    def consolidate(path):
        return {"proposal_item_count": 3, "campaign_dir": str(path)}

    result = run_open_web_campaign(
        output_dir=tmp_path, topics=topics, batch_size=1, records_per_query=2,
        allow_network=True, run_batch=run_batch, consolidate=consolidate,
    )

    assert [call["start_query"] for call in calls] == [0, 1, 2]
    assert calls[1]["skip_sources"] == ("openalex",)
    assert result["status"] == "completed"
    assert result["campaign"]["proposal_item_count"] == 3
    assert (tmp_path / "open_web_campaign_checkpoint.json").is_file()


def test_campaign_runner_plan_does_not_start_network_batches(tmp_path: Path):
    called = False

    def run_batch(**_kwargs):
        nonlocal called
        called = True
        return {}

    result = run_open_web_campaign(output_dir=tmp_path, allow_network=False, run_batch=run_batch)

    assert result["status"] == "planned_no_network"
    assert called is False
