"""Tests for Wikipedia/Wikidata Ingestion v2 (read-only candidate pipeline).

All tests are deterministic and offline. No network access, no modification of
sense_memory.py, accepted memory artifacts, planner thresholds, or validators.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.knowledge.wiki_claim_extractor import WikiClaimExtractor
from worldpgt.knowledge.wiki_claim_normalizer import (
    classify_risk,
    classify_stability,
    is_volatile_text,
    normalize_entity_id,
)
from worldpgt.knowledge.wiki_entity_extractor import WikiEntityExtractor
from worldpgt.knowledge.wiki_ingestion_v2 import (
    WikiIngestionV2,
    build_summary,
    deduplicate,
    review_candidates,
)
from worldpgt.knowledge.wiki_ingestion_v2_types import (
    ContextLink,
    DefinitionClaim,
    EntityCard,
    RelationClaim,
    SourceQualifiedFact,
    WikiPageRecord,
    WikiSource,
)
from worldpgt.knowledge.wiki_page_reader import WikiPageReader

_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_FIXTURE = _EXPERIMENTS / "wiki_pages_curated_v2.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pages():
    return WikiPageReader().load(_FIXTURE)


@pytest.fixture(scope="module")
def result(pages):
    return WikiIngestionV2().run(pages)


def _page(pages, title):
    return next(p for p in pages if p.title == title)


def _items(result, item_type):
    return [c for c in result.candidates if c.item_type == item_type]


# ---------------------------------------------------------------------------
# 1. Page reader loads curated fixture
# ---------------------------------------------------------------------------


def test_page_reader_loads_fixture(pages):
    # Corpus expanded from 10 to ~50 curated pages (see wiki_corpus_expansion_v1).
    assert len(pages) >= 45
    titles = {p.title for p in pages}
    required = {
        "Elon Musk",
        "Tesla",
        "SpaceX",
        "Forbes",
        "Leadership",
        "Electric vehicle",
        "Rocket",
        "Net worth",
    }
    assert required.issubset(titles)


def test_page_reader_preserves_links_infobox_source(pages):
    musk = _page(pages, "Elon Musk")
    assert musk.source.source_id == "curated_wiki_v2:elon_musk"
    assert musk.source.retrieved_at == "2026-06-14"
    assert musk.infobox["occupation"] == "businessman"
    assert any(l.target == "Forbes" for l in musk.links)


# ---------------------------------------------------------------------------
# 2. Entity card extraction for Elon Musk, Tesla, SpaceX, Forbes
# ---------------------------------------------------------------------------


def test_entity_card_extraction(pages):
    extractor = WikiEntityExtractor()
    by_title = {p.title: extractor.extract(p) for p in pages}

    assert by_title["Elon Musk"].entity_type == "person"
    assert by_title["Tesla"].entity_type == "organization"
    assert by_title["SpaceX"].entity_type == "organization"
    assert by_title["Forbes"].entity_type == "publication"

    for title in ("Elon Musk", "Tesla", "SpaceX", "Forbes"):
        card = by_title[title]
        assert card.item_type == "entity_card"
        assert card.label == title
        assert card.status == "candidate"


def test_entity_card_aliases_musk(pages):
    card = WikiEntityExtractor().extract(_page(pages, "Elon Musk"))
    assert "Elon Reeve Musk" in card.aliases
    assert "Musk" in card.aliases


# ---------------------------------------------------------------------------
# 3. Wikidata id preserved as wikidata:Q...
# ---------------------------------------------------------------------------


def test_wikidata_id_preserved(pages):
    card = WikiEntityExtractor().extract(_page(pages, "Elon Musk"))
    assert card.entity_id == "wikidata:Q317521"


def test_entity_id_falls_back_to_slug_without_qid(pages):
    # Businessperson has no wikidata_id in the fixture.
    card = WikiEntityExtractor().extract(_page(pages, "Businessperson"))
    assert card.entity_id == "wiki:Businessperson"


def test_normalize_entity_id_unit():
    assert normalize_entity_id("Q42", "Douglas Adams") == "wikidata:Q42"
    assert normalize_entity_id(None, "Net worth") == "wiki:Net_worth"
    assert normalize_entity_id("not-a-qid", "Net worth") == "wiki:Net_worth"


# ---------------------------------------------------------------------------
# 4. Lead-paragraph definition extraction
# ---------------------------------------------------------------------------


def test_definition_extraction_tesla(pages):
    definition = WikiClaimExtractor().extract_definition(_page(pages, "Tesla"))
    assert definition is not None
    assert definition.subject == "Tesla"
    assert definition.predicate == "is_a"
    assert definition.object == "American electric vehicle and clean energy company"
    assert definition.stability == "stable"
    assert definition.risk == "low"


def test_definition_extraction_rejects_self_loop_is_a():
    page = WikiPageRecord(
        page_id="wiki:satellite_internet",
        title="Satellite internet",
        lead_paragraph="Satellite internet is a Satellite internet.",
        source=WikiSource(
            source_id="fixture:self_loop",
            source_type="local_fixture",
            retrieved_at="2026-06-18",
        ),
    )

    assert WikiClaimExtractor().extract_definition(page) is None


def test_definition_extraction_for_all_required_pages(result):
    defs = {d.subject for d in _items(result, "definition_claim")}
    for title in ("Elon Musk", "Tesla", "SpaceX", "Forbes", "Net worth"):
        assert title in defs


# ---------------------------------------------------------------------------
# 5. "leadership of Tesla and SpaceX" => two leader_of relations
# ---------------------------------------------------------------------------


def test_leadership_creates_two_leader_of(pages):
    relations = WikiClaimExtractor().extract_relations(_page(pages, "Elon Musk"))
    leader_of = [r for r in relations if r.predicate == "leader_of"]
    objects = {r.object for r in leader_of}
    assert objects == {"Tesla", "SpaceX"}
    for r in leader_of:
        assert r.subject == "Elon Musk"
        assert r.evidence_text == "known for his leadership of Tesla and SpaceX"
        assert r.stability == "semi_stable"
        assert r.risk == "medium"
        assert r.status == "candidate"


# ---------------------------------------------------------------------------
# 6. Forbes link creates only a weak context_link, not a trusted fact
# ---------------------------------------------------------------------------


def test_forbes_is_only_context_link(result):
    # The Forbes hyperlink lives on the Elon Musk page.
    forbes_links = [
        c
        for c in _items(result, "context_link")
        if c.target == "Forbes" and c.source_page == "Elon Musk"
    ]
    assert forbes_links
    for link in forbes_links:
        assert link.strength == "weak"
        assert link.status == "candidate"
        assert link.relation == "mentioned_with"

    # On the Musk page, the Forbes mention never becomes a trusted relation
    # claim (it is only the source_name of a source-qualified fact).
    musk_relations = [
        r for r in _items(result, "relation_claim") if r.source_page == "Elon Musk"
    ]
    for rel in musk_relations:
        assert rel.object != "Forbes"
        assert rel.subject != "Forbes"


# ---------------------------------------------------------------------------
# 7. Net worth claim => source_qualified_fact, volatile, high, requires_recheck
# ---------------------------------------------------------------------------


def test_net_worth_is_source_qualified_fact(result):
    sqfs = _items(result, "source_qualified_fact")
    net_worth = [s for s in sqfs if s.predicate == "estimated_net_worth"]
    assert len(net_worth) >= 1
    fact = next(s for s in net_worth if s.subject == "Elon Musk")
    assert fact.subject == "Elon Musk"
    assert fact.source_name == "Forbes"
    assert fact.object == "US$1.1 trillion"
    assert fact.as_of == "2026-06"
    assert fact.stability == "volatile"
    assert fact.risk == "high"
    assert fact.requires_recheck is True
    assert fact.status == "candidate"


# ---------------------------------------------------------------------------
# 8. Hyperlinks are weak contextual associations
# ---------------------------------------------------------------------------


def test_all_links_become_weak_context_links(pages, result):
    total_links = sum(len(p.links) for p in pages)
    context_links = _items(result, "context_link")
    assert len(context_links) == total_links
    for link in context_links:
        assert link.strength == "weak"
        assert link.relation == "mentioned_with"


# ---------------------------------------------------------------------------
# 9. Context links are never marked trusted
# ---------------------------------------------------------------------------


def test_context_links_never_trusted(result):
    for link in _items(result, "context_link"):
        assert link.status == "candidate"
        assert link.strength == "weak"
        assert link.relation not in {"is_a", "leader_of", "founded"}


# ---------------------------------------------------------------------------
# 10. Volatile claims are not emitted as stable relation claims
# ---------------------------------------------------------------------------


def test_no_volatile_relation_claims(result):
    for rel in _items(result, "relation_claim"):
        # No relation claim should carry a concrete current measurement.
        assert not is_volatile_text(rel.evidence_text + " " + rel.object)
        assert rel.stability in {"stable", "semi_stable"}


# ---------------------------------------------------------------------------
# 11. Review catches source-qualified facts without as_of
# ---------------------------------------------------------------------------


def test_review_catches_sqf_without_as_of():
    bad = SourceQualifiedFact(
        subject="X",
        predicate="estimated_net_worth",
        object="US$1",
        source_name="Forbes",
        as_of="",
        source_page="X",
        evidence_text="X net worth US$1",
    )
    issues = review_candidates([bad])
    codes = {i.code for i in issues}
    assert "sqf_missing_as_of" in codes


# ---------------------------------------------------------------------------
# 12. Review catches volatile facts marked stable
# ---------------------------------------------------------------------------


def test_review_catches_volatile_marked_stable():
    bad = RelationClaim(
        subject="Elon Musk",
        predicate="related_to",
        object="US$1.1 trillion net worth",
        source_page="Elon Musk",
        evidence_text="net worth estimated at US$1.1 trillion",
        stability="semi_stable",
        risk="medium",
    )
    issues = review_candidates([bad])
    codes = {i.code for i in issues}
    assert "volatile_not_marked" in codes


def test_review_catches_sqf_marked_non_volatile():
    bad = SourceQualifiedFact(
        subject="X",
        predicate="estimated_net_worth",
        object="US$1",
        source_name="Forbes",
        as_of="2026-06",
        source_page="X",
        evidence_text="net worth US$1",
        stability="stable",  # wrong
    )
    issues = review_candidates([bad])
    assert "sqf_not_volatile" in {i.code for i in issues}


# ---------------------------------------------------------------------------
# 13. Review catches relation claims without evidence
# ---------------------------------------------------------------------------


def test_review_catches_relation_without_evidence():
    bad = RelationClaim(
        subject="Tesla",
        predicate="produces",
        object="electric cars",
        source_page="Tesla",
        evidence_text="",
    )
    issues = review_candidates([bad])
    assert "relation_missing_evidence" in {i.code for i in issues}


def test_review_catches_relation_empty_subject_object():
    bad = RelationClaim(
        subject="",
        predicate="produces",
        object="",
        source_page="Tesla",
        evidence_text="produces things",
    )
    assert "relation_empty_field" in {i.code for i in review_candidates([bad])}


def test_review_catches_context_link_not_weak():
    bad = ContextLink(
        source_page="Elon Musk",
        surface="Forbes",
        target="Forbes",
        relation="mentioned_with",
        strength="strong",  # type: ignore[arg-type]
    )
    assert "context_link_not_weak" in {i.code for i in review_candidates([bad])}


def test_review_catches_entity_card_no_label():
    bad = EntityCard(entity_id="wikidata:Q1", label="", source_page="X")
    assert "entity_card_no_label" in {i.code for i in review_candidates([bad])}


# ---------------------------------------------------------------------------
# 14. Duplicate candidates are deduplicated or reported
# ---------------------------------------------------------------------------


def test_duplicates_deduplicated_and_reported():
    a = EntityCard(entity_id="wikidata:Q1", label="A", source_page="A")
    b = EntityCard(entity_id="wikidata:Q1", label="A", source_page="A")
    deduped, issues = deduplicate([a, b])
    assert len(deduped) == 1
    assert any(i.code == "duplicate_candidate" for i in issues)


def test_clean_fixture_has_no_duplicates(result):
    assert result.duplicates_removed == 0


# ---------------------------------------------------------------------------
# 15 & 16. Experiment output files written; summary has safe_for_runtime_memory
# ---------------------------------------------------------------------------


def test_experiment_writes_output_files(tmp_path):
    out_json = tmp_path / "candidates.json"
    out_csv = tmp_path / "candidates.csv"
    summary_json = tmp_path / "summary.json"
    review_json = tmp_path / "review.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldpgt.experiments.run_wiki_ingestion_v2",
            "--input",
            str(_FIXTURE),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
            "--summary-json",
            str(summary_json),
            "--review-json",
            str(review_json),
        ],
        cwd=str(Path(__file__).parent.parent.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for path in (out_json, out_csv, summary_json, review_json):
        assert path.exists()

    summary = json.loads(summary_json.read_text())
    assert summary["safe_for_runtime_memory"] is False
    assert summary["pages_total"] >= 45
    assert summary["candidates_total"] > 0
    assert set(summary["by_item_type"]) <= {
        "entity_card",
        "definition_claim",
        "relation_claim",
        "context_link",
        "source_qualified_fact",
    }


def test_summary_safe_for_runtime_memory_false(result):
    assert result.summary["safe_for_runtime_memory"] is False


def test_summary_has_all_required_keys(result):
    s = result.summary
    for key in (
        "pages_total",
        "candidates_total",
        "by_item_type",
        "by_stability",
        "by_risk",
        "review_errors_count",
        "review_warnings_count",
        "volatile_claims_count",
        "context_links_count",
        "relation_claims_count",
        "entity_cards_count",
        "source_qualified_facts_count",
        "safe_for_runtime_memory",
    ):
        assert key in s


def test_clean_fixture_review_is_clean(result):
    assert result.summary["review_errors_count"] == 0


# ---------------------------------------------------------------------------
# 17, 18, 19. Live memory / sense_memory / accepted memory unchanged
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent  # worldmvp/
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"

# Pinned hashes captured before this pipeline existed.
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_memory_artifact_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


def test_pipeline_does_not_write_to_memory(pages):
    # Running the pipeline must not mutate the protected artifacts.
    before = (_sha1(_SENSE_MEMORY), _sha1(_ACCEPTED))
    WikiIngestionV2().run(pages)
    after = (_sha1(_SENSE_MEMORY), _sha1(_ACCEPTED))
    assert before == after


# ---------------------------------------------------------------------------
# Classifier unit tests (stability / risk)
# ---------------------------------------------------------------------------


def test_stability_classifier():
    assert classify_stability("is a company", "is_a") == "stable"
    assert classify_stability("known for leadership", "leader_of") == "semi_stable"
    assert classify_stability("net worth estimated at US$1 trillion") == "volatile"
    assert classify_stability("publishes business rankings", "publishes") == "semi_stable"


def test_risk_classifier():
    assert classify_risk("stable", "is_a") == "low"
    assert classify_risk("semi_stable", "leader_of") == "medium"
    assert classify_risk("volatile") == "high"


def test_is_volatile_text_strict():
    assert is_volatile_text("net worth estimated at US$1.1 trillion")
    assert is_volatile_text("ranked number 1")
    assert not is_volatile_text("publishes business rankings")
    assert not is_volatile_text("net worth is the value of assets minus liabilities")
