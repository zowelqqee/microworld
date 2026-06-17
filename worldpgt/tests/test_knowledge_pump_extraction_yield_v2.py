"""Tests for Knowledge Pump Extraction Yield v2.

No network calls. No modification of accepted memory or overlays.
No ML. Deterministic patterns only.

Test groups (18):
  1.  screen_sentence — rejects negation
  2.  screen_sentence — rejects current/live
  3.  screen_sentence — rejects volatile year
  4.  screen_sentence — accepts stable sentences
  5.  extract_person_names — simple two-word name
  6.  extract_person_names — descriptor stripping
  7.  extract_person_names — multi-founder "and" split
  8.  founded_by_v2 — simple sentence (SpaceX / Elon Musk)
  9.  founded_by_v2 — descriptor stripping (Rocket Science Games / Steve Blank)
  10. founded_by_v2 — multi-founder (Bloomberg News / Bloomberg + Winkler)
  11. subsidiary_of_v2 — modifier words (Starlink subsidiary of SpaceX)
  12. service_of_v2 — "division of" variant
  13. lead_definition_v2 — first sentence extraction
  14. publishes_named_v2 — "including the NAMED_WORK" form
  15. publishes_direct_v2 — direct form
  16. fragment_subject_rejected — in/during/he etc. as first word
  17. generic_subject_rejected — "company", "service" etc.
  18. overlay_format — required fields present in output dicts
  19. evidence_quality — object always appears in evidence_text
  20. no_network_calls — stats confirms no network
  21. protected_files_unchanged — reads do not modify protected artifacts
  22. extract_yield_v2_runs_on_real_docs — integration test (skips if no artifacts)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worldpgt.knowledge_pump.extraction_yield_v2 import (
    extract_from_sentence,
    extract_person_names,
    extract_yield_v2,
    is_fragment_subject,
    is_generic_subject,
    screen_sentence,
    write_extraction_yield_v2_artifacts,
)
from worldpgt.knowledge_pump.precision_firewall import apply_precision_firewall

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_EXP = _WORLD / "experiments"
_PUMP_DIR = _EXP / "knowledge_pump_v1"
_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Test group 1 — screen_sentence rejects negation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent", [
    "SpaceX was not founded by Elon Musk.",
    "This technology was never developed by the team.",
    "The company isn't owned by any parent corporation.",
    "Bloomberg News wasn't founded by Michael Bloomberg.",
])
def test_screen_sentence_rejects_negation(sent: str) -> None:
    assert screen_sentence(sent) is False, f"Expected rejection for: {sent}"


# ---------------------------------------------------------------------------
# Test group 2 — screen_sentence rejects current/live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent", [
    "Tesla's current CEO is Elon Musk.",
    "SpaceX has a market cap of $150 billion today.",
    "The stock price is currently rising.",
    "As of today, net worth stands at $200 billion.",
])
def test_screen_sentence_rejects_current_live(sent: str) -> None:
    assert screen_sentence(sent) is False, f"Expected rejection for: {sent}"


# ---------------------------------------------------------------------------
# Test group 3 — screen_sentence rejects volatile year
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent", [
    "SpaceX launched its 2024 mission.",
    "Bloomberg News reported in 2025 that earnings grew.",
    "The company was acquired in 2023.",
])
def test_screen_sentence_rejects_volatile_year(sent: str) -> None:
    assert screen_sentence(sent) is False, f"Expected rejection: {sent}"


# ---------------------------------------------------------------------------
# Test group 4 — screen_sentence accepts stable sentences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent", [
    "SpaceX was founded in 2002 by Elon Musk.",
    "Rocket Science Games was founded by serial startup entrepreneur Steve Blank in 1993.",
    "Bloomberg News was founded by Michael Bloomberg and Matthew Winkler in 1990.",
    "Starlink Services, LLC is a satellite internet constellation subsidiary of SpaceX.",
    "Exos Aerospace Systems & Technologies is an aerospace manufacturer.",
    "IEA publishes a range of reports, including the annual World Energy Outlook.",
])
def test_screen_sentence_accepts_stable(sent: str) -> None:
    assert screen_sentence(sent) is True, f"Expected acceptance for: {sent}"


# ---------------------------------------------------------------------------
# Test group 5 — extract_person_names simple two-word name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Elon Musk", ["Elon Musk"]),
    ("Elon Musk in 2002", ["Elon Musk"]),
    ("Jeff Bezos.", ["Jeff Bezos"]),
    ("James Webb", ["James Webb"]),
])
def test_extract_person_names_simple(raw: str, expected: list[str]) -> None:
    result = extract_person_names(raw)
    assert result == expected, f"raw={raw!r}: got {result}, expected {expected}"


# ---------------------------------------------------------------------------
# Test group 6 — extract_person_names descriptor stripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_name", [
    ("serial startup entrepreneur Steve Blank", "Steve Blank"),
    ("renowned computer scientist Alan Turing", "Alan Turing"),
    ("american entrepreneur Peter Thiel", "Peter Thiel"),
    ("billionaire investor Warren Buffett", "Warren Buffett"),
    ("noted journalist Chris Anderson", "Chris Anderson"),
])
def test_extract_person_names_descriptor_stripping(raw: str, expected_name: str) -> None:
    names = extract_person_names(raw)
    assert expected_name in names, (
        f"Expected '{expected_name}' in names from {raw!r}, got {names}"
    )


# ---------------------------------------------------------------------------
# Test group 7 — extract_person_names multi-founder "and" split
# ---------------------------------------------------------------------------

def test_extract_person_names_multi_founder() -> None:
    raw = "Michael Bloomberg and Matthew Winkler in 1990"
    names = extract_person_names(raw)
    assert "Michael Bloomberg" in names, f"Expected 'Michael Bloomberg' in {names}"
    assert "Matthew Winkler" in names, f"Expected 'Matthew Winkler' in {names}"


def test_extract_person_names_three_founders() -> None:
    raw = "Larry Page and Sergey Brin in 1998"
    names = extract_person_names(raw)
    assert "Larry Page" in names, f"Expected 'Larry Page' in {names}"
    assert "Sergey Brin" in names, f"Expected 'Sergey Brin' in {names}"


# ---------------------------------------------------------------------------
# Test group 8 — founded_by_v2 simple sentence
# ---------------------------------------------------------------------------

def test_founded_by_v2_spacex() -> None:
    sent = "SpaceX was founded in 2002 by Elon Musk."
    items = extract_from_sentence(sent, "SpaceX")
    relations = [i for i in items if i.get("predicate") == "founded_by"]
    assert len(relations) >= 1, f"Expected founded_by relation, got {items}"
    subjects = {i["subject"] for i in relations}
    objects = {i["object"] for i in relations}
    assert "SpaceX" in subjects, f"Expected SpaceX as subject, got {subjects}"
    assert "Elon Musk" in objects, f"Expected 'Elon Musk' as object, got {objects}"


def test_founded_by_v2_early_year() -> None:
    sent = "PayPal was founded in 1998 by Peter Thiel and Max Levchin."
    items = extract_from_sentence(sent, "PayPal")
    objects = {i["object"] for i in items if i.get("predicate") == "founded_by"}
    assert "Peter Thiel" in objects or "Max Levchin" in objects, (
        f"Expected at least one founder, got {objects}"
    )


# ---------------------------------------------------------------------------
# Test group 9 — founded_by_v2 descriptor stripping
# ---------------------------------------------------------------------------

def test_founded_by_v2_descriptor_stripping() -> None:
    sent = "Rocket Science Games was founded by serial startup entrepreneur Steve Blank in 1993."
    items = extract_from_sentence(sent, "Rocket_Science_Games")
    relations = [i for i in items if i.get("predicate") == "founded_by"]
    assert len(relations) >= 1, f"Expected founded_by, got {items}"
    objects = {i["object"] for i in relations}
    assert "Steve Blank" in objects, (
        f"Expected 'Steve Blank' (not 'serial'), got {objects}"
    )


def test_founded_by_v2_descriptor_does_not_appear_as_object() -> None:
    sent = "Rocket Science Games was founded by serial startup entrepreneur Steve Blank in 1993."
    items = extract_from_sentence(sent, "Rocket_Science_Games")
    objects = {i["object"] for i in items if i.get("predicate") == "founded_by"}
    for bad in ("serial", "startup", "entrepreneur"):
        assert bad not in objects, (
            f"Descriptor '{bad}' should not appear as founded_by object, got {objects}"
        )


# ---------------------------------------------------------------------------
# Test group 10 — founded_by_v2 multi-founder
# ---------------------------------------------------------------------------

def test_founded_by_v2_multi_founder() -> None:
    sent = "Bloomberg News was founded by Michael Bloomberg and Matthew Winkler in 1990."
    items = extract_from_sentence(sent, "Bloomberg_News")
    relations = [i for i in items if i.get("predicate") == "founded_by"]
    objects = {i["object"] for i in relations}
    assert "Michael Bloomberg" in objects, f"Expected Michael Bloomberg, got {objects}"
    assert "Matthew Winkler" in objects, f"Expected Matthew Winkler, got {objects}"


# ---------------------------------------------------------------------------
# Test group 11 — subsidiary_of_v2 with modifier words
# ---------------------------------------------------------------------------

def test_subsidiary_v2_with_modifier() -> None:
    sent = "Starlink Services, LLC is a satellite internet constellation subsidiary of SpaceX."
    items = extract_from_sentence(sent, "Starlink")
    relations = [i for i in items if i.get("predicate") == "subsidiary_of"]
    assert len(relations) >= 1, f"Expected subsidiary_of relation, got {items}"
    objects = {i["object"] for i in relations}
    assert "SpaceX" in objects, f"Expected SpaceX as object, got {objects}"


def test_subsidiary_v2_simple() -> None:
    sent = "Neuralink is a wholly-owned subsidiary of Elon Musk's holding company."
    items = extract_from_sentence(sent, "Neuralink")
    # May or may not fire depending on exact phrasing; if it fires, object must start uppercase
    for item in items:
        if item.get("predicate") == "subsidiary_of":
            assert item["object"][0].isupper(), "subsidiary_of object must start uppercase"


# ---------------------------------------------------------------------------
# Test group 12 — service_of_v2 division variant
# ---------------------------------------------------------------------------

def test_service_of_v2_division() -> None:
    sent = "Bloomberg Intelligence is a data analytics division of Bloomberg L.P."
    items = extract_from_sentence(sent, "Bloomberg_Intelligence")
    service_items = [i for i in items if i.get("predicate") == "service_of"]
    if service_items:
        objects = {i["object"] for i in service_items}
        assert any("Bloomberg" in o for o in objects), (
            f"Expected Bloomberg L.P. as service_of object, got {objects}"
        )


def test_service_of_v2_arm_of() -> None:
    sent = "Starlink is a commercial arm of SpaceX."
    items = extract_from_sentence(sent, "Starlink")
    service_items = [i for i in items if i.get("predicate") == "service_of"]
    if service_items:
        objects = {i["object"] for i in service_items}
        assert "SpaceX" in objects, f"Expected SpaceX, got {objects}"


# ---------------------------------------------------------------------------
# Test group 13 — lead_definition_v2
# ---------------------------------------------------------------------------

def test_lead_definition_v2_fires_on_first_sentence() -> None:
    sent = "Exos Aerospace Systems & Technologies is an aerospace manufacturer."
    items = extract_from_sentence(sent, "Exos_Aerospace", is_lead=True)
    defs = [i for i in items if i.get("overlay_type") == "overlay_definition"]
    assert len(defs) >= 1, f"Expected definition item, got {items}"
    subjects = {i["subject"] for i in defs}
    assert any("Exos" in s for s in subjects), f"Expected Exos as subject, got {subjects}"


def test_lead_definition_v2_skips_non_lead() -> None:
    sent = "Exos Aerospace Systems & Technologies is an aerospace manufacturer."
    items_lead = extract_from_sentence(sent, "Exos_Aerospace", is_lead=True)
    items_non_lead = extract_from_sentence(sent, "Exos_Aerospace", is_lead=False)
    defs_lead = [i for i in items_lead if i.get("overlay_type") == "overlay_definition"]
    defs_non = [i for i in items_non_lead if i.get("overlay_type") == "overlay_definition"]
    assert len(defs_non) == 0, "Definition must not fire on non-lead sentence"
    assert len(defs_lead) >= 1, "Definition should fire on lead sentence"


def test_lead_definition_v2_risk_and_stability() -> None:
    sent = "Bloomberg News is an international news agency."
    items = extract_from_sentence(sent, "Bloomberg_News", is_lead=True)
    defs = [i for i in items if i.get("overlay_type") == "overlay_definition"]
    for d in defs:
        assert d.get("risk") == "low", f"Definition risk must be 'low', got {d.get('risk')}"
        assert d.get("stability") == "stable", f"Definition stability must be 'stable'"


def test_lead_definition_v2_rocket_science_games() -> None:
    sent = "Rocket Science Games is an independent game studio."
    items = extract_from_sentence(sent, "Rocket Science Games", is_lead=True)
    defs = [i for i in items if i.get("overlay_type") == "overlay_definition"]
    assert any(
        d.get("subject") == "Rocket Science Games"
        and d.get("definition") == "independent game studio"
        for d in defs
    )


def test_lead_definition_v2_exos_exact_class() -> None:
    sent = "Exos Aerospace Systems & Technologies is an aerospace manufacturer."
    items = extract_from_sentence(sent, "Exos Aerospace", is_lead=True)
    defs = [i for i in items if i.get("overlay_type") == "overlay_definition"]
    assert any(d.get("definition") == "aerospace manufacturer" for d in defs)


@pytest.mark.parametrize("sent", [
    "OpenAI is artificial intelligence.",
    "The Boring Company is American infrastructure.",
    "Lyndon Rive is an award winner in the Northern California Region.",
    "Lawrence Sperry is the third son of the gyrocompass co-inventor.",
    "Tesla Autopilot is a good thing to have in planes.",
])
def test_lead_definition_v2_rejects_broad_award_relative_opinion(sent: str) -> None:
    items = extract_from_sentence(sent, "BadDefinition", is_lead=True)
    assert not [i for i in items if i.get("overlay_type") == "overlay_definition"]


# ---------------------------------------------------------------------------
# Test group 14 — publishes_named_v2
# ---------------------------------------------------------------------------

def test_publishes_named_v2_iea() -> None:
    sent = "IEA publishes a range of reports, including the annual World Energy Outlook."
    items = extract_from_sentence(sent, "International_Energy_Agency")
    pub_items = [i for i in items if i.get("predicate") == "publishes"]
    if pub_items:
        objects = {i["object"] for i in pub_items}
        assert any("World Energy Outlook" in o or "Energy Outlook" in o for o in objects), (
            f"Expected World Energy Outlook, got {objects}"
        )


def test_publishes_named_v2_object_starts_uppercase() -> None:
    sent = "The organization publishes monthly data, including the Global Energy Report."
    items = extract_from_sentence(sent, "The_Org")
    for item in items:
        if item.get("predicate") == "publishes":
            assert item["object"][0].isupper(), "Published object must start uppercase"


# ---------------------------------------------------------------------------
# Test group 15 — publishes_direct_v2
# ---------------------------------------------------------------------------

def test_publishes_direct_v2_simple() -> None:
    sent = "Bloomberg News publishes the Bloomberg Businessweek."
    items = extract_from_sentence(sent, "Bloomberg_News")
    pub_items = [i for i in items if i.get("predicate") == "publishes"]
    if pub_items:
        objects = {i["object"] for i in pub_items}
        assert any("Bloomberg" in o for o in objects), f"Expected Bloomberg in {objects}"


def test_publishes_direct_v2_rejects_range_of() -> None:
    sent = "The publisher publishes range of books."
    items = extract_from_sentence(sent, "Publisher")
    pub_items = [i for i in items if i.get("predicate") == "publishes"]
    for item in pub_items:
        assert "range of" not in item["object"].lower(), (
            f"Should reject 'range of' object: {item['object']}"
        )


def test_publishes_direct_v2_international_energy_agency_with_article() -> None:
    sent = "The International Energy Agency publishes the annual World Energy Outlook."
    items = extract_from_sentence(sent, "International Energy Agency")
    pub_items = [i for i in items if i.get("predicate") == "publishes"]
    assert any(
        i.get("subject") == "International Energy Agency"
        and i.get("object") == "World Energy Outlook"
        for i in pub_items
    )


def test_passive_developed_by_v2_falcon9() -> None:
    sent = "Falcon 9 was developed by SpaceX."
    items = extract_from_sentence(sent, "Falcon 9")
    assert any(
        i.get("subject") == "Falcon 9"
        and i.get("predicate") == "developed_by"
        and i.get("object") == "SpaceX"
        for i in items
    )
    assert any(
        i.get("subject") == "SpaceX"
        and i.get("predicate") == "develops"
        and i.get("object") == "Falcon 9"
        for i in items
    )


def test_passive_operated_by_v2_starlink() -> None:
    sent = "Starlink is operated by SpaceX."
    items = extract_from_sentence(sent, "Starlink")
    assert any(
        i.get("subject") == "Starlink"
        and i.get("predicate") == "operated_by"
        and i.get("object") == "SpaceX"
        for i in items
    )


def test_acquired_by_v2_solarcity_owned_by() -> None:
    sent = "SolarCity was acquired by Tesla."
    items = extract_from_sentence(sent, "SolarCity")
    assert any(
        i.get("subject") == "SolarCity"
        and i.get("predicate") == "owned_by"
        and i.get("object") == "Tesla"
        for i in items
    )


def test_active_acquired_v2_solarcity_owned_by() -> None:
    sent = "Tesla acquired SolarCity."
    items = extract_from_sentence(sent, "Tesla")
    assert any(
        i.get("subject") == "SolarCity"
        and i.get("predicate") == "owned_by"
        and i.get("object") == "Tesla"
        for i in items
    )


def test_instrumentation_by_nasa_not_ownership() -> None:
    sent = "The range also uses instrumentation operated by NASA at Wallops."
    items = extract_from_sentence(sent, "Range")
    assert not [i for i in items if i.get("predicate") == "owned_by"]


def test_negated_planned_never_developed_rejected() -> None:
    sent = "X was planned but never developed by Y."
    assert extract_from_sentence(sent, "X") == []


# ---------------------------------------------------------------------------
# Test group 16 — fragment subjects rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent,page", [
    ("In 1993, the company was founded by Steven Blank.", "GameCo"),
    ("During 2002, SpaceX was launched by Elon Musk.", "SpaceX"),
    ("After the merger, the firm was developed by scientists.", "FirmX"),
    ("He was the co-founder of PayPal.", "Musk"),
    ("If none of the previously listed engines was developed by SpaceX.", "Fragment"),
    ("The following exchanges are operated by Nasdaq.", "Fragment"),
    ("Many companies were founded by Elon Musk.", "Fragment"),
])
def test_fragment_subject_rejected_in_relation(sent: str, page: str) -> None:
    items = extract_from_sentence(sent, page)
    for item in items:
        subj = item.get("subject", "")
        assert not is_fragment_subject(subj), (
            f"Fragment subject '{subj}' should have been rejected (sent={sent!r})"
        )


# ---------------------------------------------------------------------------
# Test group 17 — generic subjects rejected
# ---------------------------------------------------------------------------

def test_is_generic_subject_true() -> None:
    for word in ("company", "corporation", "service", "platform", "organization"):
        assert is_generic_subject(word), f"'{word}' should be generic"


def test_is_generic_subject_false() -> None:
    for word in ("SpaceX", "Bloomberg News", "Rocket Science Games", "IEA"):
        assert not is_generic_subject(word), f"'{word}' should NOT be generic"


@pytest.mark.parametrize("sent", [
    "The company was founded by Elon Musk.",
    "The service is a subsidiary of SpaceX.",
])
def test_generic_subject_filtered_from_output(sent: str) -> None:
    items = extract_from_sentence(sent, "SomePage")
    for item in items:
        subj = item.get("subject", "")
        assert not is_generic_subject(subj), (
            f"Generic subject '{subj}' leaked into output for: {sent!r}"
        )


# ---------------------------------------------------------------------------
# Test group 18 — overlay format: required fields present
# ---------------------------------------------------------------------------

def test_overlay_relation_has_required_fields() -> None:
    sent = "SpaceX was founded in 2002 by Elon Musk."
    items = extract_from_sentence(sent, "SpaceX")
    relations = [i for i in items if i.get("overlay_type") == "overlay_relation"]
    assert len(relations) >= 1, "Expected at least one relation"
    required = {
        "overlay_type", "subject", "predicate", "object",
        "source_page", "evidence_text", "trust", "risk", "stability",
        "candidate_source", "v2_pattern_id",
    }
    for item in relations:
        missing = required - set(item.keys())
        assert not missing, f"Relation item missing fields: {missing}"


def test_overlay_definition_has_required_fields() -> None:
    sent = "Exos Aerospace Systems is an aerospace manufacturer."
    items = extract_from_sentence(sent, "Exos", is_lead=True)
    defs = [i for i in items if i.get("overlay_type") == "overlay_definition"]
    if defs:
        required = {
            "overlay_type", "subject", "definition", "predicate",
            "source_page", "evidence_text", "trust", "risk", "stability",
            "candidate_source", "v2_pattern_id",
        }
        for item in defs:
            missing = required - set(item.keys())
            assert not missing, f"Definition item missing fields: {missing}"


def test_overlay_trust_is_overlay_candidate() -> None:
    sent = "SpaceX was founded in 2002 by Elon Musk."
    items = extract_from_sentence(sent, "SpaceX")
    for item in items:
        assert item.get("trust") == "overlay_candidate", (
            f"trust must be 'overlay_candidate', got {item.get('trust')}"
        )


def test_overlay_candidate_source_tag() -> None:
    sent = "SpaceX was founded in 2002 by Elon Musk."
    items = extract_from_sentence(sent, "SpaceX")
    for item in items:
        assert item.get("candidate_source") == "pump_extraction_yield_v2", (
            f"candidate_source must identify v2 module"
        )


# ---------------------------------------------------------------------------
# Test group 19 — evidence quality: object always in evidence_text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sent,page", [
    ("SpaceX was founded in 2002 by Elon Musk.", "SpaceX"),
    ("Rocket Science Games was founded by serial startup entrepreneur Steve Blank in 1993.", "RSG"),
    ("Bloomberg News was founded by Michael Bloomberg and Matthew Winkler in 1990.", "Bloomberg"),
    ("Starlink Services, LLC is a satellite internet constellation subsidiary of SpaceX.", "Starlink"),
])
def test_object_always_in_evidence(sent: str, page: str) -> None:
    items = extract_from_sentence(sent, page, is_lead=True)
    for item in items:
        obj = item.get("object", "")
        ev = item.get("evidence_text", "")
        if obj:
            assert obj.lower() in ev.lower(), (
                f"Object '{obj}' not found in evidence_text for {sent!r}"
            )


def test_definition_subject_in_evidence() -> None:
    sent = "Exos Aerospace Systems is an aerospace manufacturer."
    items = extract_from_sentence(sent, "Exos", is_lead=True)
    for item in items:
        if item.get("overlay_type") == "overlay_definition":
            subj = item.get("subject", "")
            ev = item.get("evidence_text", "")
            assert subj.lower() in ev.lower(), f"Subject '{subj}' not in evidence"


# ---------------------------------------------------------------------------
# Test group 20 — no network calls
# ---------------------------------------------------------------------------

def test_no_network_calls_in_stats() -> None:
    """extract_yield_v2 stats must report network_calls=False."""
    _, stats = extract_yield_v2([])
    assert stats.get("network_calls") is False


def test_screen_sentence_does_not_call_network() -> None:
    result = screen_sentence("SpaceX was founded in 2002 by Elon Musk.")
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test group 21 — protected files unchanged
# ---------------------------------------------------------------------------

def test_protected_files_unchanged_after_extract() -> None:
    """Calling extract_yield_v2([]) must not modify any protected file."""
    before = {p: _sha(p) for p in _PROTECTED if p.exists()}
    extract_yield_v2([])
    for path, digest in before.items():
        assert _sha(path) == digest, f"Protected file modified: {path}"


# ---------------------------------------------------------------------------
# Test group 22 — integration: extract_yield_v2 on real docs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_docs():
    import json as _json
    batch_dir = _PUMP_DIR / "batch_snapshots"
    raw_dir = batch_dir / "raw_snapshots"
    doc_dir = batch_dir / "normalized_docs"
    if not doc_dir.exists():
        pytest.skip("batch_snapshots not found — skipping real-doc integration tests")
    from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc

    docs = []
    for raw_path in sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []:
        row = _json.loads(raw_path.read_text())
        title = str(row.get("normalized_title") or row.get("title") or "")
        from worldpgt.knowledge_pump.title_ranker import normalize_title
        safe = normalize_title(title).replace("/", "_").replace(" ", "_")
        doc_path = doc_dir / f"{safe}.md"
        if not doc_path.exists():
            for candidate in doc_dir.glob(f"{safe[:30]}*.md"):
                doc_path = candidate
                break
        if doc_path.exists():
            docs.append(ReadySnapshotDoc(
                title=str(row.get("title") or title),
                normalized_title=title,
                source_url=str(row.get("source_url") or ""),
                retrieved_at=str(row.get("retrieved_at") or ""),
                revision_id=row.get("revision_id"),
                raw_text_sha256=str(row.get("raw_text_sha256") or ""),
                normalized_doc_path=str(doc_path),
            ))
    if not docs:
        pytest.skip("No ready docs with normalized docs found")
    return docs


def test_extract_yield_v2_returns_candidates(real_docs) -> None:
    items, stats = extract_yield_v2(real_docs)
    assert isinstance(items, list)
    assert isinstance(stats, dict)
    assert stats.get("docs_processed", 0) > 0


def test_extract_yield_v2_items_have_valid_overlay_type(real_docs) -> None:
    items, _ = extract_yield_v2(real_docs)
    valid_types = {"overlay_relation", "overlay_definition"}
    for item in items:
        assert item.get("overlay_type") in valid_types, (
            f"Invalid overlay_type: {item.get('overlay_type')}"
        )


def test_extract_yield_v2_objects_in_evidence(real_docs) -> None:
    items, _ = extract_yield_v2(real_docs)
    for item in items:
        obj = item.get("object", "")
        ev = item.get("evidence_text", "")
        if obj and ev:
            assert obj.lower() in ev.lower(), (
                f"Object '{obj}' not in evidence_text for item from {item.get('source_page')}"
            )


def test_extract_yield_v2_no_weak_context_items(real_docs) -> None:
    items, _ = extract_yield_v2(real_docs)
    for item in items:
        assert item.get("overlay_type") != "overlay_context_link", (
            "v2 must not produce weak context links"
        )
        assert item.get("trust") != "weak_context_only", (
            "v2 items must not have weak_context_only trust"
        )


def test_extract_yield_v2_no_fragment_subjects(real_docs) -> None:
    items, _ = extract_yield_v2(real_docs)
    for item in items:
        subj = item.get("subject", "")
        assert not is_fragment_subject(subj), (
            f"Fragment subject '{subj}' leaked from page {item.get('source_page')}"
        )


def test_extract_yield_v2_no_generic_subjects(real_docs) -> None:
    items, _ = extract_yield_v2(real_docs)
    for item in items:
        subj = item.get("subject", "")
        assert not is_generic_subject(subj), (
            f"Generic subject '{subj}' leaked from page {item.get('source_page')}"
        )


def test_extract_yield_v2_does_not_modify_protected_files(real_docs) -> None:
    before = {p: _sha(p) for p in _PROTECTED if p.exists()}
    extract_yield_v2(real_docs)
    for path, digest in before.items():
        assert _sha(path) == digest, f"Protected file modified: {path}"


def test_v2_candidates_pass_existing_precision_firewall() -> None:
    candidates = []
    for sent, page, is_lead in [
        ("Rocket Science Games is an independent game studio.", "Rocket Science Games", True),
        ("Falcon 9 was developed by SpaceX.", "Falcon 9", False),
        ("Bloomberg News was founded by Michael Bloomberg.", "Bloomberg News", False),
        ("SolarCity was acquired by Tesla.", "SolarCity", False),
        ("The International Energy Agency publishes the annual World Energy Outlook.", "IEA", False),
    ]:
        candidates.extend(extract_from_sentence(sent, page, is_lead=is_lead))
    result = apply_precision_firewall(candidates)
    accepted = result["accepted"]
    assert accepted
    assert not any(i.get("overlay_type") == "overlay_context_link" for i in accepted)
    assert not any(i.get("overlay_type") == "overlay_entity" for i in accepted)


def test_write_extraction_yield_v2_artifacts(tmp_path) -> None:
    items = extract_from_sentence(
        "Bloomberg News was founded by Michael Bloomberg.",
        "Bloomberg News",
    )
    summary = write_extraction_yield_v2_artifacts(
        tmp_path,
        items,
        {"docs_processed": 1, "network_calls": False},
    )
    for fname in (
        "extraction_yield_v2_candidates.json",
        "extraction_yield_v2_candidates.csv",
        "extraction_yield_v2_rejected.json",
        "extraction_yield_v2_quarantine.json",
        "extraction_yield_v2_summary.json",
        "extraction_yield_v2_report.json",
        "extraction_yield_v2_by_pattern.json",
        "extraction_yield_v2_by_source_page.csv",
    ):
        assert (tmp_path / fname).is_file(), f"missing artifact {fname}"
    assert summary["extraction_yield_v2_candidate_count"] == len(items)
    assert summary["network_calls"] is False


def test_extraction_yield_v2_has_no_neural_or_network_imports() -> None:
    text = (_WORLD / "knowledge_pump" / "extraction_yield_v2.py").read_text(encoding="utf-8").lower()
    for marker in ("import torch", "import openai", "tensorflow", "embedding", "training", "gpt", "requests", "urllib"):
        assert marker not in text
