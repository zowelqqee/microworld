"""Tests for knowledge ingestion pipeline v1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.knowledge.curated_source_reader import CuratedSourceReader
from worldpgt.knowledge.fact_candidate_extractor import FactCandidateExtractor
from worldpgt.knowledge.fact_normalizer import FactNormalizer
from worldpgt.knowledge.memory_update_proposer import MemoryUpdateProposer

_FIXTURE = Path(__file__).parent.parent / "experiments" / "curated_wiki_snippets_v1.json"

_TARGET_TERMS = {"bank", "bat", "seal", "crane", "rock", "spring"}
_EXPECTED_SENSES = {
    "bank": {"financial_institution", "river_edge"},
    "bat": {"animal", "sports_equipment"},
    "seal": {"animal", "closure_stamp"},
    "crane": {"bird", "machine"},
    "rock": {"stone", "music"},
    "spring": {"season", "coil"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_snippets():
    return CuratedSourceReader().load(_FIXTURE)


def _candidates(snippets=None):
    if snippets is None:
        snippets = _load_snippets()
    return FactCandidateExtractor().extract(snippets)


def _normalized(candidates=None):
    if candidates is None:
        candidates = _candidates()
    return FactNormalizer().normalize(candidates)


def _proposals(normalized=None, snippets=None):
    if snippets is None:
        snippets = _load_snippets()
    if normalized is None:
        normalized = _normalized(_candidates(snippets))
    snippets_by_id = {s.source_id: s for s in snippets}
    return MemoryUpdateProposer().propose(normalized, snippets_by_id)


# ---------------------------------------------------------------------------
# 1. Curated source reader loads all source entries
# ---------------------------------------------------------------------------

def test_reader_loads_all_entries():
    snippets = _load_snippets()
    assert len(snippets) == 12, f"Expected 12 snippets, got {len(snippets)}"


def test_reader_covers_all_terms():
    snippets = _load_snippets()
    loaded_terms = {s.term for s in snippets}
    assert loaded_terms == _TARGET_TERMS


def test_reader_covers_all_senses():
    snippets = _load_snippets()
    for term, senses in _EXPECTED_SENSES.items():
        loaded = {s.sense for s in snippets if s.term == term}
        assert loaded == senses, f"term={term}: expected {senses}, got {loaded}"


def test_reader_fields_populated():
    snippets = _load_snippets()
    for s in snippets:
        assert s.source_id
        assert s.term
        assert s.sense
        assert s.text
        assert 0 < s.trust_seed <= 1.0


# ---------------------------------------------------------------------------
# 2 & 3. Extractor emits candidates per term/sense with expected cues
# ---------------------------------------------------------------------------

def test_extractor_covers_all_term_sense_pairs():
    cands = _candidates()
    pairs = {(c.term, c.sense) for c in cands}
    for term, senses in _EXPECTED_SENSES.items():
        for sense in senses:
            assert (term, sense) in pairs, f"Missing extraction for {term}:{sense}"


def _positive_cues_for(term: str, sense: str) -> set[str]:
    cands = _candidates()
    return {
        c.value.lower()
        for c in cands
        if c.term == term and c.sense == sense and c.fact_type == "positive_cue"
    }


def test_seal_animal_cues():
    cues = _positive_cues_for("seal", "animal")
    assert "fish" in cues
    assert "flippers" in cues
    assert "ocean" in cues


def test_seal_closure_stamp_cues():
    cues = _positive_cues_for("seal", "closure_stamp")
    assert "wax" in cues
    assert "envelope" in cues
    assert "document" in cues


def test_bat_sports_equipment_cues():
    cues = _positive_cues_for("bat", "sports_equipment")
    assert "pitcher" in cues
    assert "ball" in cues
    assert "swing" in cues
    assert "player" in cues


def test_bat_animal_cues():
    cues = _positive_cues_for("bat", "animal")
    assert "cave" in cues
    assert "insects" in cues
    assert "wings" in cues
    assert "rafters" in cues


def test_crane_machine_cues():
    cues = _positive_cues_for("crane", "machine")
    assert "hook" in cues
    assert "load" in cues
    assert "operator" in cues


def test_crane_bird_cues():
    cues = _positive_cues_for("crane", "bird")
    assert "reeds" in cues
    assert "neck" in cues
    assert "wings" in cues
    assert "lake" in cues


def test_bank_financial_institution_cues():
    cues = _positive_cues_for("bank", "financial_institution")
    assert "cash" in cues
    assert "account" in cues
    assert "teller" in cues
    assert "deposit" in cues


def test_bank_river_edge_cues():
    cues = _positive_cues_for("bank", "river_edge")
    assert "stream" in cues
    assert "reeds" in cues
    assert "bridge" in cues
    assert "current" in cues


def test_rock_music_cues():
    cues = _positive_cues_for("rock", "music")
    assert "band" in cues
    assert "concert" in cues
    assert "crowd" in cues
    assert "stage" in cues


def test_rock_stone_cues():
    cues = _positive_cues_for("rock", "stone")
    assert "cliff" in cues
    assert "trail" in cues
    assert "ground" in cues
    assert "boulder" in cues


def test_spring_season_cues():
    cues = _positive_cues_for("spring", "season")
    assert "thaw" in cues
    assert "warm" in cues
    assert "mornings" in cues


def test_spring_coil_cues():
    cues = _positive_cues_for("spring", "coil")
    assert "latch" in cues
    assert "handle" in cues
    assert "device" in cues
    assert "coil" in cues


# ---------------------------------------------------------------------------
# 4. Broad cues are rejected or downgraded
# ---------------------------------------------------------------------------

def test_broad_cues_are_rejected():
    norm = _normalized()
    rejected_vals = {n.value for n in norm if n.rejected}
    # "grass" and "soil" are NOT in denylist; "mammal" NOT broad
    # But broad denylist words should be rejected when they appear
    # Check that at least some known-broad values got rejected
    broad_present = {
        n.value for n in norm
        if n.is_broad and n.fact_type in ("positive_cue", "anti_cue")
    }
    for v in broad_present:
        assert v in rejected_vals, f"Broad cue '{v}' not rejected"


def test_broad_cue_is_marked_high_risk():
    norm = _normalized()
    for n in norm:
        if n.rejected:
            assert n.risk_level == "high", f"Rejected cue '{n.value}' not marked high risk"


# ---------------------------------------------------------------------------
# 5. Conflicting cues are not auto-safe
# ---------------------------------------------------------------------------

def test_conflicting_cues_not_auto_safe():
    props = _proposals()
    for p in props:
        if p.evidence.conflicting_cues:
            assert p.recommended_action != "auto_safe_later", (
                f"Proposal {p.proposal_id} has conflicting cues but is auto_safe_later"
            )


def test_conflicting_cues_have_human_review():
    props = _proposals()
    for p in props:
        if p.evidence.conflicting_cues:
            assert p.recommended_action == "human_review"


# ---------------------------------------------------------------------------
# 6. Proposals are deterministic
# ---------------------------------------------------------------------------

def test_proposals_are_deterministic():
    p1 = _proposals()
    p2 = _proposals()
    ids1 = [p.proposal_id for p in p1]
    ids2 = [p.proposal_id for p in p2]
    assert ids1 == ids2

    for a, b in zip(p1, p2):
        assert a.proposed_update.positive_cues == b.proposed_update.positive_cues
        assert a.risk_level == b.risk_level


# ---------------------------------------------------------------------------
# 7. Proposal IDs are stable
# ---------------------------------------------------------------------------

def test_proposal_ids_are_stable():
    props = _proposals()
    expected = {
        ("bank", "financial_institution"),
        ("bank", "river_edge"),
        ("bat", "animal"),
        ("bat", "sports_equipment"),
        ("seal", "animal"),
        ("seal", "closure_stamp"),
        ("crane", "bird"),
        ("crane", "machine"),
        ("rock", "stone"),
        ("rock", "music"),
        ("spring", "season"),
        ("spring", "coil"),
    }
    pairs = {(p.term, p.sense) for p in props}
    assert pairs == expected

    ids = [p.proposal_id for p in props]
    assert len(ids) == len(set(ids)), "Duplicate proposal IDs"
    for pid in ids:
        assert pid.startswith("prop_"), f"Unexpected id format: {pid}"


# ---------------------------------------------------------------------------
# 8. Generated proposals do not directly modify sense_memory.py
# ---------------------------------------------------------------------------

def test_proposals_do_not_import_sense_memory():
    import worldpgt.knowledge.memory_update_proposer as mod
    src = Path(mod.__file__).read_text()
    # Must not import or call sense_memory — docstring mentions are fine
    assert "import sense_memory" not in src
    assert "from worldpgt.continuation.sense_memory" not in src
    assert "ExplicitSenseMemory" not in src


def test_proposals_do_not_write_sense_memory():
    import worldpgt.knowledge.types as t
    import worldpgt.knowledge.fact_normalizer as fn
    import worldpgt.knowledge.memory_update_proposer as mp
    for mod_src in [Path(t.__file__).read_text(), Path(fn.__file__).read_text(), Path(mp.__file__).read_text()]:
        assert "import sense_memory" not in mod_src
        assert "from worldpgt.continuation.sense_memory" not in mod_src
        assert "ExplicitSenseMemory" not in mod_src
        assert ".write(" not in mod_src or "write_text" not in mod_src


# ---------------------------------------------------------------------------
# 9. Proposals never suggest threshold lowering
# ---------------------------------------------------------------------------

def test_proposals_never_suggest_threshold_lowering():
    props = _proposals()
    for p in props:
        update_str = str(p.proposed_update)
        assert "threshold" not in update_str.lower()
        evidence_str = str(p.evidence)
        assert "threshold" not in evidence_str.lower()


# ---------------------------------------------------------------------------
# 10. Proposals never suggest validator weakening
# ---------------------------------------------------------------------------

def test_proposals_never_suggest_validator_weakening():
    props = _proposals()
    for p in props:
        update_str = str(p.proposed_update)
        assert "validator" not in update_str.lower()
        assert "weaken" not in update_str.lower()


# ---------------------------------------------------------------------------
# 11. Proposals never suggest generic fallback generation
# ---------------------------------------------------------------------------

def test_proposals_never_suggest_generic_fallback():
    props = _proposals()
    for p in props:
        update_str = str(p.proposed_update)
        assert "fallback" not in update_str.lower()
        assert "generic" not in update_str.lower()


# ---------------------------------------------------------------------------
# 12. No neural/GPT/training imports or strings in knowledge package
# ---------------------------------------------------------------------------

def test_no_neural_imports_in_knowledge_package():
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    forbidden = ["torch", "transformers", "openai", "gpt", "neural", "backprop",
                 "fine-tun", "finetun", "training", "weights"]
    for py_file in knowledge_dir.glob("*.py"):
        src = py_file.read_text().lower()
        for term in forbidden:
            assert term not in src, f"Forbidden term '{term}' found in {py_file.name}"


# ---------------------------------------------------------------------------
# 13. CLI writes JSON and CSV
# ---------------------------------------------------------------------------

def test_cli_writes_json_and_csv(tmp_path):
    facts_out = tmp_path / "facts.json"
    proposals_out = tmp_path / "proposals.json"
    csv_out = tmp_path / "proposals.csv"

    from worldpgt.experiments.run_knowledge_ingestion_v1 import main
    main([
        "--source", str(_FIXTURE),
        "--output-facts", str(facts_out),
        "--output-proposals", str(proposals_out),
        "--output-proposals-csv", str(csv_out),
    ])

    assert facts_out.exists()
    assert proposals_out.exists()
    assert csv_out.exists()

    facts_data = json.loads(facts_out.read_text())
    assert len(facts_data) > 0

    proposals_data = json.loads(proposals_out.read_text())
    assert len(proposals_data) == 12

    csv_text = csv_out.read_text()
    assert "proposal_id" in csv_text
    assert "risk_level" in csv_text


# ---------------------------------------------------------------------------
# 14. Running ingestion does not modify benchmark outputs
# ---------------------------------------------------------------------------

def test_ingestion_does_not_modify_benchmark_outputs():
    benchmark_dir = Path(__file__).parent.parent / "experiments"
    v1_output = benchmark_dir / "continuation_prompts_v1.csv"
    before = v1_output.read_text() if v1_output.exists() else None

    # Run ingestion in memory only (no file side effects on benchmark artifacts)
    snippets = _load_snippets()
    cands = FactCandidateExtractor().extract(snippets)
    norm = FactNormalizer().normalize(cands)
    proposer = MemoryUpdateProposer()
    props = proposer.propose(norm, {s.source_id: s for s in snippets})

    after = v1_output.read_text() if v1_output.exists() else None
    assert before == after, "Benchmark output was modified by ingestion pipeline"


# ---------------------------------------------------------------------------
# 15. Current trusted benchmark remains unchanged (sense_memory.py not modified)
# ---------------------------------------------------------------------------

def test_sense_memory_not_imported_or_modified():
    import worldpgt.knowledge as kg
    # Importing the knowledge package must not touch sense_memory
    sm_path = Path(__file__).parent.parent / "continuation" / "sense_memory.py"
    mtime_before = sm_path.stat().st_mtime
    _ = CuratedSourceReader().load(_FIXTURE)
    mtime_after = sm_path.stat().st_mtime
    assert mtime_before == mtime_after, "sense_memory.py mtime changed during ingestion"


# ---------------------------------------------------------------------------
# 16. Safety checks required are always present
# ---------------------------------------------------------------------------

def test_safety_checks_always_present():
    from worldpgt.knowledge.types import SAFETY_CHECKS
    props = _proposals()
    for p in props:
        for check in SAFETY_CHECKS:
            assert check in p.safety_checks_required


# ---------------------------------------------------------------------------
# 17. Low-risk proposals have concrete non-empty cues
# ---------------------------------------------------------------------------

def test_low_risk_proposals_have_content():
    props = _proposals()
    low_risk = [p for p in props if p.risk_level == "low"]
    for p in low_risk:
        u = p.proposed_update
        total = (len(u.positive_cues) + len(u.typical_actions)
                 + len(u.typical_locations) + len(u.semantic_frame_hints))
        assert total > 0, f"Low-risk proposal {p.proposal_id} has no content"


# ---------------------------------------------------------------------------
# 18. Each proposal covers expected term/sense from fixture
# ---------------------------------------------------------------------------

def test_all_expected_proposals_present():
    props = _proposals()
    pairs = {(p.term, p.sense) for p in props}
    for term, senses in _EXPECTED_SENSES.items():
        for sense in senses:
            assert (term, sense) in pairs, f"Missing proposal for {term}:{sense}"


# ============================================================================
# SYNTHETIC TESTS — explicit negative / edge-case coverage
# All tests below use hand-built FactCandidate / CuratedSnippet objects so
# they remain independent of the curated_wiki_snippets_v1.json fixture.
# ============================================================================

from worldpgt.knowledge.types import FactCandidate, CuratedSnippet


def _syn_candidate(
    term: str,
    sense: str,
    value: str,
    fact_type: str = "positive_cue",
    confidence: float = 0.8,
    is_multiword: bool = False,
    sid: str = "syn_test",
) -> FactCandidate:
    return FactCandidate(
        source_id=sid,
        term=term,
        sense=sense,
        fact_type=fact_type,
        value=value,
        confidence=confidence,
        is_broad=False,
        is_multiword=is_multiword,
    )


def _syn_snippet(term: str, sense: str, sid: str = "syn_test") -> CuratedSnippet:
    return CuratedSnippet(
        source_id=sid,
        source_type="synthetic",
        term=term,
        sense=sense,
        title=f"Synthetic {term}:{sense}",
        text="synthetic text",
        source_url=None,
        trust_seed=0.8,
    )


# ---------------------------------------------------------------------------
# A. Broad cue rejection — water, light, thing, object, place, people
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broad_word", ["water", "light", "thing", "object", "place", "people"])
def test_broad_single_word_positive_cue_is_rejected(broad_word):
    cand = _syn_candidate("test_term", "test_sense", broad_word)
    norm = FactNormalizer().normalize([cand])
    nf = norm[0]
    assert nf.is_broad, f"'{broad_word}' must be marked broad"
    assert nf.rejected, f"'{broad_word}' must be rejected as positive_cue"
    assert nf.rejection_reason == "broad_cue_denylist"
    assert nf.risk_level == "high"


@pytest.mark.parametrize("broad_word", ["water", "light", "thing", "object", "place", "people"])
def test_broad_single_word_anti_cue_is_rejected(broad_word):
    cand = _syn_candidate("test_term", "test_sense", broad_word, fact_type="anti_cue")
    norm = FactNormalizer().normalize([cand])
    nf = norm[0]
    assert nf.is_broad, f"'{broad_word}' must be marked broad"
    assert nf.rejected, f"'{broad_word}' must be rejected as anti_cue"
    assert nf.risk_level == "high"


@pytest.mark.parametrize("broad_word", ["water", "light", "thing"])
def test_broad_word_as_non_cue_type_is_not_rejected(broad_word):
    # Broad words are only rejected when the fact_type is positive_cue or anti_cue.
    cand = _syn_candidate("test_term", "test_sense", broad_word, fact_type="typical_action")
    norm = FactNormalizer().normalize([cand])
    nf = norm[0]
    assert nf.is_broad, "Still flagged broad"
    assert not nf.rejected, "typical_action broad word must not be rejected"


@pytest.mark.parametrize("broad_word", ["water", "light", "thing", "object", "place", "people"])
def test_broad_word_does_not_reach_proposed_update(broad_word):
    cand = _syn_candidate("test_term", "test_sense", broad_word)
    norm = FactNormalizer().normalize([cand])
    snippets_by_id = {"syn_test": _syn_snippet("test_term", "test_sense")}
    props = MemoryUpdateProposer().propose(norm, snippets_by_id)
    assert len(props) == 1
    p = props[0]
    assert broad_word not in p.proposed_update.positive_cues
    assert broad_word in p.evidence.rejected_broad_cues


# ---------------------------------------------------------------------------
# B. Safe multiword exceptions — not broad, not rejected, not high risk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "construction site", "ocean water", "river bank", "wax seal"
])
def test_safe_multiword_exception_not_broad(phrase):
    cand = _syn_candidate("test_term", "test_sense", phrase, is_multiword=True)
    norm = FactNormalizer().normalize([cand])
    nf = norm[0]
    assert not nf.is_broad, f"'{phrase}' must NOT be broad (narrow multiword exception)"
    assert not nf.rejected, f"'{phrase}' must NOT be rejected"


@pytest.mark.parametrize("phrase", [
    "construction site", "ocean water", "river bank", "wax seal"
])
def test_safe_multiword_exception_not_high_risk(phrase):
    cand = _syn_candidate("test_term", "test_sense", phrase, confidence=0.7, is_multiword=True)
    norm = FactNormalizer().normalize([cand])
    nf = norm[0]
    assert nf.risk_level == "low", f"'{phrase}' should be low risk, got {nf.risk_level!r}"


@pytest.mark.parametrize("phrase", [
    "construction site", "ocean water", "river bank", "wax seal"
])
def test_safe_multiword_exception_reaches_proposed_update(phrase):
    cand = _syn_candidate("test_term", "test_sense", phrase, confidence=0.7, is_multiword=True)
    norm = FactNormalizer().normalize([cand])
    snippets_by_id = {"syn_test": _syn_snippet("test_term", "test_sense")}
    props = MemoryUpdateProposer().propose(norm, snippets_by_id)
    assert len(props) == 1
    assert phrase in props[0].proposed_update.positive_cues, (
        f"'{phrase}' should reach the proposed update"
    )


# ---------------------------------------------------------------------------
# C. Conflict detection — same cue under two senses of the same term
# ---------------------------------------------------------------------------

def _conflict_candidates() -> list[FactCandidate]:
    # "wing" appears under both crane:bird and crane:machine → conflict
    return [
        _syn_candidate("crane", "bird",    "wing", sid="syn_crane_bird"),
        _syn_candidate("crane", "machine", "wing", sid="syn_crane_machine"),
        _syn_candidate("crane", "bird",    "lake", sid="syn_crane_bird"),    # unique
        _syn_candidate("crane", "machine", "hook", sid="syn_crane_machine"), # unique
    ]


def _conflict_snippets() -> dict:
    return {
        "syn_crane_bird":    _syn_snippet("crane", "bird",    "syn_crane_bird"),
        "syn_crane_machine": _syn_snippet("crane", "machine", "syn_crane_machine"),
    }


def test_conflicting_cue_has_conflict_senses_populated():
    norm = FactNormalizer().normalize(_conflict_candidates())
    wing_facts = [n for n in norm if n.value == "wing"]
    assert len(wing_facts) == 2, "Expected one 'wing' entry per sense"
    for nf in wing_facts:
        assert nf.conflict_senses, f"'wing' in {nf.sense} should have conflict_senses set"
        assert len(nf.conflict_senses) == 1


def test_conflicting_cue_is_high_risk():
    norm = FactNormalizer().normalize(_conflict_candidates())
    wing_facts = [n for n in norm if n.value == "wing"]
    for nf in wing_facts:
        assert nf.risk_level == "high", (
            f"Conflicting cue 'wing' in {nf.sense} should be high risk, got {nf.risk_level!r}"
        )


def test_non_conflicting_cues_unaffected_by_sibling_conflict():
    norm = FactNormalizer().normalize(_conflict_candidates())
    lake_facts = [n for n in norm if n.value == "lake"]
    for nf in lake_facts:
        assert not nf.conflict_senses, "'lake' is unique — should have no conflicts"
        assert nf.risk_level != "high"


def test_conflict_proposal_action_is_human_review():
    norm = FactNormalizer().normalize(_conflict_candidates())
    props = MemoryUpdateProposer().propose(norm, _conflict_snippets())
    for p in props:
        assert p.recommended_action == "human_review", (
            f"[{p.term}:{p.sense}] conflict proposal must be human_review"
        )
        assert p.recommended_action != "auto_safe_later"


def test_conflict_risk_is_high_or_medium():
    norm = FactNormalizer().normalize(_conflict_candidates())
    props = MemoryUpdateProposer().propose(norm, _conflict_snippets())
    for p in props:
        assert p.risk_level in ("medium", "high"), (
            f"[{p.term}:{p.sense}] conflict proposal risk should be medium or high, got {p.risk_level!r}"
        )


def test_conflicting_cue_appears_in_evidence_not_in_update():
    norm = FactNormalizer().normalize(_conflict_candidates())
    props = MemoryUpdateProposer().propose(norm, _conflict_snippets())
    all_conflicts = [c for p in props for c in p.evidence.conflicting_cues]
    assert any("wing" in c for c in all_conflicts), "Evidence must mention the conflicting cue 'wing'"
    for p in props:
        assert "wing" not in p.proposed_update.positive_cues, (
            f"Conflicting cue 'wing' must not appear in positive_cues of {p.term}:{p.sense}"
        )


def test_conflict_between_different_terms_is_not_flagged():
    # Same cue word "wing" under bat:animal and crane:bird should NOT conflict
    # (different terms — conflict detection is per-term only)
    cands = [
        _syn_candidate("bat",   "animal", "wing", sid="syn_bat"),
        _syn_candidate("crane", "bird",   "wing", sid="syn_crane"),
    ]
    norm = FactNormalizer().normalize(cands)
    for nf in norm:
        assert not nf.conflict_senses, (
            f"Cross-term 'wing' in {nf.term}:{nf.sense} must not be flagged as conflicting"
        )


# ---------------------------------------------------------------------------
# D. Singularization edge cases — ss-ending words must not be mangled
# ---------------------------------------------------------------------------

def _normalized_value(word: str, is_multiword: bool = False) -> str:
    cand = _syn_candidate("t", "s", word, is_multiword=is_multiword)
    norm = FactNormalizer().normalize([cand])
    return norm[0].value


@pytest.mark.parametrize("word,expected", [
    # ss-ending: must NOT be mangled (regression guard for the gras/pres bug)
    ("grass",    "grass"),
    ("press",    "press"),
    ("compress", "compress"),
    ("mattress", "mattress"),
    # regular -s plurals that should singularize correctly
    ("birds",    "bird"),
    ("banks",    "bank"),
    ("cities",   "city"),
    ("boxes",    "box"),
])
def test_singularization(word: str, expected: str):
    got = _normalized_value(word)
    assert got == expected, f"singularize({word!r}) → {got!r}, expected {expected!r}"


def test_singularization_does_not_mangle_double_s_endings():
    # Explicit regression: the previous bug stripped the trailing s from ss-ending words.
    for word in ("grass", "press", "compress", "mattress"):
        got = _normalized_value(word)
        assert got.endswith("ss") or got == word, (
            f"singularize({word!r}) mangled the double-s ending: got {got!r}"
        )


# ---------------------------------------------------------------------------
# E. Safety — proposals never contain forbidden operational directives
# ---------------------------------------------------------------------------

def test_no_threshold_directive_in_any_proposal_field():
    props = _proposals()
    for p in props:
        for field_name in (
            "positive_cues", "anti_cues", "typical_actions", "typical_locations",
            "parts", "objects", "semantic_frame_hints", "phrase_candidate_hints",
        ):
            for val in getattr(p.proposed_update, field_name):
                assert "threshold" not in val.lower(), (
                    f"Threshold directive in {p.term}:{p.sense} {field_name}: {val!r}"
                )


def test_no_validator_directive_in_any_proposal_field():
    props = _proposals()
    for p in props:
        for field_name in (
            "positive_cues", "anti_cues", "typical_actions", "typical_locations",
            "parts", "objects", "semantic_frame_hints", "phrase_candidate_hints",
        ):
            for val in getattr(p.proposed_update, field_name):
                assert "validator" not in val.lower(), (
                    f"Validator directive in {p.term}:{p.sense} {field_name}: {val!r}"
                )
                assert "weaken" not in val.lower()


def test_no_fallback_directive_in_any_proposal_field():
    props = _proposals()
    for p in props:
        for field_name in (
            "positive_cues", "anti_cues", "typical_actions", "typical_locations",
            "parts", "objects", "semantic_frame_hints", "phrase_candidate_hints",
        ):
            for val in getattr(p.proposed_update, field_name):
                assert "fallback" not in val.lower(), (
                    f"Fallback directive in {p.term}:{p.sense} {field_name}: {val!r}"
                )
                assert "generic" not in val.lower()


def test_knowledge_package_imports_no_threshold_symbols():
    # Must not define or import threshold-adjustment callables/constants.
    # (The safety-check label strings "does_not_lower_thresholds" in types.py are fine.)
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    # These are code-level names that would indicate actual threshold manipulation.
    forbidden_code_patterns = [
        "lower_threshold(",   # function call
        "reduce_threshold(",
        "set_threshold(",
        "CONFIDENCE_THRESHOLD =",
        "TRUST_THRESHOLD =",
        "MIN_THRESHOLD =",
        "continuation_policy.THRESHOLD",
    ]
    for py_file in knowledge_dir.glob("*.py"):
        src = py_file.read_text()
        for pattern in forbidden_code_patterns:
            assert pattern not in src, (
                f"Forbidden threshold code pattern {pattern!r} in {py_file.name}"
            )


def test_knowledge_package_imports_no_validator_modules():
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    forbidden_imports = [
        "from worldpgt.continuation.surface_validator",
        "from worldpgt.continuation.subject_action_validator",
        "from worldpgt.continuation.prompt_tail_validator",
        "import surface_validator",
        "import subject_action_validator",
        "import prompt_tail_validator",
        "import continuation_policy",
        "from worldpgt.continuation.continuation_policy",
        "from worldpgt.continuation.continuation_engine",
    ]
    for py_file in knowledge_dir.glob("*.py"):
        src = py_file.read_text()
        for imp in forbidden_imports:
            assert imp not in src, (
                f"Forbidden import {imp!r} found in {py_file.name}"
            )


def test_knowledge_package_imports_no_neural_or_ml():
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    forbidden = [
        "torch", "transformers", "openai", "backprop", "fine-tun", "finetun",
        "gradient", "weight tensor", "neural network", "model.train", "model.eval",
        "sklearn", "scipy.stats", "tensorflow", "keras",
    ]
    for py_file in knowledge_dir.glob("*.py"):
        src = py_file.read_text().lower()
        for term in forbidden:
            assert term not in src, (
                f"Forbidden ML/neural term {term!r} found in {py_file.name}"
            )
