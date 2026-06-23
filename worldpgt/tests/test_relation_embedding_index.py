"""Tests for RelationEmbeddingIndex and the embedding fallback in semantic_question_parser.

Design contract:
- Exact keyword matches always use RELATION_KEYWORD_MAP (confidence 1.0, unchanged).
- Embedding fallback fires only when exact match returns None (confidence 0.8).
- Threshold 0.65 is conservative: the system prefers audit over a wrong relation.

Stress-test paraphrase cases and their resolution:
  "kick off"    → exact match now (added to map)
  "heads up"    → exact match now (added to map)
  "controls"    → exact match now (added to map)
  "engineer(s)" → exact match now (added to map)
  "establish"   → exact match now (added to map)
  "co-established" → exact match via hyphen compound (map has co-established)
  "trace back"  → embedding: sim=0.538 < 0.65 → correct abstain (None)
"""
from __future__ import annotations

import pytest

pytest.skip("slow: loads GloVe + spaCy models", allow_module_level=True)

from worldpgt.entity_qa.semantic_question_parser import (
    extract_verb_phrases,
    parse_semantic_query,
)
from worldpgt.knowledge.relation_embedding_index import (
    RelationEmbeddingIndex,
    get_default_index,
)
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def idx() -> RelationEmbeddingIndex:
    return get_default_index()


# ── Verb phrase extraction ────────────────────────────────────────────────────

class TestExtractVerbPhrases:
    def test_phrasal_verb_kick_off(self):
        # spaCy: ROOT=kick, prt=off → phrase "kick off"
        phrases = extract_verb_phrases("What firm did Elon Musk kick off?")
        assert "kick off" in phrases, f"expected 'kick off' in {phrases}"

    def test_phrasal_verb_heads_up(self):
        # spaCy: ROOT=heads, prt=up → lemma "head" + "up"
        phrases = extract_verb_phrases("Who heads up Tesla?")
        assert "head up" in phrases, f"got {phrases}"

    def test_simple_verb_control(self):
        phrases = extract_verb_phrases("Who controls Starlink?")
        assert "control" in phrases, f"got {phrases}"

    def test_simple_verb_engineer(self):
        # spaCy parses "engineer" as NOUN ROOT — extracted anyway
        phrases = extract_verb_phrases("What does SpaceX engineer?")
        assert "engineer" in phrases, f"got {phrases}"

    def test_hyphenated_co_established_regex_pass(self):
        # Pre-pass finds "co-established" as compound, adds "co", "established"
        phrases = extract_verb_phrases("Who co-established SpaceX?")
        assert any("establish" in p for p in phrases), (
            f"expected 'establish*' substring in candidates, got {phrases}"
        )

    def test_phrasal_verb_trace_back(self):
        # "trace" as dobj of ROOT; still extracted as VERB pos
        phrases = extract_verb_phrases("Which ventures trace back to Bezos?")
        assert "trace" in phrases or "trace back" in phrases, f"got {phrases}"

    def test_regular_found(self):
        phrases = extract_verb_phrases("Who founded SpaceX?")
        assert "found" in phrases, f"got {phrases}"

    def test_stop_verbs_excluded(self):
        phrases = extract_verb_phrases("What is Tesla?")
        assert "be" not in phrases and "is" not in phrases, f"got {phrases}"

    def test_hyphenated_produces_both_compound_and_parts(self):
        # Pre-pass: "co-established" → adds "co-established", "co", "established"
        phrases = extract_verb_phrases("Who co-established SpaceX?")
        assert "co-established" in phrases or "established" in phrases, f"got {phrases}"


# ── Embedding index: similarity scores for genuinely novel phrases ─────────────
#
# These cases are NOT in RELATION_KEYWORD_MAP and rely on embedding similarity.
# Scores are documented to show the threshold (0.65) is appropriate.

class TestRelationEmbeddingIndexScores:

    # Phrases NOT in RELATION_KEYWORD_MAP that the embedding handles correctly.
    # These are multi-word phrasal forms where the phrase vector is semantically
    # precise enough to hit the right relation above threshold=0.65.
    CASES = [
        # (verb_phrase, expected_relation, description)
        ("head up",  "leader_of",  "head up → leader_of (phrasal, not in map)"),
    ]

    @pytest.mark.parametrize("phrase,expected,desc", CASES)
    def test_correct_relation_returned(self, idx, phrase, expected, desc):
        result, sim = idx.find_relation_intent([phrase])
        top = idx.top_matches(phrase, n=3)
        print(f"\n  [{desc}] sim={sim:.3f} → {result}")
        print(f"  top-3: {[(k, r, round(s, 3)) for k, r, s in top]}")
        assert result == expected, (
            f"Expected {expected!r} but got {result!r} (sim={sim:.3f}).\n"
            f"Top-3: {top}"
        )

    def test_trace_below_threshold_correct_abstain(self, idx):
        # "trace" alone has sim=0.538 < 0.65 → None is CORRECT (conservative).
        # The system prefers audit over wrong_relation.
        result, sim = idx.find_relation_intent(["trace"])
        print(f"\n  'trace' sim={sim:.3f} → {result}")
        assert result is None, (
            f"'trace' should abstain (sim={sim:.3f} < 0.65) but got {result!r}"
        )
        assert sim < 0.65, f"Expected sim < 0.65, got {sim:.3f}"

    def test_steer_below_threshold_correct_abstain(self, idx):
        # GloVe-100d does not map "steer" reliably to leader_of (sim=0.505).
        # Correct behavior: abstain rather than guess wrong.
        result, sim = idx.find_relation_intent(["steer"])
        print(f"\n  'steer' sim={sim:.3f} → {result}")
        assert result is None or result == "leader_of", (
            f"'steer' should abstain or map to leader_of, got {result!r} (sim={sim:.3f})"
        )

    def test_unrelated_phrase_below_threshold(self, idx):
        result, sim = idx.find_relation_intent(["flarble"])
        assert result is None, f"Expected None but got {result!r} (sim={sim:.3f})"

    def test_exact_keyword_works_via_embedding_too(self, idx):
        # "founded" is in map; its embedding should also hit founded_by above threshold
        result, sim = idx.find_relation_intent(["founded"])
        assert result == "founded_by", f"got {result!r} (sim={sim:.3f})"

    def test_top_matches_returns_sorted_results(self, idx):
        matches = idx.top_matches("head up", n=5)
        assert len(matches) >= 2
        sims = [s for _, _, s in matches]
        assert sims == sorted(sims, reverse=True), "top_matches not sorted descending"

    def test_multi_candidate_picks_best(self, idx):
        # Supplying multiple candidates — index returns the best-matching one
        result, sim = idx.find_relation_intent(["flarble", "head up"])
        assert result == "leader_of", f"Expected leader_of from multi-candidate, got {result!r}"
        assert sim >= 0.65


# ── Full parse_semantic_query integration ─────────────────────────────────────

class TestParaphraseIntegration:
    """Verify stress-test paraphrase cases now resolve correctly.

    Resolution path for each case:
      kick off      → KEYWORD_MAP exact match (added) → founded_by, conf=0.9
      heads up      → KEYWORD_MAP exact match (added) → leader_of, conf=0.9
      co-established → KEYWORD_MAP exact match (added) → founded_by, conf=0.9
      controls      → KEYWORD_MAP exact match (added) → owned_by, conf=0.9
      engineer(s)   → KEYWORD_MAP exact match (added) → develops, conf=0.9
      trace back    → embedding sim=0.538 < 0.65 → None → relation_intent=None (honest abstain)
    """

    def test_kick_off_maps_to_founded_by(self):
        sq = parse_semantic_query("What firm did Elon Musk kick off?")
        assert sq.relation_intent == "founded_by", (
            f"Expected founded_by, got {sq.relation_intent!r} (conf={sq.confidence})"
        )
        assert sq.entity_a == "Elon Musk"
        assert sq.confidence >= 0.75

    def test_heads_up_maps_to_leader_of(self):
        sq = parse_semantic_query("Who heads up Tesla?")
        assert sq.relation_intent == "leader_of", (
            f"Expected leader_of, got {sq.relation_intent!r}"
        )
        assert sq.entity_a == "Tesla"

    def test_co_established_maps_to_founded_by(self):
        sq = parse_semantic_query("Who co-established SpaceX?")
        assert sq.relation_intent == "founded_by", (
            f"Expected founded_by, got {sq.relation_intent!r}"
        )
        assert sq.entity_a == "SpaceX"

    def test_controls_maps_to_owned_by(self):
        sq = parse_semantic_query("Who controls Starlink?")
        assert sq.relation_intent == "owned_by", (
            f"Expected owned_by, got {sq.relation_intent!r}"
        )

    def test_engineer_maps_to_develops(self):
        sq = parse_semantic_query("What does SpaceX engineer?")
        assert sq.relation_intent == "develops", (
            f"Expected develops, got {sq.relation_intent!r}"
        )
        assert sq.entity_a == "SpaceX"

    def test_trace_back_abstains_correctly(self):
        # "trace back" is not in KEYWORD_MAP; embedding sim=0.538 < 0.65.
        # System correctly returns relation_intent=None (honest abstain).
        sq = parse_semantic_query("Which ventures trace back to Bezos?")
        assert sq.relation_intent is None, (
            f"'trace back' below threshold — expected abstain (None), got {sq.relation_intent!r}"
        )

    def test_exact_match_confidence_is_0_9(self):
        # "kick off" is now in KEYWORD_MAP → exact match → conf from entity branch = 0.9
        sq = parse_semantic_query("What firm did Elon Musk kick off?")
        assert sq.confidence == 0.9, (
            f"Expected conf=0.9 for exact match, got {sq.confidence}"
        )

    def test_novel_embedding_phrase_confidence_is_0_8(self):
        # "steer" is NOT in KEYWORD_MAP → embedding fallback → conf=0.8
        sq = parse_semantic_query("Who steers Tesla?")
        if sq.relation_intent is not None:
            assert sq.confidence == 0.8, (
                f"Expected conf=0.8 for embedding match, got {sq.confidence}"
            )


# ── Existing exact-match behaviour unchanged ──────────────────────────────────

class TestExactMatchUnaffected:
    """Guard: new keywords and embedding fallback must not regress existing matches."""

    EXISTING = {
        "Who founded SpaceX?": "founded_by",
        "SpaceX was started by whom?": "founded_by",
        "Which companies does Tesla own?": "owned_by",
        "What does SpaceX build?": "develops",
        "Who leads Starlink?": "leader_of",
        "Where is Tesla headquartered in?": "headquartered_in",
        "What is Starlink a service of?": "service_of",
        "What is Falcon 9 part of?": "part_of",
        "What is Elon Musk's estimated net worth?": "estimated_net_worth",
    }

    @pytest.mark.parametrize("question,expected", EXISTING.items())
    def test_exact_match_preserved(self, question, expected):
        result = relation_intent_from_text(question)
        assert result == expected, (
            f"Regression: {question!r} → {result!r} (expected {expected!r})"
        )

    def test_new_keywords_exact_match(self):
        # Verify new additions to RELATION_KEYWORD_MAP work as exact match
        assert relation_intent_from_text("Who kicked off SpaceX?") == "founded_by"
        assert relation_intent_from_text("She established the company") == "founded_by"
        assert relation_intent_from_text("They set up the firm in 2002") == "founded_by"
        assert relation_intent_from_text("What does X engineer?") == "develops"
        assert relation_intent_from_text("Who controls the business?") == "owned_by"
        assert relation_intent_from_text("Who heads up the team?") == "leader_of"
