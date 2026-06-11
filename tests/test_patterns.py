import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.patterns import Pattern, PatternDiscoveryEngine
from core.relations import Relation


# ── helpers ────────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


# ── shared test datasets ───────────────────────────────────────────────────────
#
#  BIGRAM_RELS: two bigram types with different frequencies + one noise row
#
#   apple, banana, cherry  --is_a-->  fruit  --has_property-->  edible   (3 chains)
#   dog, cat               --is_a-->  animal --capable_of-->    move      (2 chains)
#   hammer                 --at_location--> workshop --made_of--> concrete (1 chain)

BIGRAM_RELS = _rels(
    ("apple",    "is_a",         "fruit"),
    ("banana",   "is_a",         "fruit"),
    ("cherry",   "is_a",         "fruit"),
    ("fruit",    "has_property", "edible"),
    ("dog",      "is_a",         "animal"),
    ("cat",      "is_a",         "animal"),
    ("animal",   "capable_of",   "move"),
    ("hammer",   "at_location",  "workshop"),
    ("workshop", "made_of",      "concrete"),
)

# Extend BIGRAM_RELS with one more hop so trigrams exist:
#   edible --is_a--> food  →  trigram: is_a -> has_property -> is_a  (3 chains)
TRIGRAM_RELS = BIGRAM_RELS + _rels(
    ("edible", "is_a", "food"),
)


# ── bigram discovery ───────────────────────────────────────────────────────────

class TestBigramDiscovery:
    def setup_method(self):
        self.eng = PatternDiscoveryEngine(BIGRAM_RELS)

    def test_returns_list_of_patterns(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        assert isinstance(pats, list)
        assert all(isinstance(p, Pattern) for p in pats)

    def test_finds_is_a_has_property(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        assert ("is_a", "has_property") in [p.relations for p in pats]

    def test_finds_is_a_capable_of(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        assert ("is_a", "capable_of") in [p.relations for p in pats]

    def test_count_is_a_has_property(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "has_property"))
        assert p.count == 3  # apple, banana, cherry each form one chain

    def test_count_is_a_capable_of(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "capable_of"))
        assert p.count == 2  # dog, cat

    def test_sorted_by_count_descending(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        counts = [p.count for p in pats]
        assert counts == sorted(counts, reverse=True)

    def test_min_count_excludes_below_threshold(self):
        pats = self.eng.discover_relation_bigrams(min_count=2)
        keys = [p.relations for p in pats]
        assert ("at_location", "made_of") not in keys  # count=1

    def test_min_count_includes_at_threshold(self):
        pats = self.eng.discover_relation_bigrams(min_count=2)
        keys = [p.relations for p in pats]
        assert ("is_a", "capable_of") in keys  # count=2

    def test_min_count_3_leaves_only_one_pattern(self):
        pats = self.eng.discover_relation_bigrams(min_count=3)
        assert len(pats) == 1
        assert pats[0].relations == ("is_a", "has_property")

    def test_support_formula(self):
        total = len(BIGRAM_RELS)
        pats = self.eng.discover_relation_bigrams(min_count=1)
        for p in pats:
            assert p.support == pytest.approx(p.count / total, rel=1e-5)

    def test_support_is_positive(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        for p in pats:
            assert p.support > 0.0

    def test_examples_are_stored(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "has_property"))
        assert len(p.examples) >= 1

    def test_examples_have_three_nodes(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        for p in pats:
            for ex in p.examples:
                assert len(ex) == 3

    def test_examples_capped_at_five(self):
        # 20 sources all going through the same hub
        rels = _rels(
            *[(f"x{i}", "is_a", "hub") for i in range(20)],
            ("hub", "has_property", "nice"),
        )
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "has_property"))
        assert p.count == 20
        assert len(p.examples) <= 5

    def test_empty_relations(self):
        pats = PatternDiscoveryEngine([]).discover_relation_bigrams(min_count=1)
        assert pats == []

    def test_no_chains_returns_empty(self):
        # Disconnected pairs — no node is both source and target
        rels = _rels(("a", "is_a", "b"), ("c", "capable_of", "d"))
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        assert pats == []

    def test_relations_tuple_type(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        for p in pats:
            assert isinstance(p.relations, tuple)
            assert len(p.relations) == 2


# ── trigram discovery ──────────────────────────────────────────────────────────

class TestTrigramDiscovery:
    def setup_method(self):
        self.eng = PatternDiscoveryEngine(TRIGRAM_RELS)

    def test_returns_list_of_patterns(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        assert isinstance(pats, list)
        assert all(isinstance(p, Pattern) for p in pats)

    def test_finds_is_a_has_property_is_a(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        assert ("is_a", "has_property", "is_a") in [p.relations for p in pats]

    def test_trigram_count_correct(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "has_property", "is_a"))
        assert p.count == 3  # apple, banana, cherry

    def test_examples_have_four_nodes(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        for p in pats:
            for ex in p.examples:
                assert len(ex) == 4

    def test_sorted_by_count_descending(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        counts = [p.count for p in pats]
        assert counts == sorted(counts, reverse=True)

    def test_min_count_filtering_trigrams(self):
        # is_a -> has_property -> is_a has count=3; all others should be < 3
        pats = self.eng.discover_relation_trigrams(min_count=3)
        assert all(p.count >= 3 for p in pats)

    def test_support_formula_trigrams(self):
        total = len(TRIGRAM_RELS)
        pats = self.eng.discover_relation_trigrams(min_count=1)
        for p in pats:
            assert p.support == pytest.approx(p.count / total, rel=1e-5)

    def test_empty_input_trigrams(self):
        pats = PatternDiscoveryEngine([]).discover_relation_trigrams(min_count=1)
        assert pats == []

    def test_relations_tuple_has_length_three(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        for p in pats:
            assert isinstance(p.relations, tuple)
            assert len(p.relations) == 3


# ── discover_patterns combined ─────────────────────────────────────────────────

class TestDiscoverPatterns:
    def test_returns_both_keys(self):
        result = PatternDiscoveryEngine(BIGRAM_RELS).discover_patterns(min_count=1)
        assert set(result.keys()) == {"bigrams", "trigrams"}

    def test_bigrams_match_direct_call(self):
        eng = PatternDiscoveryEngine(BIGRAM_RELS)
        direct = eng.discover_relation_bigrams(min_count=2)
        combined = eng.discover_patterns(min_count=2)["bigrams"]
        assert [(p.relations, p.count) for p in direct] == \
               [(p.relations, p.count) for p in combined]

    def test_trigrams_match_direct_call(self):
        eng = PatternDiscoveryEngine(TRIGRAM_RELS)
        direct = eng.discover_relation_trigrams(min_count=1)
        combined = eng.discover_patterns(min_count=1)["trigrams"]
        assert [(p.relations, p.count) for p in direct] == \
               [(p.relations, p.count) for p in combined]


# ── no duplicate pattern counts ───────────────────────────────────────────────

class TestNoDuplicateCounts:
    def test_single_chain_counted_once(self):
        rels = _rels(("a", "r1", "b"), ("b", "r2", "c"))
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("r1", "r2"))
        assert p.count == 1

    def test_two_left_edges_counted_separately(self):
        # Two distinct sources flowing through the same intermediate node
        rels = _rels(("a1", "r1", "b"), ("a2", "r1", "b"), ("b", "r2", "c"))
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("r1", "r2"))
        assert p.count == 2

    def test_two_right_edges_counted_separately(self):
        # One source, two distinct chains
        rels = _rels(("a", "r1", "b"), ("b", "r2", "c1"), ("b", "r2", "c2"))
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("r1", "r2"))
        assert p.count == 2

    def test_no_double_counting_with_shared_hub(self):
        # 3 × 2 = 6 chains through the hub
        rels = _rels(
            ("a1", "r1", "hub"), ("a2", "r1", "hub"), ("a3", "r1", "hub"),
            ("hub", "r2", "b1"), ("hub", "r2", "b2"),
        )
        pats = PatternDiscoveryEngine(rels).discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("r1", "r2"))
        assert p.count == 6


# ── deterministic results ──────────────────────────────────────────────────────

class TestDeterministicResults:
    def test_same_input_same_bigrams(self):
        p1 = PatternDiscoveryEngine(BIGRAM_RELS).discover_relation_bigrams(min_count=1)
        p2 = PatternDiscoveryEngine(BIGRAM_RELS).discover_relation_bigrams(min_count=1)
        assert [(p.relations, p.count) for p in p1] == \
               [(p.relations, p.count) for p in p2]

    def test_same_input_same_trigrams(self):
        p1 = PatternDiscoveryEngine(TRIGRAM_RELS).discover_relation_trigrams(min_count=1)
        p2 = PatternDiscoveryEngine(TRIGRAM_RELS).discover_relation_trigrams(min_count=1)
        assert [(p.relations, p.count) for p in p1] == \
               [(p.relations, p.count) for p in p2]


# ── compute_coverage ──────────────────────────────────────────────────────────

class TestComputeCoverage:
    def setup_method(self):
        self.eng = PatternDiscoveryEngine(BIGRAM_RELS)

    def test_all_relations_participate_when_all_patterns_valid(self):
        bigrams = self.eng.discover_relation_bigrams(min_count=1)
        coverage = self.eng.compute_coverage(bigrams)
        assert coverage == pytest.approx(1.0)

    def test_partial_coverage_excludes_low_count_pattern(self):
        # min_count=2 excludes (at_location, made_of) → hammer & workshop don't participate
        bigrams = self.eng.discover_relation_bigrams(min_count=2)
        coverage = self.eng.compute_coverage(bigrams)
        # 7 of 9 relations participate: apple/banana/cherry/fruit/dog/cat/animal
        assert coverage == pytest.approx(7 / 9)

    def test_empty_bigrams_gives_zero_coverage(self):
        coverage = self.eng.compute_coverage([])
        assert coverage == pytest.approx(0.0)

    def test_empty_relations_gives_zero_coverage(self):
        eng = PatternDiscoveryEngine([])
        assert eng.compute_coverage([]) == pytest.approx(0.0)

    def test_coverage_in_unit_interval(self):
        for min_c in (1, 2, 3, 10):
            bigrams = self.eng.discover_relation_bigrams(min_count=min_c)
            cov = self.eng.compute_coverage(bigrams)
            assert 0.0 <= cov <= 1.0


# ── conceptnet fixture ─────────────────────────────────────────────────────────

_FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "conceptnet_tiny.csv")
)


class TestConceptNetFixture:
    def setup_method(self):
        from core.datasets import load_relations_csv, build_world_from_relations
        rows = load_relations_csv(_FIXTURE)
        w = build_world_from_relations(rows)
        self.eng = PatternDiscoveryEngine(w.get_relations())
        self.total = len(w.get_relations())

    def test_bigrams_found(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        assert len(pats) > 0

    def test_is_a_has_property_present(self):
        # apple/banana/…/peach -is_a-> fruit -has_property-> edible (5 chains)
        pats = self.eng.discover_relation_bigrams(min_count=1)
        assert ("is_a", "has_property") in [p.relations for p in pats]

    def test_is_a_has_property_count(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        p = next(x for x in pats if x.relations == ("is_a", "has_property"))
        assert p.count == 5  # apple, banana, cherry, mango, peach

    def test_trigrams_run_without_error(self):
        pats = self.eng.discover_relation_trigrams(min_count=1)
        assert isinstance(pats, list)

    def test_support_values_positive(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        for p in pats:
            assert p.support > 0

    def test_all_patterns_sorted(self):
        pats = self.eng.discover_relation_bigrams(min_count=1)
        counts = [p.count for p in pats]
        assert counts == sorted(counts, reverse=True)

    def test_discover_patterns_returns_both(self):
        result = self.eng.discover_patterns(min_count=1)
        assert "bigrams" in result and "trigrams" in result

    def test_coverage_nontrivial(self):
        bigrams = self.eng.discover_relation_bigrams(min_count=1)
        cov = self.eng.compute_coverage(bigrams)
        assert cov > 0.0
