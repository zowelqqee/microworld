import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, StructuralSimilarityEngine, SimilarityGraph
from core.relations import Relation


# ── helpers ──────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


CONTAINS_WORLD = _rels(
    ("яблоко",     "contains", "семя"),
    ("груша_плод", "contains", "семя"),
    ("апельсин",   "contains", "семя"),
    ("персик",     "contains", "косточка"),
    ("абрикос",    "contains", "косточка"),
    ("слива",      "contains", "косточка"),
)


def _lifecycle_world(*lifecycles) -> World:
    w = World()
    for tree, fruit, conn in lifecycles:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")
    return w


MIXED = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]


# ── entity_profile ────────────────────────────────────────────────────────────

class TestEntityProfile:
    def test_outgoing_features(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        prof = eng.entity_profile("яблоко")
        assert "out:contains" in prof
        assert "out:contains:семя" in prof

    def test_incoming_features(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        prof = eng.entity_profile("семя")
        assert "in:contains" in prof
        assert "in:contains:яблоко" in prof
        assert "in:contains:груша_плод" in prof
        # персик contains косточка, NOT семя
        assert "in:contains:персик" not in prof

    def test_unknown_entity_empty_profile(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.entity_profile("дракон") == set()

    def test_profile_is_a_copy(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        p = eng.entity_profile("яблоко")
        p.add("junk")
        assert "junk" not in eng.entity_profile("яблоко")

    def test_same_group_identical_profiles(self):
        """In a contains-only world, same-connector fruits have identical profiles."""
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.entity_profile("яблоко") == eng.entity_profile("груша_плод")

    def test_member_of_excluded_from_profile(self):
        rels = _rels(
            ("яблоко", "contains", "семя"),
            ("яблоко", "member_of", "concept_1"),
        )
        eng = StructuralSimilarityEngine(rels)
        prof = eng.entity_profile("яблоко")
        assert "out:contains" in prof
        assert not any("member_of" in f for f in prof)


# ── Jaccard similarity ────────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_profiles_score_one(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.similarity("яблоко", "груша_плод") == pytest.approx(1.0)

    def test_cross_group_lower(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        same = eng.similarity("персик", "абрикос")
        cross = eng.similarity("персик", "яблоко")
        assert cross < same

    def test_cross_group_exact_value(self):
        """персик{out:contains, out:contains:косточка} vs яблоко{out:contains, out:contains:семя}
        → 1 shared / 3 union = 0.333."""
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.similarity("персик", "яблоко") == pytest.approx(1 / 3)

    def test_empty_profile_returns_zero(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.similarity("яблоко", "дракон") == pytest.approx(0.0)

    def test_both_unknown_returns_zero(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.similarity("a", "b") == pytest.approx(0.0)

    def test_symmetric(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        assert eng.similarity("персик", "слива") == pytest.approx(
            eng.similarity("слива", "персик")
        )

    def test_connector_nodes_low_similarity(self):
        """семя and косточка share only 'in:contains' → low Jaccard."""
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        score = eng.similarity("семя", "косточка")
        assert score < 0.5


# ── discover_similarities ──────────────────────────────────────────────────────

class TestDiscoverSimilarities:
    def test_returns_in_group_pairs(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        pairs = eng.discover_similarities(min_score=0.5)
        names = {frozenset((a, b)) for a, b, _ in pairs}
        assert frozenset(("яблоко", "груша_плод")) in names
        assert frozenset(("персик", "абрикос")) in names
        assert frozenset(("персик", "слива")) in names

    def test_excludes_cross_group_below_threshold(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        pairs = eng.discover_similarities(min_score=0.5)
        names = {frozenset((a, b)) for a, b, _ in pairs}
        assert frozenset(("яблоко", "персик")) not in names

    def test_sorted_by_score_descending(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        pairs = eng.discover_similarities(min_score=0.1)
        scores = [s for _, _, s in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        high = eng.discover_similarities(min_score=0.9)
        low = eng.discover_similarities(min_score=0.1)
        assert len(high) <= len(low)

    def test_no_self_pairs(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        for a, b, _ in eng.discover_similarities(min_score=0.0):
            assert a != b

    def test_no_duplicate_pairs(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        pairs = eng.discover_similarities(min_score=0.5)
        keys = [frozenset((a, b)) for a, b, _ in pairs]
        assert len(keys) == len(set(keys))

    def test_empty_world_returns_empty(self):
        eng = StructuralSimilarityEngine([])
        assert eng.discover_similarities() == []


# ── add_discovered_similarities ────────────────────────────────────────────────

class TestAddDiscoveredSimilarities:
    def test_loads_into_similarity_graph(self):
        sg = SimilarityGraph()
        eng = StructuralSimilarityEngine(CONTAINS_WORLD, similarity_graph=sg)
        count = eng.add_discovered_similarities(min_score=0.5)
        assert count > 0
        assert sg.similarity("яблоко", "груша_плод") == pytest.approx(1.0)

    def test_returns_count(self):
        sg = SimilarityGraph()
        eng = StructuralSimilarityEngine(CONTAINS_WORLD, similarity_graph=sg)
        count = eng.add_discovered_similarities(min_score=0.5)
        discovered = eng.discover_similarities(min_score=0.5)
        assert count == len(discovered)

    def test_raises_without_similarity_graph(self):
        eng = StructuralSimilarityEngine(CONTAINS_WORLD)
        with pytest.raises(ValueError):
            eng.add_discovered_similarities()


# ── World integration ──────────────────────────────────────────────────────────

class TestWorldIntegration:
    def test_structural_similarity_method(self):
        w = World()
        for fruit, conn in [("яблоко", "семя"), ("груша_плод", "семя")]:
            w.observe(f"{fruit} содержит {conn}")
        assert w.structural_similarity("яблоко", "груша_плод") == pytest.approx(1.0)

    def test_discover_structural_similarities_method(self):
        w = World()
        for fruit, conn in [
            ("яблоко", "семя"), ("груша_плод", "семя"),
            ("персик", "косточка"), ("абрикос", "косточка"),
        ]:
            w.observe(f"{fruit} содержит {conn}")
        pairs = w.discover_structural_similarities(min_score=0.5)
        assert len(pairs) > 0

    def test_add_structural_similarities_populates_graph(self):
        w = _lifecycle_world(*MIXED)
        w.observe("сливовое_дерево производит слива")
        w.observe("слива содержит косточка")
        added = w.add_structural_similarities(min_score=0.5)
        assert added > 0
        # слива should now be similar to персик in the graph
        assert w.similarity("слива", "персик") > 0.0


# ── prediction uses discovered similarity (no oracle) ──────────────────────────

class TestPredictionWithoutOracle:
    def test_prediction_uses_discovered_similarity(self):
        """
        Partial 'слива' anchor; NO manual add_similarity.
        After add_structural_similarities, prediction should pick косточка.
        """
        w = _lifecycle_world(*MIXED)
        w.observe("сливовое_дерево производит слива")
        # слива has produces_result anchor only; structural similarity comes from
        # its produces_result-incoming + (no contains yet). Add contains so слива
        # shares structure with персик/абрикос.
        w.observe("слива содержит косточка")
        w.add_structural_similarities(min_score=0.5)
        preds = w.predict_missing_links()
        closing = next(
            (p for p in preds
             if p.source == "косточка" and p.target == "сливовое_дерево"),
            None,
        )
        assert closing is not None
        assert "nearest lifecycle match" in closing.reason

    def test_no_manual_add_similarity_needed(self):
        """Without structural discovery AND without oracle → majority (семя)."""
        w = _lifecycle_world(*MIXED)
        w.observe("сливовое_дерево производит слива")
        # No add_similarity, no add_structural_similarities
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "семя"  # majority fallback
        assert "majority" in p.reason

    def test_structural_discovery_changes_outcome(self):
        """Same world, structural similarity flips слива from семя to косточка."""
        # Without structural discovery
        w_base = _lifecycle_world(*MIXED)
        w_base.observe("сливовое_дерево производит слива")
        w_base.observe("слива содержит косточка")
        p_base = next(p for p in w_base.predict_missing_links() if p.source == "слива")
        # 'слива содержит косточка' is observed, so the prediction is the closing link.
        # слива->contains is already known; check the closing grows_into prediction:
        base_closing = next(
            (p for p in w_base.predict_missing_links()
             if p.source == "косточка" and p.target == "сливовое_дерево"),
            None,
        )

        # With structural discovery
        w_struct = _lifecycle_world(*MIXED)
        w_struct.observe("сливовое_дерево производит слива")
        w_struct.observe("слива содержит косточка")
        w_struct.add_structural_similarities(min_score=0.5)
        struct_closing = next(
            (p for p in w_struct.predict_missing_links()
             if p.source == "косточка" and p.target == "сливовое_дерево"),
            None,
        )
        assert struct_closing is not None
        assert "nearest lifecycle match" in struct_closing.reason
