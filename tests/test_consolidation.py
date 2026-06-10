import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, ConsolidationEngine, ConsolidatedPattern
from core.relations import Relation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(*triples) -> ConsolidationEngine:
    rels = [Relation(source=s, relation_type=r, target=t) for s, r, t in triples]
    return ConsolidationEngine(rels)


def patterns_of_kind(patterns, kind):
    return [p for p in patterns if p.kind == kind]


# ---------------------------------------------------------------------------
# Relation clusters
# ---------------------------------------------------------------------------

class TestRelationClusters:
    def test_single_relation_type_creates_cluster(self):
        e = make_engine(("A", "produces_result", "B"))
        clusters = e.find_relation_clusters()
        assert len(clusters) == 1
        assert clusters[0].kind == "relation_cluster"

    def test_cluster_name_uses_human_label(self):
        e = make_engine(("A", "produces_result", "B"))
        clusters = e.find_relation_clusters()
        assert clusters
        assert clusters[0].name == "source -> result"

    def test_cluster_members_listed(self):
        e = make_engine(
            ("яблоня", "produces_result", "яблоко"),
            ("курица", "produces_result", "яйцо"),
        )
        clusters = e.find_relation_clusters()
        assert len(clusters) == 1
        members = clusters[0].members
        assert "яблоня->яблоко" in members
        assert "курица->яйцо" in members

    def test_two_relation_types_two_clusters(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("C", "causes_effect", "D"),
        )
        clusters = e.find_relation_clusters()
        kinds = {c.name for c in clusters}
        assert "source -> result" in kinds
        assert "cause -> effect" in kinds

    def test_confidence_scales_with_count(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("C", "produces_result", "D"),
            ("E", "produces_result", "F"),
            ("G", "produces_result", "H"),
            ("I", "produces_result", "J"),
        )
        clusters = e.find_relation_clusters()
        assert clusters[0].confidence == 1.0

    def test_confidence_below_one_for_few_instances(self):
        e = make_engine(("A", "produces_result", "B"))
        clusters = e.find_relation_clusters()
        assert clusters[0].confidence < 1.0

    def test_evidence_collected_from_all_relations(self):
        rels = [
            Relation("A", "produces_result", "B", evidence=["A дал B"]),
            Relation("C", "produces_result", "D", evidence=["C дал D"]),
        ]
        engine = ConsolidationEngine(rels)
        clusters = engine.find_relation_clusters()
        assert "A дал B" in clusters[0].evidence
        assert "C дал D" in clusters[0].evidence


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------

class TestCycles:
    def test_simple_cycle_detected(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "contains", "C"),
            ("C", "grows_into", "A"),
        )
        cycles = e.find_cycles()
        assert len(cycles) == 1
        assert cycles[0].kind == "cycle"

    def test_cycle_name_contains_nodes(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "grows_into", "A"),
        )
        cycles = e.find_cycles()
        assert len(cycles) == 1
        assert "A" in cycles[0].name and "B" in cycles[0].name

    def test_cycle_confidence_is_0_8(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "grows_into", "A"),
        )
        cycles = e.find_cycles()
        assert cycles[0].confidence == pytest.approx(0.8)

    def test_cycle_members_contain_nodes(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "grows_into", "A"),
        )
        cycles = e.find_cycles()
        members = set(cycles[0].members)
        assert "A" in members and "B" in members

    def test_no_cycle_in_linear_chain(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "causes_effect", "C"),
        )
        assert e.find_cycles() == []

    def test_cycle_not_duplicated(self):
        # Same cycle reachable from multiple starting points
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "contains", "C"),
            ("C", "grows_into", "A"),
        )
        cycles = e.find_cycles()
        assert len(cycles) == 1

    def test_apple_tree_cycle_via_world(self):
        world = World()
        world.observe("яблоня дала яблоко")
        world.observe("яблоко содержит семя")
        world.observe("семя вырастает в яблоню")
        engine = ConsolidationEngine(world.get_relations())
        cycles = engine.find_cycles()
        assert len(cycles) == 1
        assert "яблоня" in cycles[0].name


# ---------------------------------------------------------------------------
# Causal chains
# ---------------------------------------------------------------------------

class TestCausalChains:
    def test_two_hop_chain_discovered(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "causes_effect", "C"),
        )
        chains = e.find_causal_chains()
        assert len(chains) >= 1
        found = any("A" in c.members and "C" in c.members for c in chains)
        assert found

    def test_single_hop_not_included(self):
        e = make_engine(("A", "causes_effect", "B"))
        chains = e.find_causal_chains()
        assert chains == []

    def test_chain_kind_is_causal_chain(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "causes_effect", "C"),
        )
        chains = e.find_causal_chains()
        assert all(c.kind == "causal_chain" for c in chains)

    def test_chain_confidence_scales_with_length(self):
        e = make_engine(
            ("A", "produces_result", "B"),
            ("B", "causes_effect", "C"),
            ("C", "enables", "D"),
            ("D", "requires", "E"),
        )
        chains = e.find_causal_chains()
        longest = max(chains, key=lambda c: len(c.members))
        # 5 nodes = 4 hops → min(1.0, 4/4) = 1.0
        assert longest.confidence == pytest.approx(1.0)

    def test_chain_name_includes_nodes(self):
        e = make_engine(
            ("огонь", "causes_effect", "дым"),
            ("дым", "enables", "туман"),
        )
        chains = e.find_causal_chains()
        assert any("огонь" in c.name and "туман" in c.name for c in chains)

    def test_non_causal_relation_type_breaks_chain(self):
        # "unknown_type" is not in CAUSAL_RELATION_TYPES → chain stops
        rels = [
            Relation("A", "produces_result", "B"),
            Relation("B", "unknown_type", "C"),
            Relation("C", "causes_effect", "D"),
        ]
        engine = ConsolidationEngine(rels)
        chains = engine.find_causal_chains()
        # A->B is causal, but B->C is not, so chain A->B is length 2 (< 3 nodes) → excluded
        assert not any("D" in c.members and "A" in c.members for c in chains)


# ---------------------------------------------------------------------------
# score_relation_strength
# ---------------------------------------------------------------------------

class TestScoreRelationStrength:
    def test_single_evidence_low_score(self):
        rels = [Relation("A", "causes_effect", "B", evidence=["A вызвал B"])]
        engine = ConsolidationEngine(rels)
        scores = engine.score_relation_strength()
        key = "A--causes_effect-->B"
        assert key in scores
        assert scores[key] == pytest.approx(1 / 3)

    def test_three_evidence_max_score(self):
        rels = [Relation("A", "causes_effect", "B", evidence=["e1", "e2", "e3"])]
        engine = ConsolidationEngine(rels)
        scores = engine.score_relation_strength()
        assert scores["A--causes_effect-->B"] == pytest.approx(1.0)

    def test_score_capped_at_one(self):
        rels = [Relation("A", "causes_effect", "B", evidence=["e1", "e2", "e3", "e4", "e5"])]
        engine = ConsolidationEngine(rels)
        scores = engine.score_relation_strength()
        assert scores["A--causes_effect-->B"] == pytest.approx(1.0)

    def test_repeated_observation_increases_strength(self):
        world = World()
        world.observe("яблоня дала яблоко")
        world.observe("яблони дала яблоко")   # second evidence for same relation
        engine = ConsolidationEngine(world.get_relations())
        scores = engine.score_relation_strength()
        key = "яблоня--produces_result-->яблоко"
        assert key in scores
        assert scores[key] > 1 / 3   # higher than single-evidence score


# ---------------------------------------------------------------------------
# consolidate() integration
# ---------------------------------------------------------------------------

class TestConsolidateIntegration:
    def test_returns_multiple_kinds(self):
        world = World()
        for text in [
            "яблоня дала яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "огонь вызвал дым",
            "огонь уничтожает лес",
        ]:
            world.observe(text)
        patterns = world.consolidate()
        kinds = {p.kind for p in patterns}
        assert "relation_cluster" in kinds
        assert "cycle" in kinds
        assert "causal_chain" in kinds

    def test_all_patterns_have_required_fields(self):
        world = World()
        world.observe("яблоня дала яблоко")
        world.observe("яблоко содержит семя")
        world.observe("семя вырастает в яблоню")
        for p in world.consolidate():
            assert isinstance(p.name, str) and p.name
            assert isinstance(p.kind, str)
            assert isinstance(p.members, list)
            assert isinstance(p.evidence, list)
            assert 0.0 <= p.confidence <= 1.0

    def test_empty_world_returns_no_patterns(self):
        world = World()
        assert world.consolidate() == []

    def test_cluster_count_matches_relation_types(self):
        world = World()
        world.observe("яблоня дала яблоко")    # produces_result
        world.observe("огонь вызвал дым")       # causes_effect
        world.observe("яблоко содержит семя")   # contains
        patterns = world.consolidate()
        clusters = patterns_of_kind(patterns, "relation_cluster")
        assert len(clusters) == 3
