import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, Concept, ConceptEngine
from core.relations import Relation


# ── helpers ──────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _lifecycle_world(*lifecycles) -> World:
    w = World()
    for tree, fruit, connector in lifecycles:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    return w


SEMYA_CYCLES = [
    ("яблоня",   "яблоко",     "семя"),
    ("груша",    "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин", "семя"),
]
MIXED_CYCLES = SEMYA_CYCLES + [("персиковое_дерево", "персик", "косточка")]


# ── ConceptEngine.discover_relation_groups ────────────────────────────────────

class TestRelationGroups:
    def test_finds_group_sharing_same_target(self):
        rels = _rels(
            ("персик",   "contains", "косточка"),
            ("слива",    "contains", "косточка"),
            ("абрикос",  "contains", "косточка"),
        )
        concepts = ConceptEngine(rels).discover_relation_groups()
        assert len(concepts) == 1
        c = concepts[0]
        assert set(c.members) == {"персик", "слива", "абрикос"}

    def test_concept_kind_is_relation_group(self):
        rels = _rels(("A", "contains", "X"), ("B", "contains", "X"))
        c = ConceptEngine(rels).discover_relation_groups()[0]
        assert c.kind == "relation_group"

    def test_common_relations_label(self):
        rels = _rels(("A", "contains", "X"), ("B", "contains", "X"))
        c = ConceptEngine(rels).discover_relation_groups()[0]
        assert c.common_relations == ["contains -> X"]

    def test_no_group_when_only_one_member(self):
        rels = _rels(("A", "contains", "X"), ("B", "contains", "Y"))
        concepts = ConceptEngine(rels).discover_relation_groups()
        assert concepts == []

    def test_two_distinct_groups(self):
        rels = _rels(
            ("A", "contains", "X"), ("B", "contains", "X"),
            ("C", "contains", "Y"), ("D", "contains", "Y"),
        )
        concepts = ConceptEngine(rels).discover_relation_groups()
        assert len(concepts) == 2
        ids = {c.id for c in concepts}
        assert "concept_rel_contains_X" in ids
        assert "concept_rel_contains_Y" in ids

    def test_confidence_formula(self):
        # 3 members → (3-1)/4 = 0.5
        rels = _rels(
            ("A", "contains", "X"),
            ("B", "contains", "X"),
            ("C", "contains", "X"),
        )
        c = ConceptEngine(rels).discover_relation_groups()[0]
        assert c.confidence == pytest.approx(0.5)

    def test_confidence_capped_at_one(self):
        rels = _rels(*[(str(i), "contains", "X") for i in range(10)])
        c = ConceptEngine(rels).discover_relation_groups()[0]
        assert c.confidence == pytest.approx(1.0)

    def test_member_of_edges_not_used_as_patterns(self):
        """member_of relations should not trigger new relation-group concepts."""
        rels = _rels(
            ("A", "member_of", "concept_1"),
            ("B", "member_of", "concept_1"),
        )
        concepts = ConceptEngine(rels).discover_relation_groups()
        assert concepts == []

    def test_concept_id_encodes_pattern(self):
        rels = _rels(("A", "grows_into", "tree"), ("B", "grows_into", "tree"))
        c = ConceptEngine(rels).discover_relation_groups()[0]
        assert "grows_into" in c.id
        assert "tree" in c.id


# ── ConceptEngine.discover_lifecycle_groups ───────────────────────────────────

class TestLifecycleGroups:
    def test_finds_lifecycle_group(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = ConceptEngine(w.get_relations()).discover_lifecycle_groups()
        assert len(concepts) == 1
        c = concepts[0]
        assert set(c.members) == {"яблоня", "груша", "апельсиновое_дерево"}

    def test_lifecycle_concept_kind(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        c = ConceptEngine(w.get_relations()).discover_lifecycle_groups()[0]
        assert c.kind == "lifecycle_group"

    def test_lifecycle_partitioned_by_connector(self):
        w = _lifecycle_world(*MIXED_CYCLES)
        concepts = ConceptEngine(w.get_relations()).discover_lifecycle_groups()
        kinds = {c.id for c in concepts}
        assert "concept_lifecycle_семя" in kinds
        assert "concept_lifecycle_косточка" not in kinds  # only 1 косточка cycle → no group

    def test_no_group_for_single_lifecycle(self):
        w = _lifecycle_world(("яблоня", "яблоко", "семя"))
        concepts = ConceptEngine(w.get_relations()).discover_lifecycle_groups()
        assert concepts == []

    def test_lifecycle_members_are_trees(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        c = ConceptEngine(w.get_relations()).discover_lifecycle_groups()[0]
        for member in c.members:
            assert member in {"яблоня", "груша", "апельсиновое_дерево"}

    def test_lifecycle_confidence(self):
        # 3 trees in семя group → (3-1)/4 = 0.5
        w = _lifecycle_world(*SEMYA_CYCLES)
        c = ConceptEngine(w.get_relations()).discover_lifecycle_groups()[0]
        assert c.confidence == pytest.approx(0.5)

    def test_lifecycle_common_relations_mentions_connector(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        c = ConceptEngine(w.get_relations()).discover_lifecycle_groups()[0]
        assert "семя" in c.common_relations[0]

    def test_two_connectors_two_groups_when_both_have_two_members(self):
        cycles = [
            ("яблоня",   "яблоко",  "семя"),
            ("груша",    "груша_плод", "семя"),
            ("персиковое_дерево", "персик",  "косточка"),
            ("сливовое_дерево",   "слива",   "косточка"),
        ]
        w = _lifecycle_world(*cycles)
        concepts = ConceptEngine(w.get_relations()).discover_lifecycle_groups()
        assert len(concepts) == 2


# ── discover_concepts combines both ──────────────────────────────────────────

class TestDiscoverConcepts:
    def test_returns_both_kinds(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = ConceptEngine(w.get_relations()).discover_concepts()
        kinds = {c.kind for c in concepts}
        assert "relation_group" in kinds
        assert "lifecycle_group" in kinds

    def test_empty_world_returns_empty(self):
        concepts = ConceptEngine([]).discover_concepts()
        assert concepts == []

    def test_no_duplicate_members_in_concept(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        for c in ConceptEngine(w.get_relations()).discover_concepts():
            assert len(c.members) == len(set(c.members))


# ── World integration ─────────────────────────────────────────────────────────

class TestWorldIntegration:
    def test_discover_concepts_method_exists(self):
        assert callable(World().discover_concepts)

    def test_returns_concept_list(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        assert isinstance(concepts, list)
        assert all(isinstance(c, Concept) for c in concepts)

    def test_member_of_relations_added(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        member_rels = [r for r in w.get_relations() if r.relation_type == "member_of"]
        assert len(member_rels) > 0

    def test_member_of_links_correct_concept(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        lc = next(c for c in concepts if c.kind == "lifecycle_group")
        member_rels = {
            r.source: r.target
            for r in w.get_relations()
            if r.relation_type == "member_of" and r.target == lc.id
        }
        for member in lc.members:
            assert member in member_rels

    def test_concept_objects_added_to_world(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        object_names = {o.name for o in w.get_objects()}
        for c in concepts:
            assert c.id in object_names

    def test_idempotent_repeated_discover(self):
        """Calling discover_concepts twice should not duplicate member_of edges."""
        w = _lifecycle_world(*SEMYA_CYCLES)
        w.discover_concepts()
        count_after_first = len(w.get_relations())
        w.discover_concepts()
        assert len(w.get_relations()) == count_after_first

    def test_explain_object_shows_membership(self):
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        explanation = w.explain_object("яблоня")
        assert "member_of" in explanation

    def test_mixed_cycles_create_relation_group_for_семя_fruits(self):
        """апельсин/яблоко/груша_плод all contain семя → relation_group concept."""
        w = _lifecycle_world(*SEMYA_CYCLES)
        concepts = w.discover_concepts()
        rg = next(
            (c for c in concepts
             if c.kind == "relation_group" and "семя" in c.common_relations[0]),
            None,
        )
        assert rg is not None
        assert len(rg.members) == 3
