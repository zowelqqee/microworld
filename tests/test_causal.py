import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, CausalReasoner, TinyParser
from core.relations import Relation


# ---------------------------------------------------------------------------
# Parser — new relation types
# ---------------------------------------------------------------------------

class TestParserNewVerbs:
    def setup_method(self):
        self.p = TinyParser()

    def _parse_ok(self, text, expected_verb, expected_rel):
        e = self.p.parse(text)
        assert e is not None, f"Failed to parse: {text!r}"
        assert e.verb == expected_verb
        assert self.p.relation_type(e.verb) == expected_rel

    def test_contains(self):
        self._parse_ok("яблоко содержит семя", "содержит", "contains")

    def test_grows_into(self):
        # without preposition
        self._parse_ok("семя вырастает яблоню", "вырастает", "grows_into")
        # with preposition "в" — parser should strip it
        e = self.p.parse("семя вырастает в яблоню")
        assert e is not None
        assert e.subject == "семя"
        assert e.verb == "вырастает"
        assert e.object == "яблоню"  # "в" stripped

    def test_requires(self):
        self._parse_ok("яблоня требует воду", "требует", "requires")

    def test_reduces(self):
        self._parse_ok("засуха уменьшает урожай", "уменьшает", "reduces")

    def test_prevents(self):
        self._parse_ok("огонь уничтожает лес", "уничтожает", "prevents")

    def test_enables(self):
        self._parse_ok("дождь помогает росту", "помогает", "enables")

    def test_enables_usilvaet(self):
        self._parse_ok("инвестиции усиливает производство", "усиливает", "enables")


# ---------------------------------------------------------------------------
# CausalReasoner — trace_forward
# ---------------------------------------------------------------------------

def _make_reasoner(*triples) -> CausalReasoner:
    """Build a CausalReasoner from (source, rel_type, target) triples."""
    rels = [Relation(source=s, relation_type=r, target=t) for s, r, t in triples]
    return CausalReasoner(rels)


class TestTraceForward:
    def test_single_hop(self):
        r = _make_reasoner(("A", "causes_effect", "B"))
        chains = r.trace_forward("A", max_depth=3)
        assert len(chains) == 1
        names = [s.object_name for s in chains[0].steps]
        assert names == ["A", "B"]

    def test_two_hops(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("B", "causes_effect", "C"),
        )
        chains = r.trace_forward("A", max_depth=3)
        # trace_forward returns leaf-terminated chains only
        all_names = [tuple(s.object_name for s in c.steps) for c in chains]
        assert ("A", "B", "C") in all_names

    def test_max_depth_respected(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("B", "causes_effect", "C"),
            ("C", "causes_effect", "D"),
        )
        chains = r.trace_forward("A", max_depth=1)
        for chain in chains:
            assert len(chain.steps) <= 2  # start + 1 hop

    def test_cycle_does_not_loop(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("B", "causes_effect", "A"),
        )
        chains = r.trace_forward("A", max_depth=5)
        # should not crash and each chain should be short
        for chain in chains:
            assert len(chain.steps) <= 3

    def test_no_outgoing_returns_empty(self):
        r = _make_reasoner(("A", "causes_effect", "B"))
        chains = r.trace_forward("B", max_depth=3)
        assert chains == []


# ---------------------------------------------------------------------------
# CausalReasoner — what_if_removed
# ---------------------------------------------------------------------------

class TestWhatIfRemoved:
    def test_direct_downstream_affected(self):
        r = _make_reasoner(("яблоко", "contains", "семя"))
        lines = r.what_if_removed("яблоко")
        assert lines[0] == "яблоко removed"
        assert any("семя" in l for l in lines)

    def test_transitive_impact(self):
        # яблоко → семя → яблоня
        r = _make_reasoner(
            ("яблоко", "contains", "семя"),
            ("семя", "grows_into", "яблоня"),
        )
        lines = r.what_if_removed("яблоко")
        assert any("семя" in l for l in lines)
        assert any("яблоня" in l for l in lines)

    def test_removal_order(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("B", "causes_effect", "C"),
        )
        lines = r.what_if_removed("A")
        # A removed first, then B, then C
        objects = [l.split()[0] for l in lines]
        assert objects.index("A") < objects.index("B") < objects.index("C")

    def test_no_downstream(self):
        r = _make_reasoner(("A", "causes_effect", "B"))
        lines = r.what_if_removed("B")
        assert len(lines) == 1
        assert "removed" in lines[0]

    def test_produces_result_impact_is_lost(self):
        r = _make_reasoner(("яблоня", "produces_result", "яблоко"))
        lines = r.what_if_removed("яблоня")
        assert any("lost" in l for l in lines)


# ---------------------------------------------------------------------------
# CausalReasoner — explain_chain
# ---------------------------------------------------------------------------

class TestExplainChain:
    def test_direct_path(self):
        r = _make_reasoner(("A", "causes_effect", "B"))
        lines = r.explain_chain("A", "B")
        assert len(lines) == 1
        assert "causes_effect" in lines[0]

    def test_indirect_path(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("B", "causes_effect", "C"),
        )
        lines = r.explain_chain("A", "C")
        assert len(lines) == 1
        assert "A" in lines[0] and "C" in lines[0]

    def test_no_path(self):
        r = _make_reasoner(("A", "causes_effect", "B"))
        lines = r.explain_chain("B", "A")
        assert "No causal path" in lines[0]

    def test_multiple_paths(self):
        r = _make_reasoner(
            ("A", "causes_effect", "B"),
            ("A", "enables", "C"),
            ("B", "causes_effect", "D"),
            ("C", "causes_effect", "D"),
        )
        lines = r.explain_chain("A", "D")
        assert len(lines) == 2

    def test_apple_tree_lifecycle(self):
        world = World()
        for text in [
            "яблоня дала яблоко",
            "яблоко содержит семя",
            "семя вырастает яблоню",  # яблоню normalised → яблоня at observe time
        ]:
            world.observe(text)
        # CausalReasoner works on already-normalised relations; use canonical names
        r = CausalReasoner(world.get_relations())
        lines = r.explain_chain("яблоня", "яблоня")
        assert len(lines) >= 1
        assert "No causal path" not in lines[0]
