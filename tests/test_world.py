import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, AbstractionMiner, TinyParser
from core.relations import Relation


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestTinyParser:
    def setup_method(self):
        self.parser = TinyParser()

    def test_parse_dala(self):
        e = self.parser.parse("яблоня дала яблоко")
        assert e is not None
        assert e.subject == "яблоня"
        assert e.verb == "дала"
        assert e.object == "яблоко"

    def test_parse_snesla(self):
        e = self.parser.parse("курица снесла яйцо")
        assert e is not None
        assert e.subject == "курица"
        assert e.verb == "снесла"
        assert e.object == "яйцо"

    def test_parse_proizvel(self):
        e = self.parser.parse("завод произвел машину")
        assert e is not None
        assert e.subject == "завод"
        assert e.verb == "произвел"
        assert e.object == "машину"

    def test_parse_napisal(self):
        e = self.parser.parse("автор написал книгу")
        assert e is not None
        assert e.subject == "автор"
        assert e.verb == "написал"
        assert e.object == "книгу"

    def test_parse_vyzval(self):
        e = self.parser.parse("огонь вызвал дым")
        assert e is not None
        assert e.subject == "огонь"
        assert e.verb == "вызвал"
        assert e.object == "дым"

    def test_parse_trailing_period(self):
        e = self.parser.parse("огонь вызвал дым.")
        assert e is not None
        assert e.subject == "огонь"

    def test_parse_unknown_returns_none(self):
        assert self.parser.parse("что-то непонятное") is None

    def test_relation_type_produces(self):
        assert self.parser.relation_type("дала") == "produces_result"
        assert self.parser.relation_type("снесла") == "produces_result"
        assert self.parser.relation_type("написал") == "produces_result"

    def test_relation_type_causes(self):
        assert self.parser.relation_type("вызвал") == "causes_effect"


# ---------------------------------------------------------------------------
# World — object storage
# ---------------------------------------------------------------------------

class TestWorldObjects:
    def setup_method(self):
        self.world = World()

    def test_observe_creates_objects(self):
        self.world.observe("яблоня дала яблоко")
        assert self.world.get_object("яблоня") is not None
        assert self.world.get_object("яблоко") is not None

    def test_unknown_object_returns_none(self):
        assert self.world.get_object("несуществующий") is None

    def test_duplicate_observation_does_not_duplicate_objects(self):
        self.world.observe("яблоня дала яблоко")
        self.world.observe("яблоня дала яблоко")
        names = [o.name for o in self.world.get_objects()]
        assert names.count("яблоня") == 1

    def test_explain_object_contains_name(self):
        self.world.observe("яблоня дала яблоко")
        explanation = self.world.explain_object("яблоня")
        assert "яблоня" in explanation


# ---------------------------------------------------------------------------
# World — relations
# ---------------------------------------------------------------------------

class TestWorldRelations:
    def setup_method(self):
        self.world = World()

    def test_produces_result_relation_created(self):
        self.world.observe("яблоня дала яблоко")
        rels = self.world.get_relations()
        assert len(rels) == 1
        assert rels[0].source == "яблоня"
        assert rels[0].relation_type == "produces_result"
        assert rels[0].target == "яблоко"

    def test_causes_effect_relation_created(self):
        self.world.observe("огонь вызвал дым")
        rels = self.world.get_relations()
        assert any(r.relation_type == "causes_effect" for r in rels)

    def test_evidence_stored(self):
        text = "яблоня дала яблоко"
        self.world.observe(text)
        rel = self.world.get_relations()[0]
        assert text in rel.evidence

    def test_duplicate_observation_merges_evidence(self):
        self.world.observe("яблоня дала яблоко")
        self.world.observe("яблоня дала яблоко")
        assert len(self.world.get_relations()) == 1

    def test_explain_relation(self):
        self.world.observe("дождь вызвал наводнение")
        explanation = self.world.explain_relation("дождь", "наводнение")
        assert "causes_effect" in explanation

    def test_explain_missing_relation(self):
        explanation = self.world.explain_relation("солнце", "луна")
        assert "No relation" in explanation

    def test_manual_add_relation(self):
        rel = Relation(source="A", relation_type="test", target="B", evidence=["manual"])
        self.world.add_relation(rel)
        assert rel in self.world.get_relations()


# ---------------------------------------------------------------------------
# Abstractions
# ---------------------------------------------------------------------------

class TestAbstractionMiner:
    def setup_method(self):
        self.world = World()
        self.miner = AbstractionMiner()

    def _feed(self, *texts):
        for t in texts:
            self.world.observe(t)

    def test_produces_result_abstraction_found(self):
        self._feed("яблоня дала яблоко", "курица снесла яйцо")
        abstractions = self.miner.mine(self.world.get_relations())
        types = [ab.relation_type for ab in abstractions]
        assert "produces_result" in types

    def test_causes_effect_abstraction_found(self):
        self._feed("огонь вызвал дым", "дождь вызвал наводнение")
        abstractions = self.miner.mine(self.world.get_relations())
        types = [ab.relation_type for ab in abstractions]
        assert "causes_effect" in types

    def test_analogy_generated(self):
        self._feed("яблоня дала яблоко", "курица снесла яйцо")
        abstractions = self.miner.mine(self.world.get_relations())
        analogies = self.miner.analogies(abstractions)
        assert len(analogies) >= 1
        assert "analogous" in analogies[0]

    def test_no_analogy_for_single_instance(self):
        self._feed("яблоня дала яблоко")
        abstractions = self.miner.mine(self.world.get_relations())
        analogies = self.miner.analogies(abstractions)
        assert analogies == []

    def test_abstraction_pattern_label(self):
        self._feed("яблоня дала яблоко")
        abstractions = self.miner.mine(self.world.get_relations())
        ab = next(a for a in abstractions if a.relation_type == "produces_result")
        assert ab.pattern == "source -> result"
