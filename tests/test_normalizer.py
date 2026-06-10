import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, EntityNormalizer


# ---------------------------------------------------------------------------
# EntityNormalizer unit tests
# ---------------------------------------------------------------------------

class TestEntityNormalizer:
    def setup_method(self):
        self.norm = EntityNormalizer()

    def test_known_surface_resolves_to_canonical(self):
        assert self.norm.normalize("яблоню") == "яблоня"
        assert self.norm.normalize("семена") == "семя"
        assert self.norm.normalize("машину") == "машина"

    def test_canonical_unchanged(self):
        assert self.norm.normalize("яблоня") == "яблоня"
        assert self.norm.normalize("семя") == "семя"

    def test_unknown_word_returned_unchanged(self):
        assert self.norm.normalize("спутник") == "спутник"

    def test_add_alias_and_normalize(self):
        self.norm.add_alias("яблочку", "яблоко")
        assert self.norm.normalize("яблочку") == "яблоко"

    def test_get_aliases_returns_known_surfaces(self):
        aliases = self.norm.get_aliases("яблоня")
        assert "яблоню" in aliases
        assert "яблоне" in aliases
        assert "яблони" in aliases

    def test_get_aliases_empty_for_unknown(self):
        assert self.norm.get_aliases("марсоход") == []

    def test_add_alias_appears_in_get_aliases(self):
        self.norm.add_alias("семечку", "семя")
        assert "семечку" in self.norm.get_aliases("семя")

    def test_add_alias_no_duplicate(self):
        self.norm.add_alias("яблоню", "яблоня")  # already seeded
        assert self.norm.get_aliases("яблоня").count("яблоню") == 1


# ---------------------------------------------------------------------------
# World integration: normalization during observe()
# ---------------------------------------------------------------------------

class TestWorldNormalization:
    def setup_method(self):
        self.world = World()

    def test_inflected_object_stored_as_canonical(self):
        self.world.observe("семя вырастает в яблоню")
        assert self.world.get_object("яблоня") is not None
        assert self.world.get_object("яблоню") is not None  # lookup also normalizes

    def test_inflected_subject_stored_as_canonical(self):
        self.world.observe("яблони требует воду")   # яблони → яблоня
        # canonical lookup works
        assert self.world.get_object("яблоня") is not None
        # inflected lookup also resolves to the same canonical object
        assert self.world.get_object("яблони") is self.world.get_object("яблоня")
        # raw surface "яблони" is NOT a separate entry in the object store
        assert "яблони" not in self.world._objects

    def test_event_subject_is_canonical(self):
        event = self.world.observe("яблони требует воду")
        assert event.subject == "яблоня"

    def test_event_object_is_canonical(self):
        event = self.world.observe("семя вырастает в яблоню")
        assert event.object == "яблоня"

    def test_raw_text_preserved(self):
        text = "семя вырастает в яблоню"
        event = self.world.observe(text)
        assert event.raw_text == text

    def test_relation_uses_canonical_names(self):
        self.world.observe("семя вырастает в яблоню")
        rels = self.world.get_relations()
        assert len(rels) == 1
        assert rels[0].source == "семя"
        assert rels[0].target == "яблоня"

    def test_семена_normalized_to_семя(self):
        self.world.observe("яблоко содержит семена")
        assert self.world.get_object("семя") is not None
        rels = self.world.get_relations()
        assert rels[0].target == "семя"

    # ------------------------------------------------------------------
    # Deduplication across surface forms
    # ------------------------------------------------------------------

    def test_same_relation_different_surfaces_deduplicates(self):
        self.world.observe("яблоня содержит семя")
        self.world.observe("яблоня содержит семена")  # семена → семя
        rels = self.world.get_relations()
        assert len(rels) == 1

    def test_deduplication_accumulates_evidence(self):
        self.world.observe("яблоня содержит семя")
        self.world.observe("яблоня содержит семена")
        rel = self.world.get_relations()[0]
        assert len(rel.evidence) == 2

    def test_inflected_subject_deduplication(self):
        self.world.observe("яблоня требует воду")
        self.world.observe("яблони требует воду")   # яблони → яблоня
        rels = self.world.get_relations()
        assert len(rels) == 1

    # ------------------------------------------------------------------
    # explain_object / explain_relation accept inflected input
    # ------------------------------------------------------------------

    def test_explain_object_inflected_input(self):
        self.world.observe("яблоня дала яблоко")
        result = self.world.explain_object("яблоню")  # inflected input
        assert "яблоня" in result

    def test_explain_relation_inflected_input(self):
        self.world.observe("дождь вызвал наводнение")
        result = self.world.explain_relation("дождь", "наводнение")
        assert "causes_effect" in result

    # ------------------------------------------------------------------
    # explain_chain accepts inflected source/target
    # ------------------------------------------------------------------

    def test_explain_chain_canonical(self):
        self.world.observe("семя вырастает яблоня")
        lines = self.world.explain_chain("семя", "яблоня")
        assert "No causal path" not in lines[0]

    def test_explain_chain_inflected_target(self):
        self.world.observe("семя вырастает в яблоню")
        # target given as inflected form
        lines = self.world.explain_chain("семя", "яблоню")
        assert "No causal path" not in lines[0]

    def test_explain_chain_inflected_source(self):
        self.world.observe("яблони требует воду")   # яблони → яблоня
        self.world.observe("засуха уменьшает воду")
        # chain via воду doesn't exist (no путь from яблоня to засуха)
        # but explain_chain(яблоню, воду) should resolve яблоню → яблоня
        lines = self.world.explain_chain("яблоню", "воду")
        assert "No causal path" not in lines[0]

    # ------------------------------------------------------------------
    # Custom normalizer injected at World construction
    # ------------------------------------------------------------------

    def test_custom_normalizer_injected(self):
        norm = EntityNormalizer()
        norm.add_alias("яблочко", "яблоко")
        w = World(normalizer=norm)
        event = w.observe("яблочко содержит семя")
        assert event.subject == "яблоко"
