import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import gzip
import io
import pytest

from core.datasets import load_relations_csv, build_world_from_relations
from core.world import World

# ── shared fixture path ────────────────────────────────────────────────────────

# The 21-row fixture shipped with the repo — read-only in tests.
FIXTURE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "conceptnet_tiny.csv")
)

# Separate path used by TestLoadRelationsCSV so it doesn't clobber FIXTURE_PATH.
_TINY_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "conceptnet_tiny_tmp.csv")
)

TINY_ROWS: list[dict[str, str]] = [
    {"source": "apple",  "relation_type": "is_a",        "target": "fruit"},
    {"source": "banana", "relation_type": "is_a",        "target": "fruit"},
    {"source": "cherry", "relation_type": "has_property", "target": "red"},
    {"source": "knife",  "relation_type": "used_for",    "target": "cutting"},
    {"source": "apple",  "relation_type": "at_location", "target": "tree"},
]


def _write_tiny_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relation_type", "target"])
        writer.writeheader()
        writer.writerows(rows)


# ── loading CSV ────────────────────────────────────────────────────────────────

class TestLoadRelationsCSV:
    def setup_method(self):
        _write_tiny_csv(_TINY_PATH, TINY_ROWS)

    def test_returns_list_of_dicts(self):
        rows = load_relations_csv(_TINY_PATH)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_correct_row_count(self):
        rows = load_relations_csv(_TINY_PATH)
        assert len(rows) == len(TINY_ROWS)

    def test_required_fields_present(self):
        rows = load_relations_csv(_TINY_PATH)
        for r in rows:
            assert "source" in r
            assert "relation_type" in r
            assert "target" in r

    def test_values_are_strings(self):
        rows = load_relations_csv(_TINY_PATH)
        for r in rows:
            assert isinstance(r["source"], str)
            assert isinstance(r["relation_type"], str)
            assert isinstance(r["target"], str)

    def test_first_row_values(self):
        rows = load_relations_csv(_TINY_PATH)
        assert rows[0] == {"source": "apple", "relation_type": "is_a", "target": "fruit"}

    def test_relation_types_preserved(self):
        rows = load_relations_csv(_TINY_PATH)
        rel_types = {r["relation_type"] for r in rows}
        assert "is_a" in rel_types
        assert "has_property" in rel_types
        assert "used_for" in rel_types

    def test_reads_fixture_from_disk(self):
        """The bundled conceptnet_tiny.csv fixture is loadable as-is."""
        rows = load_relations_csv(FIXTURE_PATH)
        assert len(rows) >= 5  # the shipped fixture has 21 rows


# ── building world ─────────────────────────────────────────────────────────────

class TestBuildWorldFromRelations:
    def test_returns_world_instance(self):
        assert isinstance(build_world_from_relations(TINY_ROWS), World)

    def test_object_count(self):
        # apple, fruit, banana, cherry, red, knife, cutting, tree → 8
        w = build_world_from_relations(TINY_ROWS)
        assert len(w.get_objects()) == 8

    def test_relation_count_matches_input(self):
        w = build_world_from_relations(TINY_ROWS)
        assert len(w.get_relations()) == len(TINY_ROWS)

    def test_objects_accessible_by_name(self):
        w = build_world_from_relations(TINY_ROWS)
        assert w.get_object("apple") is not None
        assert w.get_object("fruit") is not None
        assert w.get_object("knife") is not None

    def test_relation_types_in_world(self):
        w = build_world_from_relations(TINY_ROWS)
        rel_types = {r.relation_type for r in w.get_relations()}
        assert "is_a" in rel_types
        assert "used_for" in rel_types

    def test_empty_input(self):
        w = build_world_from_relations([])
        assert len(w.get_objects()) == 0
        assert len(w.get_relations()) == 0

    def test_richer_fixture_loads(self):
        rows = load_relations_csv(FIXTURE_PATH)
        w = build_world_from_relations(rows)
        assert len(w.get_objects()) > 0
        assert len(w.get_relations()) > 0


# ── relation normalization ─────────────────────────────────────────────────────

class TestRelationNormalization:
    def test_all_nine_relation_names_accepted(self):
        rows = [
            {"source": "a", "relation_type": "is_a",        "target": "b"},
            {"source": "c", "relation_type": "part_of",     "target": "d"},
            {"source": "e", "relation_type": "has_a",       "target": "f"},
            {"source": "g", "relation_type": "used_for",    "target": "h"},
            {"source": "i", "relation_type": "capable_of",  "target": "j"},
            {"source": "k", "relation_type": "causes",      "target": "l"},
            {"source": "m", "relation_type": "has_property","target": "n"},
            {"source": "o", "relation_type": "at_location", "target": "p"},
            {"source": "q", "relation_type": "made_of",     "target": "r"},
        ]
        w = build_world_from_relations(rows)
        rel_types = {r.relation_type for r in w.get_relations()}
        for expected in (
            "is_a", "part_of", "has_a", "used_for", "capable_of",
            "causes", "has_property", "at_location", "made_of",
        ):
            assert expected in rel_types

    def test_no_slash_prefixes_in_names(self):
        rows = [{"source": "apple", "relation_type": "is_a", "target": "fruit"}]
        w = build_world_from_relations(rows)
        for r in w.get_relations():
            assert not r.relation_type.startswith("/")
            assert not r.source.startswith("/")
            assert not r.target.startswith("/")

    def test_snake_case_preserved(self):
        rows = [
            {"source": "x", "relation_type": "capable_of",  "target": "y"},
            {"source": "a", "relation_type": "has_property", "target": "b"},
            {"source": "c", "relation_type": "at_location",  "target": "d"},
        ]
        w = build_world_from_relations(rows)
        rel_types = {r.relation_type for r in w.get_relations()}
        assert "capable_of" in rel_types
        assert "has_property" in rel_types
        assert "at_location" in rel_types


# ── no duplicate rows ──────────────────────────────────────────────────────────

class TestNoDuplicates:
    def test_exact_duplicates_collapsed(self):
        rows = [
            {"source": "apple", "relation_type": "is_a", "target": "fruit"},
            {"source": "apple", "relation_type": "is_a", "target": "fruit"},
            {"source": "apple", "relation_type": "is_a", "target": "fruit"},
        ]
        w = build_world_from_relations(rows)
        same = [r for r in w.get_relations()
                if r.source == "apple" and r.relation_type == "is_a"]
        assert len(same) == 1

    def test_different_target_not_deduplicated(self):
        rows = [
            {"source": "apple", "relation_type": "is_a", "target": "fruit"},
            {"source": "apple", "relation_type": "is_a", "target": "food"},
        ]
        w = build_world_from_relations(rows)
        assert len(w.get_relations()) == 2

    def test_different_relation_type_not_deduplicated(self):
        rows = [
            {"source": "apple", "relation_type": "is_a",        "target": "fruit"},
            {"source": "apple", "relation_type": "has_property", "target": "fruit"},
        ]
        w = build_world_from_relations(rows)
        assert len(w.get_relations()) == 2

    def test_doubled_rows_same_count_as_single(self):
        w_single = build_world_from_relations(TINY_ROWS)
        w_double = build_world_from_relations(TINY_ROWS + TINY_ROWS)
        assert len(w_single.get_relations()) == len(w_double.get_relations())


# ── demo pipeline doesn't crash ───────────────────────────────────────────────

class TestDemoPipeline:
    def test_smoke_tiny_rows(self):
        """Core pipeline: rows -> world -> concepts -> similarity -> predict."""
        w = build_world_from_relations(TINY_ROWS)
        concepts = w.discover_concepts()
        assert isinstance(concepts, list)
        n = w.add_structural_similarities(min_score=0.3)
        assert isinstance(n, int) and n >= 0
        preds = w.predict_missing_links()
        assert isinstance(preds, list)

    def test_concept_discovery_finds_groups(self):
        """With apple/banana both is_a fruit, a relation-group concept should form."""
        rows = [
            {"source": "apple",  "relation_type": "is_a", "target": "fruit"},
            {"source": "banana", "relation_type": "is_a", "target": "fruit"},
            {"source": "cherry", "relation_type": "is_a", "target": "fruit"},
        ]
        w = build_world_from_relations(rows)
        concepts = w.discover_concepts()
        fruit_concept = next(
            (c for c in concepts if "fruit" in c.id and c.kind == "relation_group"),
            None,
        )
        assert fruit_concept is not None
        assert len(fruit_concept.members) >= 2

    def test_structural_similarity_finds_same_group(self):
        """apple and banana both -is_a-> fruit → Jaccard score > 0."""
        rows = [
            {"source": "apple",  "relation_type": "is_a", "target": "fruit"},
            {"source": "banana", "relation_type": "is_a", "target": "fruit"},
        ]
        w = build_world_from_relations(rows)
        score = w.structural_similarity("apple", "banana")
        assert score > 0.0

    def test_predict_does_not_crash_on_fixture(self):
        rows = load_relations_csv(FIXTURE_PATH)
        w = build_world_from_relations(rows)
        w.discover_concepts()
        w.add_structural_similarities(min_score=0.4)
        preds = w.predict_missing_links()
        assert isinstance(preds, list)

    def test_full_pipeline_on_richer_fixture(self):
        """Full fixture: load -> world -> concepts -> similarity -> predict."""
        rows = load_relations_csv(FIXTURE_PATH)
        w = build_world_from_relations(rows)
        assert len(w.get_objects()) >= 10
        concepts = w.discover_concepts()
        assert len(concepts) >= 1
        n_pairs = w.add_structural_similarities(min_score=0.3)
        assert n_pairs >= 0
        preds = w.predict_missing_links()
        assert isinstance(preds, list)


# ── build_conceptnet_sample core logic ────────────────────────────────────────

class TestBuildConceptnetSample:
    """Tests for the extraction logic in scripts/build_conceptnet_sample.py."""

    @staticmethod
    def _make_gz_input(lines: list[str]) -> str:
        """Write a synthetic .gz assertions file to a temp path, return path."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".csv.gz")
        os.close(fd)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def _run(self, lines: list[str], lang: str = "en", max_rows: int = 100) -> list[list[str]]:
        import tempfile
        sys.path.insert(0, os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "scripts")
        ))
        from build_conceptnet_sample import build_sample

        gz_path = self._make_gz_input(lines)
        fd, out_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            build_sample(gz_path, out_path, lang=lang, max_rows=max_rows)
            with open(out_path, newline="", encoding="utf-8") as f:
                return list(csv.reader(f))
        finally:
            os.unlink(gz_path)
            os.unlink(out_path)

    def _assertion(self, relation: str, src: str, tgt: str) -> str:
        uri = f"/a/[{relation},{src},{tgt}]"
        meta = '{"weight": 1.0}'
        return f"{uri}\t{relation}\t{src}\t{tgt}\t{meta}"

    def test_keeps_english_is_a(self):
        line = self._assertion("/r/IsA", "/c/en/apple/n", "/c/en/fruit/n")
        rows = self._run([line])
        assert rows[0] == ["source", "relation_type", "target"]
        assert ["apple", "is_a", "fruit"] in rows

    def test_skips_non_english_source(self):
        line = self._assertion("/r/IsA", "/c/de/apfel", "/c/en/fruit")
        rows = self._run([line])
        assert len(rows) == 1  # header only

    def test_skips_non_english_target(self):
        line = self._assertion("/r/IsA", "/c/en/apple", "/c/de/frucht")
        rows = self._run([line])
        assert len(rows) == 1

    def test_skips_unrecognised_relation(self):
        line = self._assertion("/r/Antonym", "/c/en/hot", "/c/en/cold")
        rows = self._run([line])
        assert len(rows) == 1

    def test_deduplicates_rows(self):
        line = self._assertion("/r/IsA", "/c/en/apple", "/c/en/fruit")
        rows = self._run([line, line, line])
        data = [r for r in rows if r != ["source", "relation_type", "target"]]
        assert len(data) == 1

    def test_max_rows_respected(self):
        lines = [
            self._assertion("/r/IsA", f"/c/en/thing{i}", "/c/en/stuff")
            for i in range(20)
        ]
        rows = self._run(lines, max_rows=5)
        data = [r for r in rows if r != ["source", "relation_type", "target"]]
        assert len(data) == 5

    def test_relation_name_mapped(self):
        cases = [
            ("/r/IsA",         "is_a"),
            ("/r/PartOf",      "part_of"),
            ("/r/HasA",        "has_a"),
            ("/r/UsedFor",     "used_for"),
            ("/r/CapableOf",   "capable_of"),
            ("/r/Causes",      "causes"),
            ("/r/HasProperty", "has_property"),
            ("/r/AtLocation",  "at_location"),
            ("/r/MadeOf",      "made_of"),
        ]
        for cn_rel, expected in cases:
            line = self._assertion(cn_rel, "/c/en/a", "/c/en/b")
            rows = self._run([line])
            data = [r for r in rows if r[0] != "source"]
            assert data[0][1] == expected, f"{cn_rel} should map to {expected}"

    def test_node_uri_stripped_to_concept(self):
        line = self._assertion("/r/IsA", "/c/en/fire_engine/n", "/c/en/vehicle")
        rows = self._run([line])
        data = [r for r in rows if r[0] != "source"]
        assert data[0][0] == "fire_engine"
        assert data[0][2] == "vehicle"

    def test_skips_self_loops(self):
        line = self._assertion("/r/IsA", "/c/en/apple", "/c/en/apple")
        rows = self._run([line])
        assert len(rows) == 1

    def test_output_has_header(self):
        rows = self._run([])
        assert rows[0] == ["source", "relation_type", "target"]
